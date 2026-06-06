package http

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
	"pulse/internal/domain/entity"
	"pulse/internal/usecase/chat"
	"pulse/internal/usecase/document"
	"pulse/internal/usecase/ports"
	"pulse/internal/usecase/relation"
)

type Controller struct {
	chatUseCase     *chat.ChatUseCase
	relationUseCase *relation.RelationUseCase
	documentUseCase *document.DocumentUseCase
	llm             ports.LLMService
}

func NewController(
	chatUseCase *chat.ChatUseCase,
	relationUseCase *relation.RelationUseCase,
	documentUseCase *document.DocumentUseCase,
	llm ports.LLMService,
) *Controller {
	return &Controller{
		chatUseCase:     chatUseCase,
		relationUseCase: relationUseCase,
		documentUseCase: documentUseCase,
		llm:             llm,
	}
}

// RegisterRoutes binds controllers to HTTP mux endpoints
func (c *Controller) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/chat", c.HandleChat)
	mux.HandleFunc("/relation", c.HandleRelation)
	mux.HandleFunc("/ingest/file", c.HandleIngestFile)
	mux.HandleFunc("/ingest/link", c.HandleIngestLink)
	mux.HandleFunc("/documents/", c.HandleGetDocument)
	mux.HandleFunc("/search/documents", c.HandleSearchDocuments)
}

type ChatRequest struct {
	SessionID    string `json:"session_id"`
	EntityID     string `json:"entity_id"`
	AgentRole    string `json:"agent_role"`
	Message      string `json:"message"`
	IncludeFacts bool   `json:"includeFacts"`
	IncludeFachs bool   `json:"includeFachs"`
}

type ChatResponse struct {
	ResponseMessage string        `json:"responseMessage"`
	EntityFacts     []entity.Fact `json:"entityFacts,omitempty"`
	DocumentFacts   []entity.Fact `json:"documentFacts,omitempty"`
}

func (c *Controller) HandleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	if req.Message == "" {
		http.Error(w, "Message parameter is required", http.StatusBadRequest)
		return
	}

	entityID, err := uuid.Parse(req.EntityID)
	if err != nil {
		// Fallback deterministic namespace UUID for easy testing
		entityID = uuid.NewMD5(uuid.NameSpaceDNS, []byte("default-user"))
	}

	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = uuid.New().String()
	}

	input := chat.ChatInput{
		SessionID:    sessionID,
		EntityID:     entityID,
		AgentRole:    req.AgentRole,
		Message:      req.Message,
		IncludeFacts: req.IncludeFacts || req.IncludeFachs,
	}

	output, err := c.chatUseCase.Execute(r.Context(), input)
	if err != nil {
		log.Printf("Chat use case execution error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ChatResponse{
		ResponseMessage: output.ResponseMessage,
		EntityFacts:     output.EntityFacts,
		DocumentFacts:   output.DocumentFacts,
	})
}

type RelationRequest struct {
	SourceID   string `json:"source_id"`
	TargetID   string `json:"target_id"`
	Type       string `json:"type"`
	AgentOwner string `json:"agent_owner"`
}

func (c *Controller) HandleRelation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RelationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	sourceUUID, err := uuid.Parse(req.SourceID)
	if err != nil {
		http.Error(w, "Invalid source_id UUID", http.StatusBadRequest)
		return
	}

	targetUUID, err := uuid.Parse(req.TargetID)
	if err != nil {
		http.Error(w, "Invalid target_id UUID", http.StatusBadRequest)
		return
	}

	if req.Type == "" {
		http.Error(w, "Relationship type is required", http.StatusBadRequest)
		return
	}

	agentOwner := uuid.Nil
	if req.AgentOwner != "" {
		if parsed, err := uuid.Parse(req.AgentOwner); err == nil {
			agentOwner = parsed
		}
	}

	input := relation.RelationInput{
		SourceID:   sourceUUID,
		TargetID:   targetUUID,
		Type:       req.Type,
		AgentOwner: agentOwner,
	}

	if err := c.relationUseCase.Execute(r.Context(), input); err != nil {
		log.Printf("Relation use case execution error: %v", err)
		http.Error(w, "Failed to store relation", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	w.Write([]byte(`{"status":"success"}`))
}

type IngestLinkRequest struct {
	URL        string `json:"url"`
	Title      string `json:"title"`
	SourceType string `json:"source_type"`
	EntityID   string `json:"entity_id"`
}

type IngestResponse struct {
	DocumentID string `json:"document_id"`
	Status     string `json:"status"`
	Message    string `json:"message"`
}

func (c *Controller) HandleIngestFile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(50 << 20)
	if err != nil {
		http.Error(w, "Failed to parse multipart form", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Missing file parameter", http.StatusBadRequest)
		return
	}
	defer file.Close()

	title := r.FormValue("title")
	if title == "" {
		title = header.Filename
	}

	var srcType entity.SourceType
	ext := strings.ToLower(filepath.Ext(header.Filename))
	switch ext {
	case ".pdf":
		srcType = entity.SourcePDF
	case ".md", ".markdown":
		srcType = entity.SourceMarkdown
	default:
		http.Error(w, "Unsupported file format. Must be PDF or Markdown.", http.StatusBadRequest)
		return
	}

	tempDir := filepath.Join(os.TempDir(), "pulse-uploads")
	_ = os.MkdirAll(tempDir, 0755)

	docID := uuid.New()
	tempFilePath := filepath.Join(tempDir, fmt.Sprintf("%s%s", docID, ext))
	tempFile, err := os.Create(tempFilePath)
	if err != nil {
		log.Printf("Failed to create temporary file: %v", err)
		http.Error(w, "Failed to initialize upload storage", http.StatusInternalServerError)
		return
	}
	defer tempFile.Close()

	_, err = io.Copy(tempFile, file)
	if err != nil {
		log.Printf("Failed to save temporary file: %v", err)
		http.Error(w, "Failed to store uploaded file", http.StatusInternalServerError)
		return
	}

	input := document.IngestFileInput{
		Title:      title,
		SourceType: srcType,
		FilePath:   tempFilePath,
		EntityID:   r.FormValue("entity_id"),
	}

	createdID, err := c.documentUseCase.IngestFile(r.Context(), input)
	if err != nil {
		log.Printf("Document use case IngestFile error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	resp := IngestResponse{
		DocumentID: createdID.String(),
		Status:     string(entity.StatusPending),
		Message:    "Document file uploaded and queued for processing",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(resp)
}

func (c *Controller) HandleIngestLink(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req IngestLinkRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	if req.URL == "" {
		http.Error(w, "URL is required", http.StatusBadRequest)
		return
	}

	var srcType entity.SourceType
	switch req.SourceType {
	case "web_page":
		srcType = entity.SourceWebPage
	case "google_docs":
		srcType = entity.SourceGoogleDocs
	default:
		if strings.Contains(req.URL, "docs.google.com") {
			srcType = entity.SourceGoogleDocs
		} else {
			srcType = entity.SourceWebPage
		}
	}

	title := req.Title
	if title == "" {
		title = req.URL
	}

	input := document.IngestLinkInput{
		URL:        req.URL,
		Title:      title,
		SourceType: srcType,
		EntityID:   req.EntityID,
	}

	createdID, err := c.documentUseCase.IngestLink(r.Context(), input)
	if err != nil {
		log.Printf("Document use case IngestLink error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	resp := IngestResponse{
		DocumentID: createdID.String(),
		Status:     string(entity.StatusPending),
		Message:    "Link queued for background fetch and parsing",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(resp)
}

func (c *Controller) HandleGetDocument(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	idStr := r.URL.Path[len("/documents/"):]
	docUUID, err := uuid.Parse(idStr)
	if err != nil {
		http.Error(w, "Invalid document UUID format", http.StatusBadRequest)
		return
	}

	doc, err := c.documentUseCase.GetDocument(r.Context(), docUUID)
	if err != nil {
		log.Printf("Document use case GetDocument error: %v", err)
		http.Error(w, "Document not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(doc)
}

type SearchDocumentsRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

type SearchDocumentsResponse struct {
	Results []SearchChunkResult `json:"results"`
}

type SearchChunkResult struct {
	DocumentID string  `json:"document_id"`
	ChunkIndex int     `json:"chunk_index"`
	Content    string  `json:"content"`
	Score      float32 `json:"score"`
}

func (c *Controller) HandleSearchDocuments(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SearchDocumentsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "Query text is required", http.StatusBadRequest)
		return
	}

	limit := req.Limit
	if limit <= 0 {
		limit = 5
	}

	ctx := r.Context()
	queryVector, err := c.llm.GenerateEmbedding(ctx, req.Query)
	if err != nil {
		log.Printf("Failed to generate embedding for query: %v", err)
		http.Error(w, "Failed to generate semantic search vector", http.StatusInternalServerError)
		return
	}

	chunks, err := c.documentUseCase.SearchDocuments(ctx, queryVector, limit)
	if err != nil {
		log.Printf("Failed to search document chunks: %v", err)
		http.Error(w, "Semantic vector search failed", http.StatusInternalServerError)
		return
	}

	results := make([]SearchChunkResult, len(chunks))
	for i, chunk := range chunks {
		results[i] = SearchChunkResult{
			DocumentID: chunk.DocumentID.String(),
			ChunkIndex: chunk.ChunkIndex,
			Content:    chunk.Content,
			Score:      1.0,
		}
	}

	resp := SearchDocumentsResponse{
		Results: results,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
