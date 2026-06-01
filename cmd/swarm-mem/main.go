package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"pulse/internal/agent"
	"pulse/internal/consolidation"
	"pulse/internal/document"
	"pulse/internal/memory"
	"pulse/internal/privacy"
)

type Server struct {
	Store      memory.MemoryStore
	ChatMemory memory.ChatMemory
	LLM        agent.LLMClient
	Filter     *privacy.LocalPrivacyFilter
	WorkerPool *consolidation.WorkerPool
	Cache      memory.SemanticCache
}

type ChatRequest struct {
	SessionID    string `json:"session_id"`
	EntityID     string `json:"entity_id"` // Represents the user or object this memory belongs to
	AgentRole    string `json:"agent_role"`
	Message      string `json:"message"`
	IncludeFacts bool   `json:"includeFacts"`
	IncludeFachs bool   `json:"includeFachs"` // Support typo variation in payload
}

type ChatResponse struct {
	ResponseMessage string        `json:"responseMessage"`
	EntityFacts     []memory.Fact `json:"entityFacts,omitempty"`
	DocumentFacts   []memory.Fact `json:"documentFacts,omitempty"`
}

type RelationRequest struct {
	SourceID string `json:"source_id"`
	TargetID string `json:"target_id"`
	Type     string `json:"type"`
}



func main() {
	log.Println("Starting Multi-Agent Swarm Memory Server...")

	// 1. Load environment variables
	_ = godotenv.Load()
	if execPath, err := os.Executable(); err == nil {
		execDir := filepath.Dir(execPath)
		_ = godotenv.Load(filepath.Join(execDir, ".env"))
		_ = godotenv.Load(filepath.Join(filepath.Dir(execDir), ".env"))
	}

	dbProvider := os.Getenv("DB_PROVIDER")
	if dbProvider == "" {
		dbProvider = "postgres"
	}

	dbURL := os.Getenv("DATABASE_URL")
	if dbProvider == "postgres" && dbURL == "" {
		log.Fatal("DATABASE_URL environment variable is required when DB_PROVIDER is 'postgres'.")
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 2. Initialize database store using the factory
	cfg := memory.Config{
		Provider:          dbProvider,
		PostgresURL:       dbURL,
		Neo4jURI:          os.Getenv("NEO4J_URI"),
		Neo4jUsername:     os.Getenv("NEO4J_USERNAME"),
		Neo4jPassword:     os.Getenv("NEO4J_PASSWORD"),
		FalkorDBURL:       os.Getenv("FALKORDB_URL"),
		FalkorDBGraphName: os.Getenv("FALKORDB_GRAPH_NAME"),
	}

	log.Printf("Connecting to database using provider: %s...", dbProvider)
	store, err := memory.NewMemoryStore(cfg)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer store.Close()

	// Initialize database schemas
	log.Println("Initializing database schemas...")
	if err := store.InitSchema(ctx); err != nil {
		log.Fatalf("Database schema initialization failed: %v", err)
	}

	// 3. Initialize LLM Client using the factory
	llmProvider := os.Getenv("LLM_PROVIDER")
	if llmProvider == "" {
		llmProvider = "gemini"
	}

	var llmAPIKey string
	var genModelName string
	var embedModelName string

	if llmProvider == "openai" {
		llmAPIKey = os.Getenv("OPENAI_API_KEY")
		if llmAPIKey == "" {
			llmAPIKey = os.Getenv("LLM_API_KEY")
		}
		genModelName = os.Getenv("OPENAI_GENERATION_MODEL")
		if genModelName == "" {
			genModelName = "gpt-4o-mini"
		}
		embedModelName = os.Getenv("OPENAI_EMBEDDING_MODEL")
		if embedModelName == "" {
			embedModelName = "text-embedding-3-small"
		}
	} else {
		// default to gemini
		llmAPIKey = os.Getenv("GEMINI_API_KEY")
		if llmAPIKey == "" {
			llmAPIKey = os.Getenv("LLM_API_KEY")
		}
		genModelName = os.Getenv("GEMINI_GENERATION_MODEL")
		if genModelName == "" {
			genModelName = "gemini-2.5-flash"
		}
		embedModelName = os.Getenv("GEMINI_EMBEDDING_MODEL")
		if embedModelName == "" {
			embedModelName = "text-embedding-004"
		}
	}

	if llmAPIKey == "" || llmAPIKey == "your_gemini_api_key_here" || llmAPIKey == "your_openai_api_key_here" {
		log.Fatalf("LLM API key is required (set LLM_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY).")
	}

	llmCfg := agent.Config{
		Provider:       llmProvider,
		APIKey:         llmAPIKey,
		GenModelName:   genModelName,
		EmbedModelName: embedModelName,
	}

	log.Printf("Initializing LLM Client (Provider: %s, Gen: %s, Embed: %s)...", llmCfg.Provider, llmCfg.GenModelName, llmCfg.EmbedModelName)
	rawLLMClient, err := agent.NewLLMClient(ctx, llmCfg)
	if err != nil {
		log.Fatalf("LLM client initialization failed: %v", err)
	}
	llmClient := agent.NewBatchingLLMClient(ctx, rawLLMClient)
	defer llmClient.Close()

	// 4. Initialize components and background workers
	filter := privacy.NewLocalPrivacyFilter()

	maxWorkers := 3
	if val := os.Getenv("CONSOLIDATION_MAX_WORKERS"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			maxWorkers = parsed
		}
	}

	queueSize := 100
	if val := os.Getenv("CONSOLIDATION_QUEUE_SIZE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			queueSize = parsed
		}
	}

	// 3b. Initialize Short-Term Chat Memory using the factory
	chatMemProvider := os.Getenv("CHAT_MEMORY_PROVIDER")
	if chatMemProvider == "" {
		chatMemProvider = "redis" // Default to redis
	}

	chatMemURL := os.Getenv("CHAT_MEMORY_URL")
	if chatMemURL == "" {
		chatMemURL = os.Getenv("REDIS_URL")
		if chatMemURL == "" {
			chatMemURL = os.Getenv("FALKORDB_URL")
			if chatMemURL == "" {
				chatMemURL = "localhost:6379"
			}
		}
	}

	log.Printf("Connecting to short-term chat memory using provider: %s at %s...", chatMemProvider, chatMemURL)
	chatMemCfg := memory.ChatMemoryConfig{
		Provider: chatMemProvider,
		URL:      chatMemURL,
	}

	chatMemory, err := memory.NewChatMemory(chatMemCfg)
	if err != nil {
		log.Fatalf("Short-term chat memory connection failed: %v", err)
	}
	defer chatMemory.Close()

	// 3c. Initialize Semantic Cache using the factory
	semanticCacheThreshold := 0.95
	if val := os.Getenv("SEMANTIC_CACHE_THRESHOLD"); val != "" {
		if parsed, err := strconv.ParseFloat(val, 64); err == nil && parsed > 0 {
			semanticCacheThreshold = parsed
		}
	}
	log.Printf("Initializing Semantic Cache (Provider: %s, threshold: %.2f)...", chatMemProvider, semanticCacheThreshold)
	semanticCache, err := memory.NewSemanticCacheFactory(chatMemProvider, chatMemURL, semanticCacheThreshold)
	if err != nil {
		log.Fatalf("Semantic Cache initialization failed: %v", err)
	}
	defer semanticCache.Close()

	workerPool := consolidation.NewWorkerPool(store, chatMemory, llmClient, queueSize, maxWorkers)
	workerPool.Start(ctx)
	defer workerPool.Stop()

	server := &Server{
		Store:      store,
		ChatMemory: chatMemory,
		LLM:        llmClient,
		Filter:     filter,
		WorkerPool: workerPool,
		Cache:      semanticCache,
	}

	// 5. Define routes
	mux := http.NewServeMux()
	mux.HandleFunc("/chat", server.handleChat)
	mux.HandleFunc("/relation", server.handleRelation)
	mux.HandleFunc("/ingest/file", server.handleIngestFile)
	mux.HandleFunc("/ingest/link", server.handleIngestLink)
	mux.HandleFunc("/documents/", server.handleGetDocument)
	mux.HandleFunc("/search/documents", server.handleSearchDocuments)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	httpServer := &http.Server{
		Addr:    ":" + port,
		Handler: mux,
	}

	// 6. Start server with Graceful Shutdown handling
	go func() {
		log.Printf("Swarm Memory Server is running on port %s", port)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("HTTP server failure: %v", err)
		}
	}()

	// Wait for termination signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down gracefully...")
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP shutdown error: %v", err)
	}

	log.Println("Server stopped successfully.")
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
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

	// Default entities if not provided or invalid
	entityID, err := uuid.Parse(req.EntityID)
	if err != nil {
		// Use a fallback deterministic namespace UUID for ease of use
		entityID = uuid.NewMD5(uuid.NameSpaceDNS, []byte("default-user"))
	}

	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = uuid.New().String()
	}

	ctx := r.Context()

	// 1. Apply Local Privacy Filter (PII Scrubbing Proxy)
	cleanMessage, err := s.Filter.ScrubText(ctx, req.Message)
	if err != nil {
		http.Error(w, "Privacy scrubbing failed", http.StatusInternalServerError)
		return
	}

	// 2. Generate Vector Embedding of the Query Text for similarity search
	queryVector, err := s.LLM.GenerateEmbedding(ctx, cleanMessage)
	if err != nil {
		// Log embedding error and fall back to keyword search (non-vector)
		log.Printf("Embedding generation failed, falling back to basic metadata retrieval: %v", err)
		queryVector = nil
	}

	// 2b. Check Semantic Cache for hits (Cos similarity >= 0.95)
	if len(queryVector) > 0 {
		cachedReply, found, err := s.Cache.Get(ctx, queryVector)
		if err == nil && found {
			log.Printf("[Semantic Cache HIT] Returning cached response directly for query: %s", cleanMessage)
			// Append cache turn to short-term chat memory to keep session state updated
			userMsg := memory.ChatMessage{
				Role:      req.AgentRole,
				Content:   cleanMessage,
				Timestamp: time.Now(),
			}
			_ = s.ChatMemory.AppendMessage(ctx, sessionID, userMsg)

			assistantMsg := memory.ChatMessage{
				Role:      "assistant",
				Content:   cachedReply,
				Timestamp: time.Now(),
			}
			_ = s.ChatMemory.AppendMessage(ctx, sessionID, assistantMsg)

			// Respond to caller immediately
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(ChatResponse{
				ResponseMessage: cachedReply,
			})
			return
		}
	}

	// 3. Search database for active facts (Hybrid Search)
	searchQuery := &memory.MemorySearchQuery{
		QueryText:     cleanMessage,
		QueryVector:   queryVector,
		TargetEntity:  entityID,
		RequiredScope: req.AgentRole,
		MaxResults:    5,
	}

	retrievedFacts, err := s.Store.SearchHybrid(ctx, searchQuery)
	if err != nil {
		log.Printf("Database retrieval error: %v", err)
		retrievedFacts = []memory.Fact{}
	}

	// 3b. Search database for active general/shared knowledge facts extracted from documents
	sharedEntityID := uuid.NewMD5(uuid.NameSpaceDNS, []byte("shared-general-knowledge"))
	sharedSearchQuery := &memory.MemorySearchQuery{
		QueryText:     cleanMessage,
		QueryVector:   queryVector,
		TargetEntity:  sharedEntityID,
		RequiredScope: req.AgentRole,
		MaxResults:    5,
	}

	sharedFacts, err := s.Store.SearchHybrid(ctx, sharedSearchQuery)
	if err != nil {
		log.Printf("Shared database retrieval error: %v", err)
	} else {
		retrievedFacts = append(retrievedFacts, sharedFacts...)
	}

	// 4. Validate Access (Role-Based Access Control)
	var filteredFacts []memory.Fact
	for _, fact := range retrievedFacts {
		if s.Filter.ValidateAccess(ctx, req.AgentRole, &fact) {
			filteredFacts = append(filteredFacts, fact)
		}
	}

	// 4b. Perform vector similarity search over raw document chunks (Standard RAG)
	if len(queryVector) > 0 {
		chunks, err := s.Store.SearchDocumentChunks(ctx, queryVector, 3)
		if err == nil {
			for _, chunk := range chunks {
				docTitle := "Document"
				if chunk.Metadata != nil && chunk.Metadata["title"] != "" {
					docTitle = chunk.Metadata["title"]
				} else if chunk.Metadata != nil && chunk.Metadata["source_url"] != "" {
					docTitle = chunk.Metadata["source_url"]
				}
				filteredFacts = append(filteredFacts, memory.Fact{
					ID:              chunk.ID,
					EntityID:        sharedEntityID,
					Attribute:       "document_chunk",
					Value:           fmt.Sprintf("[Source: %s] %s", docTitle, chunk.Content),
					ConfidenceScore: 1.0,
					SourceAgent:     "document",
				})
			}
		} else {
			log.Printf("Document chunks retrieval error: %v", err)
		}

		// 4c. Retrieve community summaries for Macro-Narrative GraphRAG context
		communities, err := s.Store.SearchCommunitySummaries(ctx, queryVector, 2)
		if err == nil {
			for _, comm := range communities {
				filteredFacts = append(filteredFacts, memory.Fact{
					ID:              comm.ID,
					EntityID:        sharedEntityID,
					Attribute:       "macro_narrative",
					Value:           fmt.Sprintf("[Macro Narrative Summary: %s] %s", comm.Name, comm.Summary),
					ConfidenceScore: 1.0,
					SourceAgent:     "community_graphrag",
				})
			}
		} else {
			log.Printf("Community summaries retrieval error: %v", err)
		}
	}

	// 4b. Retrieve 1-hop and 2-hop active relationships for Graph Context (Graph RAG)
	rels, err := s.Store.GetActiveRelations(ctx, entityID)
	if err == nil && len(rels) > 0 {
		entityIDs := make(map[uuid.UUID]bool)
		entityIDs[entityID] = true
		for _, r := range rels {
			entityIDs[r.SourceID] = true
			entityIDs[r.TargetID] = true
		}

		// Fetch 2-hop relations for connected entities
		var hop2Rels []memory.Relation
		for _, r := range rels {
			otherID := r.TargetID
			if otherID != entityID {
				otherRels, err := s.Store.GetActiveRelations(ctx, otherID)
				if err == nil {
					for _, or := range otherRels {
						entityIDs[or.SourceID] = true
						entityIDs[or.TargetID] = true
						hop2Rels = append(hop2Rels, or)
					}
				}
			}
		}
		rels = append(rels, hop2Rels...)

		// Resolve human-readable names for all entity IDs
		nameMap := make(map[uuid.UUID]string)
		nameMap[entityID] = "user" // Speaker is mapped to "user"

		for eID := range entityIDs {
			if eID == entityID {
				continue
			}
			// Search for "name" fact of this entity to map UUID -> human name
			facts, err := s.Store.SearchHybrid(ctx, &memory.MemorySearchQuery{
				TargetEntity: eID,
				MaxResults:   20,
			})
			if err == nil {
				for _, f := range facts {
					if f.Attribute == "name" {
						nameMap[eID] = f.Value
						break
					}
				}
			}
			if _, exists := nameMap[eID]; !exists {
				if len(facts) > 0 {
					nameMap[eID] = facts[0].Value
				} else {
					nameMap[eID] = eID.String()[:8]
				}
			}
		}

		// Deduplicate and format relations as artificial Graph Relation facts
		seenRels := make(map[string]bool)
		for _, r := range rels {
			srcName := nameMap[r.SourceID]
			tgtName := nameMap[r.TargetID]
			if srcName == "" || tgtName == "" || srcName == tgtName {
				continue
			}
			relKey := fmt.Sprintf("%s-%s-%s", srcName, r.Type, tgtName)
			if !seenRels[relKey] {
				seenRels[relKey] = true
				filteredFacts = append(filteredFacts, memory.Fact{
					Attribute:       "graph_relation",
					Value:           fmt.Sprintf("Graph Relation: %s is linked to %s via %s", srcName, tgtName, r.Type),
					ConfidenceScore: 1.0,
				})
			}
		}
	}

	// 4c. Retrieve short-term chat history
	history, err := s.ChatMemory.GetSessionHistory(ctx, sessionID, 15) // Fetch last 15 messages for context
	if err != nil {
		log.Printf("Failed to retrieve chat history for session %s: %v", sessionID, err)
		history = []memory.ChatMessage{}
	}

	// 5. Generate Answer via LLM using the retrieved facts and short-term history
	reply, err := s.LLM.GenerateAnswer(ctx, cleanMessage, history, filteredFacts)
	if err != nil {
		log.Printf("LLM generation error: %v", err)
		http.Error(w, "Failed to generate answer from model", http.StatusInternalServerError)
		return
	}

	// 5b. Register response in semantic cache asynchronously
	if len(queryVector) > 0 && err == nil {
		go func(qText string, qVec []float32, rText string) {
			_ = s.Cache.Set(context.Background(), qText, qVec, rText)
		}(cleanMessage, queryVector, reply)
	}

	// 5b. Append interaction turn to short-term chat memory
	userMsg := memory.ChatMessage{
		Role:      req.AgentRole,
		Content:   cleanMessage,
		Timestamp: time.Now(),
	}
	if err := s.ChatMemory.AppendMessage(ctx, sessionID, userMsg); err != nil {
		log.Printf("Failed to append user message to chat memory: %v", err)
	}

	assistantMsg := memory.ChatMessage{
		Role:      "assistant",
		Content:   reply,
		Timestamp: time.Now(),
	}
	if err := s.ChatMemory.AppendMessage(ctx, sessionID, assistantMsg); err != nil {
		log.Printf("Failed to append assistant message to chat memory: %v", err)
	}

	// 6. Push event to the Asynchronous Consolidation queue (The Sleep-Time Pattern)
	s.WorkerPool.JobQueue <- consolidation.InteractionLog{
		SessionID: sessionID,
		EntityID:  entityID,
		Sender:    req.AgentRole,
		Message:   cleanMessage,
		Timestamp: time.Now(),
	}

	var entityFacts []memory.Fact
	var documentFacts []memory.Fact

	if req.IncludeFacts || req.IncludeFachs {
		entityFacts = []memory.Fact{}
		documentFacts = []memory.Fact{}
		for _, fact := range filteredFacts {
			if fact.SourceAgent == "document" || fact.Attribute == "document_chunk" {
				documentFacts = append(documentFacts, fact)
			} else {
				entityFacts = append(entityFacts, fact)
			}
		}
	}

	// Respond to the caller immediately
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ChatResponse{
		ResponseMessage: reply,
		EntityFacts:     entityFacts,
		DocumentFacts:   documentFacts,
	})
}

func (s *Server) handleRelation(w http.ResponseWriter, r *http.Request) {
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

	ctx := r.Context()
	relation := &memory.Relation{
		ID:        uuid.New(),
		SourceID:  sourceUUID,
		TargetID:  targetUUID,
		Type:      req.Type,
		ValidFrom: time.Now(),
	}

	if err := s.Store.InsertRelation(ctx, relation); err != nil {
		log.Printf("Failed to insert relation: %v", err)
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

func (s *Server) handleIngestFile(w http.ResponseWriter, r *http.Request) {
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

	var srcType document.SourceType
	ext := strings.ToLower(filepath.Ext(header.Filename))
	switch ext {
	case ".pdf":
		srcType = document.SourcePDF
	case ".md", ".markdown":
		srcType = document.SourceMarkdown
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

	ctx := r.Context()
	
	doc := &document.Document{
		ID:         docID,
		Title:      title,
		SourceType: srcType,
		FilePath:   tempFilePath,
		Status:     document.StatusPending,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Metadata:   make(map[string]string),
	}

	if err := s.Store.InsertDocument(ctx, doc); err != nil {
		log.Printf("Failed to insert document metadata: %v", err)
		http.Error(w, "Failed to insert document record", http.StatusInternalServerError)
		return
	}

	entityIDStr := r.FormValue("entity_id")
	if entityIDStr != "" {
		if entityUUID, err := uuid.Parse(entityIDStr); err == nil {
			_ = s.Store.LinkDocumentToAuthor(ctx, docID, entityUUID)
			doc.Metadata["target_entity"] = entityIDStr
		}
	}

	s.WorkerPool.DocumentQueue <- consolidation.DocumentJob{
		DocumentID: docID,
		FilePath:   tempFilePath,
		SourceType: srcType,
		Metadata:   doc.Metadata,
	}

	resp := IngestResponse{
		DocumentID: docID.String(),
		Status:     string(document.StatusPending),
		Message:    "Document file uploaded and queued for processing",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleIngestLink(w http.ResponseWriter, r *http.Request) {
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

	var srcType document.SourceType
	switch req.SourceType {
	case "web_page":
		srcType = document.SourceWebPage
	case "google_docs":
		srcType = document.SourceGoogleDocs
	default:
		if strings.Contains(req.URL, "docs.google.com") {
			srcType = document.SourceGoogleDocs
		} else {
			srcType = document.SourceWebPage
		}
	}

	title := req.Title
	if title == "" {
		title = req.URL
	}

	docID := uuid.New()
	ctx := r.Context()

	doc := &document.Document{
		ID:         docID,
		Title:      title,
		SourceType: srcType,
		SourceURL:  req.URL,
		Status:     document.StatusPending,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Metadata:   make(map[string]string),
	}

	if err := s.Store.InsertDocument(ctx, doc); err != nil {
		log.Printf("Failed to insert document metadata: %v", err)
		http.Error(w, "Failed to insert document record", http.StatusInternalServerError)
		return
	}

	if req.EntityID != "" {
		if entityUUID, err := uuid.Parse(req.EntityID); err == nil {
			_ = s.Store.LinkDocumentToAuthor(ctx, docID, entityUUID)
			doc.Metadata["target_entity"] = req.EntityID
		}
	}

	s.WorkerPool.DocumentQueue <- consolidation.DocumentJob{
		DocumentID: docID,
		URL:        req.URL,
		SourceType: srcType,
		Metadata:   doc.Metadata,
	}

	resp := IngestResponse{
		DocumentID: docID.String(),
		Status:     string(document.StatusPending),
		Message:    "Link queued for background fetch and parsing",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleGetDocument(w http.ResponseWriter, r *http.Request) {
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

	ctx := r.Context()
	doc, err := s.Store.GetDocument(ctx, docUUID)
	if err != nil {
		log.Printf("Failed to get document status: %v", err)
		http.Error(w, "Document not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(doc)
}

func (s *Server) handleSearchDocuments(w http.ResponseWriter, r *http.Request) {
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

	queryVector, err := s.LLM.GenerateEmbedding(ctx, req.Query)
	if err != nil {
		log.Printf("Failed to generate embedding for query: %v", err)
		http.Error(w, "Failed to generate semantic search vector", http.StatusInternalServerError)
		return
	}

	chunks, err := s.Store.SearchDocumentChunks(ctx, queryVector, limit)
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



