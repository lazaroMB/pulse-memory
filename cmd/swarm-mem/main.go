package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"pulse/internal/agent"
	"pulse/internal/consolidation"
	"pulse/internal/memory"
	"pulse/internal/privacy"
)

type Server struct {
	Store      *memory.PGStore
	Gemini     *agent.GeminiClient
	Filter     *privacy.LocalPrivacyFilter
	WorkerPool *consolidation.WorkerPool
}

type ChatRequest struct {
	SessionID string `json:"session_id"`
	EntityID  string `json:"entity_id"` // Represents the user or object this memory belongs to
	AgentRole string `json:"agent_role"`
	Message   string `json:"message"`
}

type ChatResponse struct {
	Reply     string        `json:"reply"`
	FactsUsed []memory.Fact `json:"facts_used"`
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

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatal("DATABASE_URL environment variable is required.")
	}

	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" || apiKey == "your_gemini_api_key_here" {
		log.Fatal("GEMINI_API_KEY environment variable is required.")
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 2. Initialize database store using sqlx
	log.Println("Connecting to database...")
	store, err := memory.NewPGStore(dbURL)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer store.Close()

	// Initialize database schemas
	log.Println("Initializing database schemas...")
	if err := store.InitSchema(ctx); err != nil {
		log.Fatalf("Database schema initialization failed: %v", err)
	}

	// 3. Initialize Google Gemini Client
	genModelName := os.Getenv("GEMINI_GENERATION_MODEL")
	if genModelName == "" {
		genModelName = "gemini-1.5-flash"
	}

	embedModelName := os.Getenv("GEMINI_EMBEDDING_MODEL")
	if embedModelName == "" {
		embedModelName = "text-embedding-004"
	}

	log.Printf("Initializing Google Gemini API Client (Gen: %s, Embed: %s)...", genModelName, embedModelName)
	gemini, err := agent.NewGeminiClient(ctx, apiKey, genModelName, embedModelName)
	if err != nil {
		log.Fatalf("Gemini client initialization failed: %v", err)
	}
	defer gemini.Close()

	// 4. Initialize components and background workers
	filter := privacy.NewLocalPrivacyFilter()
	workerPool := consolidation.NewWorkerPool(store, gemini, 100, 3) // queue size: 100, workers: 3
	workerPool.Start(ctx)
	defer workerPool.Stop()

	server := &Server{
		Store:      store,
		Gemini:     gemini,
		Filter:     filter,
		WorkerPool: workerPool,
	}

	// 5. Define routes
	mux := http.NewServeMux()
	mux.HandleFunc("/chat", server.handleChat)
	mux.HandleFunc("/relation", server.handleRelation)
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
	queryVector, err := s.Gemini.GenerateEmbedding(ctx, cleanMessage)
	if err != nil {
		// Log embedding error and fall back to keyword search (non-vector)
		log.Printf("Embedding generation failed, falling back to basic metadata retrieval: %v", err)
		queryVector = nil
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

	// 4. Validate Access (Role-Based Access Control)
	var filteredFacts []memory.Fact
	for _, fact := range retrievedFacts {
		if s.Filter.ValidateAccess(ctx, req.AgentRole, &fact) {
			filteredFacts = append(filteredFacts, fact)
		}
	}

	// 5. Generate Answer via Gemini using the retrieved facts
	reply, err := s.Gemini.GenerateAnswer(ctx, cleanMessage, filteredFacts)
	if err != nil {
		log.Printf("LLM generation error: %v", err)
		http.Error(w, "Failed to generate answer from model", http.StatusInternalServerError)
		return
	}

	// 6. Push event to the Asynchronous Consolidation queue (The Sleep-Time Pattern)
	s.WorkerPool.JobQueue <- consolidation.InteractionLog{
		SessionID: sessionID,
		EntityID:  entityID,
		Sender:    req.AgentRole,
		Message:   cleanMessage,
		Timestamp: time.Now(),
	}

	// Respond to the caller immediately
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ChatResponse{
		Reply:     reply,
		FactsUsed: filteredFacts,
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


