package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	cache_adapter "pulse/internal/adapter/cache/redis"
	chatmem_adapter "pulse/internal/adapter/chatmem"
	controller_adapter "pulse/internal/adapter/controller/http"
	llm_adapter "pulse/internal/adapter/llm"
	privacy_adapter "pulse/internal/adapter/privacy"
	repo_adapter "pulse/internal/adapter/repository/arcadedb"
	chat_usecase "pulse/internal/usecase/chat"
	consolidation_usecase "pulse/internal/usecase/consolidation"
	doc_usecase "pulse/internal/usecase/document"
	relation_usecase "pulse/internal/usecase/relation"
)

func main() {
	log.Println("Starting Multi-Agent Swarm Memory Server (Clean Architecture)...")

	// 1. Load environment variables
	_ = godotenv.Load()
	if execPath, err := os.Executable(); err == nil {
		execDir := filepath.Dir(execPath)
		_ = godotenv.Load(filepath.Join(execDir, ".env"))
		_ = godotenv.Load(filepath.Join(filepath.Dir(execDir), ".env"))
	}

	arcadeURL := os.Getenv("ARCADEDB_URL")
	if arcadeURL == "" {
		arcadeURL = "http://localhost:2480"
	}

	arcadeDB := os.Getenv("ARCADEDB_DATABASE")
	if arcadeDB == "" {
		arcadeDB = "pulse"
	}

	arcadeUser := os.Getenv("ARCADEDB_USERNAME")
	if arcadeUser == "" {
		arcadeUser = "root"
	}

	arcadePass := os.Getenv("ARCADEDB_PASSWORD")
	if arcadePass == "" {
		arcadePass = "playwithdata"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 2. Initialize database store using the adapter
	cfg := repo_adapter.Config{
		URL:      arcadeURL,
		Database: arcadeDB,
		Username: arcadeUser,
		Password: arcadePass,
	}

	log.Printf("Connecting to ArcadeDB database: %s at %s...", arcadeDB, arcadeURL)
	store, err := repo_adapter.NewMemoryStore(cfg)
	if err != nil {
		log.Fatalf("Database connection failed: %v", err)
	}
	defer store.Close()

	// Initialize database schemas
	log.Println("Initializing database schemas...")
	if err := store.InitSchema(ctx); err != nil {
		log.Fatalf("Database schema initialization failed: %v", err)
	}

	// 3. Initialize LLM Service using the factory adapter
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

	llmCfg := llm_adapter.Config{
		Provider:       llmProvider,
		APIKey:         llmAPIKey,
		GenModelName:   genModelName,
		EmbedModelName: embedModelName,
	}

	log.Printf("Initializing LLM Service (Provider: %s, Gen: %s, Embed: %s)...", llmCfg.Provider, llmCfg.GenModelName, llmCfg.EmbedModelName)
	rawLLMClient, err := llm_adapter.NewLLMService(ctx, llmCfg)
	if err != nil {
		log.Fatalf("LLM client initialization failed: %v", err)
	}
	llmClient := llm_adapter.NewBatchingLLMClient(ctx, rawLLMClient)
	defer llmClient.Close()

	// 4. Initialize Short-Term Chat Memory using the adapter
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
	chatMemCfg := chatmem_adapter.ChatMemoryConfig{
		Provider: chatMemProvider,
		URL:      chatMemURL,
	}

	chatMemory, err := chatmem_adapter.NewChatMemory(chatMemCfg)
	if err != nil {
		log.Fatalf("Short-term chat memory connection failed: %v", err)
	}
	defer chatMemory.Close()

	// 5. Initialize Semantic Cache using the adapter
	semanticCacheThreshold := 0.95
	if val := os.Getenv("SEMANTIC_CACHE_THRESHOLD"); val != "" {
		if parsed, err := strconv.ParseFloat(val, 64); err == nil && parsed > 0 {
			semanticCacheThreshold = parsed
		}
	}
	log.Printf("Initializing Semantic Cache (Provider: %s, threshold: %.2f)...", chatMemProvider, semanticCacheThreshold)
	semanticCache, err := cache_adapter.NewSemanticCacheFactory(chatMemProvider, chatMemURL, semanticCacheThreshold)
	if err != nil {
		log.Fatalf("Semantic Cache initialization failed: %v", err)
	}
	defer semanticCache.Close()

	// 6. Initialize background workers & crons
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

	workerPool := consolidation_usecase.NewWorkerPool(store, chatMemory, llmClient, semanticCache, queueSize, maxWorkers)
	workerPool.Start(ctx)
	defer workerPool.Stop()

	// 7. Initialize Privacy Filter adapter
	privacyService := privacy_adapter.NewLocalPrivacyFilter()

	// 8. Initialize Use Cases
	chatUseCase := chat_usecase.NewChatUseCase(store, chatMemory, llmClient, semanticCache, privacyService, workerPool)
	relationUseCase := relation_usecase.NewRelationUseCase(store, semanticCache)
	documentUseCase := doc_usecase.NewDocumentUseCase(store, workerPool)

	// 9. Initialize HTTP Controllers
	controller := controller_adapter.NewController(chatUseCase, relationUseCase, documentUseCase, llmClient)

	// 10. Define HTTP routes
	mux := http.NewServeMux()
	controller.RegisterRoutes(mux)
	
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	httpServer := &http.Server{
		Addr:    ":" + port,
		Handler: mux,
	}

	// 11. Start server with Graceful Shutdown handling
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
