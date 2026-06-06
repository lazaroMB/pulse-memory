package ports

import (
	"context"

	"github.com/google/uuid"
	"pulse/internal/domain/entity"
)

// MemoryRepository defines the boundary for database writes, graph traversals, and hybrid lookups.
type MemoryRepository interface {
	InitSchema(ctx context.Context) error
	InsertFact(ctx context.Context, fact *entity.Fact, vector []float32) error
	SearchHybrid(ctx context.Context, query *entity.MemorySearchQuery) ([]entity.Fact, error)
	DeactivateFact(ctx context.Context, factID uuid.UUID) error
	InsertRelation(ctx context.Context, relation *entity.Relation) error
	GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]entity.Relation, error)
	Close() error

	// Document Ingestion & Graph Knowledge Extensions
	InsertDocument(ctx context.Context, doc *entity.Document) error
	GetDocument(ctx context.Context, id uuid.UUID) (*entity.Document, error)
	UpdateDocumentStatus(ctx context.Context, docID uuid.UUID, status entity.IngestionStatus, errMsg string) error
	InsertDocumentChunks(ctx context.Context, chunks []entity.DocumentChunk, embeddings [][]float32) error
	SearchDocumentChunks(ctx context.Context, queryVector []float32, limit int) ([]entity.DocumentChunk, error)
	LinkDocumentToAuthor(ctx context.Context, docID uuid.UUID, authorID uuid.UUID) error
	LinkDocumentToTopic(ctx context.Context, docID uuid.UUID, topicName string) error
	LinkFactToSource(ctx context.Context, factID uuid.UUID, docID uuid.UUID, chunkID uuid.UUID) error
	InsertFactWithProvenance(ctx context.Context, fact *entity.Fact, vector []float32, docID uuid.UUID, chunkID uuid.UUID) error

	// Entity Resolution Methods
	GetAllEntities(ctx context.Context) ([]entity.EntityNode, error)
	MergeEntities(ctx context.Context, canonicalID, duplicateID uuid.UUID) error

	// Community Summaries for GraphRAG
	InsertCommunitySummary(ctx context.Context, id uuid.UUID, name string, summary string, embedding []float32, entities []uuid.UUID) error
	SearchCommunitySummaries(ctx context.Context, queryVector []float32, limit int) ([]entity.CommunitySummary, error)

	// Batch optimized relations and names
	GetActiveRelationsBatch(ctx context.Context, entityIDs []uuid.UUID) ([]entity.Relation, error)
	GetEntityNamesBatch(ctx context.Context, entityIDs []uuid.UUID) (map[uuid.UUID]string, error)
}

// ChatMemoryRepository defines the boundary for storing and retrieving conversational short-term memory.
type ChatMemoryRepository interface {
	AppendMessage(ctx context.Context, sessionID string, msg entity.ChatMessage) error
	GetSessionHistory(ctx context.Context, sessionID string, limit int) ([]entity.ChatMessage, error)
	ClearSession(ctx context.Context, sessionID string) error
	Close() error
}

// LLMService defines the boundary for LLM interaction, including vector embeddings,
// conversational answer generation, and structured factual claim extraction.
type LLMService interface {
	GenerateEmbedding(ctx context.Context, text string) ([]float32, error)
	GenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error)
	GenerateAnswer(ctx context.Context, message string, history []entity.ChatMessage, facts []entity.Fact) (string, error)
	ExtractFacts(ctx context.Context, message string) ([]entity.ExtractedFact, error)
	ExtractRelations(ctx context.Context, message string) ([]entity.ExtractedRelation, error)
	ValidateConflict(ctx context.Context, candidate string, existing []string) (string, error)
	Close()
}

// SemanticCache defines the boundary for hybrid vector cache operations.
type SemanticCache interface {
	Get(ctx context.Context, queryVector []float32) (string, bool, error)
	Set(ctx context.Context, queryText string, queryVector []float32, replyText string) error
	Invalidate(ctx context.Context, entityID uuid.UUID) error
	Close() error
}

// PrivacyService defines the boundary for local PII filtering and Role-Based Access Control (RBAC).
type PrivacyService interface {
	ScrubText(ctx context.Context, text string) (string, error)
	ValidateAccess(ctx context.Context, agentRole string, fact *entity.Fact) bool
}

// ConsolidationService defines the boundary for sending tasks to the background worker pool.
type ConsolidationService interface {
	QueueInteraction(log entity.InteractionLog)
	QueueDocument(job entity.DocumentJob)
}

