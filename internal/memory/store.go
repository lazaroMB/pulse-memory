package memory

import (
	"context"
	"time"

	"github.com/google/uuid"
	"pulse/internal/document"
)

// Fact represents a single, atomic unit of semantic knowledge.
type Fact struct {
	ID              uuid.UUID  `db:"id" json:"id"`
	EntityID        uuid.UUID  `db:"entity_id" json:"entity_id"`
	Attribute       string     `db:"attribute" json:"attribute"`
	Value           string     `db:"val" json:"value"` // 'val' in DB as 'value' is a PG keyword sometimes
	ConfidenceScore float64    `db:"confidence_score" json:"confidence_score"`
	ValidFrom       time.Time  `db:"valid_from" json:"valid_from"`
	ValidUntil      *time.Time `db:"valid_until" json:"valid_until"`
	SourceAgent     string     `db:"source_agent" json:"source_agent"`
}

// Relation represents an edge in the temporal knowledge graph connecting two entities.
type Relation struct {
	ID         uuid.UUID  `db:"id" json:"id"`
	SourceID   uuid.UUID  `db:"source_id" json:"source_id"`
	TargetID   uuid.UUID  `db:"target_id" json:"target_id"`
	Type       string     `db:"rel_type" json:"type"` // 'rel_type' in DB
	ValidFrom  time.Time  `db:"valid_from" json:"valid_from"`
	ValidUntil *time.Time `db:"valid_until" json:"valid_until"`
}

// MemorySearchQuery represents the input parameters for a hybrid search.
type MemorySearchQuery struct {
	QueryText     string
	QueryVector   []float32
	TargetEntity  uuid.UUID
	RequiredScope string
	MaxResults    int
}

// MemoryStore defines the boundary for database writes and hybrid lookups.
type MemoryStore interface {
	InitSchema(ctx context.Context) error
	InsertFact(ctx context.Context, fact *Fact, vector []float32) error
	SearchHybrid(ctx context.Context, query *MemorySearchQuery) ([]Fact, error)
	DeactivateFact(ctx context.Context, factID uuid.UUID) error
	InsertRelation(ctx context.Context, relation *Relation) error
	GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]Relation, error)
	Close() error

	// Document Ingestion & Graph Knowledge Extensions
	InsertDocument(ctx context.Context, doc *document.Document) error
	GetDocument(ctx context.Context, id uuid.UUID) (*document.Document, error)
	UpdateDocumentStatus(ctx context.Context, docID uuid.UUID, status document.IngestionStatus, errMsg string) error
	InsertDocumentChunks(ctx context.Context, chunks []document.DocumentChunk, embeddings [][]float32) error
	SearchDocumentChunks(ctx context.Context, queryVector []float32, limit int) ([]document.DocumentChunk, error)
	LinkDocumentToAuthor(ctx context.Context, docID uuid.UUID, authorID uuid.UUID) error
	LinkDocumentToTopic(ctx context.Context, docID uuid.UUID, topicName string) error
	LinkFactToSource(ctx context.Context, factID uuid.UUID, docID uuid.UUID, chunkID uuid.UUID) error
}

