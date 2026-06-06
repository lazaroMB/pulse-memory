package entity

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/google/uuid"
)

// Fact represents a single, atomic unit of semantic knowledge.
type Fact struct {
	ID              uuid.UUID  `db:"id" json:"id"`
	EntityID        uuid.UUID  `db:"entity_id" json:"entity_id"`
	Attribute       string     `db:"attribute" json:"attribute"`
	Value           string     `db:"val" json:"value"`
	ConfidenceScore float64    `db:"confidence_score" json:"confidence_score"`
	ValidFrom       time.Time  `db:"valid_from" json:"valid_from"`
	ValidUntil      *time.Time `db:"valid_until" json:"valid_until"`
	SourceAgent     string     `db:"source_agent" json:"source_agent"`
	MemoryStrength  float64    `db:"memory_strength" json:"memory_strength"`
	Stability       float64    `db:"stability" json:"stability"`
	LastAccessed    time.Time  `db:"last_accessed" json:"last_accessed"`
	AgentOwner      uuid.UUID  `db:"agent_owner" json:"agent_owner,omitempty"`
}

// Relation represents an edge in the temporal knowledge graph connecting two entities.
type Relation struct {
	ID             uuid.UUID  `db:"id" json:"id"`
	SourceID       uuid.UUID  `db:"source_id" json:"source_id"`
	TargetID       uuid.UUID  `db:"target_id" json:"target_id"`
	Type           string     `db:"rel_type" json:"type"`
	ValidFrom      time.Time  `db:"valid_from" json:"valid_from"`
	ValidUntil     *time.Time `db:"valid_until" json:"valid_until"`
	MemoryStrength float64    `db:"memory_strength" json:"memory_strength"`
	Stability      float64    `db:"stability" json:"stability"`
	LastAccessed   time.Time  `db:"last_accessed" json:"last_accessed"`
	AgentOwner     uuid.UUID  `db:"agent_owner" json:"agent_owner,omitempty"`
}

// EntityNode represents a unique concept/entity extracted in the memory graph.
type EntityNode struct {
	ID        uuid.UUID
	Name      string
	Embedding []float32
}

// CommunitySummary represents a macro narrative summary for a cluster of entities.
type CommunitySummary struct {
	ID        uuid.UUID `db:"id"`
	Name      string    `db:"name"`
	Summary   string    `db:"summary"`
	Entities  []uuid.UUID
	CreatedAt time.Time `db:"created_at"`
}

// ExtractedFact holds the intermediate structured output from the LLM.
type ExtractedFact struct {
	Attribute       string  `json:"attribute" db:"attribute"`
	Value           string  `json:"value" db:"val"`
	ConfidenceScore float64 `json:"confidence_score" db:"confidence_score"`
}

// ExtractedRelation holds structured relationship details extracted by the LLM.
type ExtractedRelation struct {
	SourceEntity string  `json:"source_entity"`
	TargetEntity string  `json:"target_entity"`
	RelationType string  `json:"relation_type"`
	Confidence   float64 `json:"confidence"`
}

// MemorySearchQuery represents the input parameters for a hybrid search.
type MemorySearchQuery struct {
	QueryText     string
	QueryVector   []float32
	TargetEntity  uuid.UUID
	RequiredScope string
	MaxResults    int
	AgentOwner    uuid.UUID
}

// CosineSimilarity calculates the cosine similarity between two 32-bit floating point vectors.
func CosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0, fmt.Errorf("vector dimensions mismatch or empty")
	}
	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		valA := float64(a[i])
		valB := float64(b[i])
		dotProduct += valA * valB
		normA += valA * valA
		normB += valB * valB
	}
	if normA == 0.0 || normB == 0.0 {
		return 0.0, nil
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}

type contextKey string

const AgentOwnerKey contextKey = "agent_owner"

func WithAgentOwner(ctx context.Context, ownerID uuid.UUID) context.Context {
	return context.WithValue(ctx, AgentOwnerKey, ownerID)
}

func GetAgentOwner(ctx context.Context) (uuid.UUID, bool) {
	val, ok := ctx.Value(AgentOwnerKey).(uuid.UUID)
	return val, ok
}


