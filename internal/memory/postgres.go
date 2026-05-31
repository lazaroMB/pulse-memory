package memory

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
)

type PGStore struct {
	db *sqlx.DB
}

func NewPGStore(databaseURL string) (*PGStore, error) {
	db, err := sqlx.Connect("pgx", databaseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}
	return &PGStore{db: db}, nil
}

func (s *PGStore) Close() error {
	return s.db.Close()
}

func (s *PGStore) InitSchema(ctx context.Context) error {
	// 1. Enable pgvector extension (must run outside transactional block in some PG environments)
	_, err := s.db.ExecContext(ctx, "CREATE EXTENSION IF NOT EXISTS vector;")
	if err != nil {
		return fmt.Errorf("failed to enable vector extension: %w", err)
	}

	// 2. Execute pgvector migrations outside the main transaction.
	// Since 3072-dimensional embeddings exceed the 2000-dimension limit for pgvector HNSW/IVFFlat indexes,
	// we drop any existing HNSW index and avoid recreating it. exact sequential scanning is performed.
	_, _ = s.db.ExecContext(ctx, "DROP INDEX IF EXISTS facts_embedding_hnsw_idx;")
	_, _ = s.db.ExecContext(ctx, "ALTER TABLE facts ALTER COLUMN embedding TYPE VECTOR(3072);")

	// 3. Start a transaction for core schema tables and index creations
	tx, err := s.db.BeginTxx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to start schema transaction: %w", err)
	}
	defer tx.Rollback()

	// Create facts table
	factsSchema := `
	CREATE TABLE IF NOT EXISTS facts (
		id UUID PRIMARY KEY,
		entity_id UUID NOT NULL,
		attribute VARCHAR(255) NOT NULL,
		val TEXT NOT NULL,
		embedding VECTOR(3072),
		confidence_score DOUBLE PRECISION NOT NULL,
		valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
		valid_until TIMESTAMP WITH TIME ZONE,
		source_agent VARCHAR(255) NOT NULL
	);
	`
	if _, err := tx.ExecContext(ctx, factsSchema); err != nil {
		return fmt.Errorf("failed to create facts table: %w", err)
	}

	if _, err := tx.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS facts_entity_idx ON facts (entity_id);"); err != nil {
		return fmt.Errorf("failed to create facts_entity_idx: %w", err)
	}

	// Create relations table
	relationsSchema := `
	CREATE TABLE IF NOT EXISTS relations (
		id UUID PRIMARY KEY,
		source_id UUID NOT NULL,
		target_id UUID NOT NULL,
		rel_type VARCHAR(255) NOT NULL,
		valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
		valid_until TIMESTAMP WITH TIME ZONE
	);
	`
	if _, err := tx.ExecContext(ctx, relationsSchema); err != nil {
		return fmt.Errorf("failed to create relations table: %w", err)
	}

	if _, err := tx.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS relations_source_idx ON relations (source_id);"); err != nil {
		return fmt.Errorf("failed to create relations_source_idx: %w", err)
	}

	if _, err := tx.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS relations_target_idx ON relations (target_id);"); err != nil {
		return fmt.Errorf("failed to create relations_target_idx: %w", err)
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit schema transaction: %w", err)
	}

	return nil
}

func (s *PGStore) InsertFact(ctx context.Context, fact *Fact, vector []float32) error {
	query := `
	INSERT INTO facts (id, entity_id, attribute, val, embedding, confidence_score, valid_from, valid_until, source_agent)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`
	vecStr := vectorToString(vector)
	_, err := s.db.ExecContext(ctx, query,
		fact.ID,
		fact.EntityID,
		fact.Attribute,
		fact.Value,
		vecStr,
		fact.ConfidenceScore,
		fact.ValidFrom,
		fact.ValidUntil,
		fact.SourceAgent,
	)
	if err != nil {
		return fmt.Errorf("failed to insert fact: %w", err)
	}
	return nil
}

func (s *PGStore) SearchHybrid(ctx context.Context, q *MemorySearchQuery) ([]Fact, error) {
	// If query vector is empty, perform a normal active search by Entity ID
	if len(q.QueryVector) == 0 {
		var facts []Fact
		query := `
		SELECT id, entity_id, attribute, val, confidence_score, valid_from, valid_until, source_agent
		FROM facts
		WHERE entity_id = $1 AND valid_until IS NULL
		ORDER BY valid_from DESC
		LIMIT $2
		`
		err := s.db.SelectContext(ctx, &facts, query, q.TargetEntity, q.MaxResults)
		if err != nil {
			return nil, fmt.Errorf("failed to select active facts: %w", err)
		}
		return facts, nil
	}

	// Vector Cosine Similarity Search combined with Entity filtering and Active State check
	vecStr := vectorToString(q.QueryVector)
	var facts []Fact
	query := `
	SELECT id, entity_id, attribute, val, confidence_score, valid_from, valid_until, source_agent
	FROM facts
	WHERE entity_id = $1 AND valid_until IS NULL
	ORDER BY embedding <=> $2
	LIMIT $3
	`
	err := s.db.SelectContext(ctx, &facts, query, q.TargetEntity, vecStr, q.MaxResults)
	if err != nil {
		return nil, fmt.Errorf("failed to perform hybrid vector search: %w", err)
	}
	return facts, nil
}

func (s *PGStore) DeactivateFact(ctx context.Context, factID uuid.UUID) error {
	query := `
	UPDATE facts
	SET valid_until = $1
	WHERE id = $2 AND valid_until IS NULL
	`
	_, err := s.db.ExecContext(ctx, query, time.Now(), factID)
	if err != nil {
		return fmt.Errorf("failed to deactivate fact: %w", err)
	}
	return nil
}

func (s *PGStore) InsertRelation(ctx context.Context, relation *Relation) error {
	query := `
	INSERT INTO relations (id, source_id, target_id, rel_type, valid_from, valid_until)
	VALUES ($1, $2, $3, $4, $5, $6)
	`
	_, err := s.db.ExecContext(ctx, query,
		relation.ID,
		relation.SourceID,
		relation.TargetID,
		relation.Type,
		relation.ValidFrom,
		relation.ValidUntil,
	)
	if err != nil {
		return fmt.Errorf("failed to insert relation: %w", err)
	}
	return nil
}

func (s *PGStore) GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]Relation, error) {
	var relations []Relation
	query := `
	SELECT id, source_id, target_id, rel_type, valid_from, valid_until
	FROM relations
	WHERE (source_id = $1 OR target_id = $1) AND valid_until IS NULL
	`
	err := s.db.SelectContext(ctx, &relations, query, entityID)
	if err != nil {
		return nil, fmt.Errorf("failed to get active relations: %w", err)
	}
	return relations, nil
}

// Helper to serialize float32 slice to pgvector string format "[v1,v2,v3,...]"
func vectorToString(v []float32) string {
	if len(v) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteByte('[')
	for i, val := range v {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(strconv.FormatFloat(float64(val), 'f', 6, 32))
	}
	sb.WriteByte(']')
	return sb.String()
}
