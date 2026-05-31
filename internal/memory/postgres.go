package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
	"pulse/internal/document"
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

	// 4. Start a separate transaction for document schema tables and vector types
	docTx, err := s.db.BeginTxx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to start document schema transaction: %w", err)
	}
	defer docTx.Rollback()

	documentsSchema := `
	CREATE TABLE IF NOT EXISTS documents (
		id UUID PRIMARY KEY,
		title VARCHAR(255) NOT NULL,
		source_type VARCHAR(50) NOT NULL,
		source_url TEXT,
		file_path TEXT,
		status VARCHAR(50) NOT NULL,
		error_message TEXT,
		metadata JSONB,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
	);
	`
	if _, err := docTx.ExecContext(ctx, documentsSchema); err != nil {
		return fmt.Errorf("failed to create documents table: %w", err)
	}

	chunksSchema := `
	CREATE TABLE IF NOT EXISTS document_chunks (
		id UUID PRIMARY KEY,
		document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
		chunk_index INT NOT NULL,
		content TEXT NOT NULL,
		embedding VECTOR(3072),
		metadata JSONB
	);
	`
	if _, err := docTx.ExecContext(ctx, chunksSchema); err != nil {
		return fmt.Errorf("failed to create document_chunks table: %w", err)
	}

	if _, err := docTx.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS doc_chunks_doc_idx ON document_chunks (document_id);"); err != nil {
		return fmt.Errorf("failed to create doc_chunks_doc_idx: %w", err)
	}

	// Index drop and update logic for PG vector length on chunks table, similar to facts table
	_, _ = docTx.ExecContext(ctx, "DROP INDEX IF EXISTS doc_chunks_embedding_hnsw_idx;")
	_, _ = docTx.ExecContext(ctx, "DROP INDEX IF EXISTS doc_chunks_embedding_idx;")

	topicsSchema := `
	CREATE TABLE IF NOT EXISTS document_topics (
		document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
		topic_name VARCHAR(255) NOT NULL,
		PRIMARY KEY (document_id, topic_name)
	);
	`
	if _, err := docTx.ExecContext(ctx, topicsSchema); err != nil {
		return fmt.Errorf("failed to create document_topics table: %w", err)
	}

	authorsSchema := `
	CREATE TABLE IF NOT EXISTS document_authors (
		document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
		author_entity_id UUID NOT NULL,
		PRIMARY KEY (document_id, author_entity_id)
	);
	`
	if _, err := docTx.ExecContext(ctx, authorsSchema); err != nil {
		return fmt.Errorf("failed to create document_authors table: %w", err)
	}

	provenanceSchema := `
	CREATE TABLE IF NOT EXISTS fact_document_provenance (
		fact_id UUID PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
		document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
		chunk_id UUID REFERENCES document_chunks(id) ON DELETE SET NULL
	);
	`
	if _, err := docTx.ExecContext(ctx, provenanceSchema); err != nil {
		return fmt.Errorf("failed to create fact_document_provenance table: %w", err)
	}

	if err := docTx.Commit(); err != nil {
		return fmt.Errorf("failed to commit document schema transaction: %w", err)
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

func (s *PGStore) InsertDocument(ctx context.Context, doc *document.Document) error {
	metadataBytes, err := json.Marshal(doc.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal document metadata: %w", err)
	}

	query := `
	INSERT INTO documents (id, title, source_type, source_url, file_path, status, error_message, metadata, created_at, updated_at)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`
	_, err = s.db.ExecContext(ctx, query,
		doc.ID,
		doc.Title,
		doc.SourceType,
		doc.SourceURL,
		doc.FilePath,
		doc.Status,
		doc.ErrorMessage,
		metadataBytes,
		doc.CreatedAt,
		doc.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert document: %w", err)
	}
	return nil
}

func (s *PGStore) GetDocument(ctx context.Context, id uuid.UUID) (*document.Document, error) {
	type docRow struct {
		ID           uuid.UUID `db:"id"`
		Title        string    `db:"title"`
		SourceType   string    `db:"source_type"`
		SourceURL    *string   `db:"source_url"`
		FilePath     *string   `db:"file_path"`
		Status       string    `db:"status"`
		ErrorMessage *string   `db:"error_message"`
		Metadata     []byte    `db:"metadata"`
		CreatedAt    time.Time `db:"created_at"`
		UpdatedAt    time.Time `db:"updated_at"`
	}
	var r docRow
	err := s.db.GetContext(ctx, &r, "SELECT id, title, source_type, source_url, file_path, status, error_message, metadata, created_at, updated_at FROM documents WHERE id = $1", id)
	if err != nil {
		return nil, fmt.Errorf("failed to select document: %w", err)
	}
	var meta map[string]string
	if len(r.Metadata) > 0 {
		_ = json.Unmarshal(r.Metadata, &meta)
	}
	var srcURL, filePath, errMsg string
	if r.SourceURL != nil { srcURL = *r.SourceURL }
	if r.FilePath != nil { filePath = *r.FilePath }
	if r.ErrorMessage != nil { errMsg = *r.ErrorMessage }

	return &document.Document{
		ID:           r.ID,
		Title:        r.Title,
		SourceType:   document.SourceType(r.SourceType),
		SourceURL:    srcURL,
		FilePath:     filePath,
		Status:       document.IngestionStatus(r.Status),
		ErrorMessage: errMsg,
		Metadata:     meta,
		CreatedAt:    r.CreatedAt,
		UpdatedAt:    r.UpdatedAt,
	}, nil
}

func (s *PGStore) UpdateDocumentStatus(ctx context.Context, docID uuid.UUID, status document.IngestionStatus, errMsg string) error {
	query := `
	UPDATE documents
	SET status = $1, error_message = $2, updated_at = $3
	WHERE id = $4
	`
	_, err := s.db.ExecContext(ctx, query, status, errMsg, time.Now(), docID)
	if err != nil {
		return fmt.Errorf("failed to update document status: %w", err)
	}
	return nil
}

func (s *PGStore) InsertDocumentChunks(ctx context.Context, chunks []document.DocumentChunk, embeddings [][]float32) error {
	if len(chunks) == 0 {
		return nil
	}

	tx, err := s.db.BeginTxx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for document chunks: %w", err)
	}
	defer tx.Rollback()

	query := `
	INSERT INTO document_chunks (id, document_id, chunk_index, content, embedding, metadata)
	VALUES ($1, $2, $3, $4, $5, $6)
	`
	for i, chunk := range chunks {
		metadataBytes, err := json.Marshal(chunk.Metadata)
		if err != nil {
			return fmt.Errorf("failed to marshal chunk metadata at index %d: %w", i, err)
		}

		var vecStr *string
		if len(embeddings) > i && len(embeddings[i]) > 0 {
			s := vectorToString(embeddings[i])
			vecStr = &s
		}

		_, err = tx.ExecContext(ctx, query,
			chunk.ID,
			chunk.DocumentID,
			chunk.ChunkIndex,
			chunk.Content,
			vecStr,
			metadataBytes,
		)
		if err != nil {
			return fmt.Errorf("failed to insert chunk at index %d: %w", i, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit document chunks: %w", err)
	}
	return nil
}

func (s *PGStore) SearchDocumentChunks(ctx context.Context, queryVector []float32, limit int) ([]document.DocumentChunk, error) {
	vecStr := vectorToString(queryVector)
	
	type chunkRow struct {
		ID         uuid.UUID `db:"id"`
		DocumentID uuid.UUID `db:"document_id"`
		ChunkIndex int       `db:"chunk_index"`
		Content    string    `db:"content"`
		Metadata   []byte    `db:"metadata"`
	}

	query := `
	SELECT id, document_id, chunk_index, content, metadata
	FROM document_chunks
	ORDER BY embedding <=> $1
	LIMIT $2
	`
	var rows []chunkRow
	err := s.db.SelectContext(ctx, &rows, query, vecStr, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to search document chunks: %w", err)
	}

	chunks := make([]document.DocumentChunk, len(rows))
	for i, r := range rows {
		var meta map[string]string
		if len(r.Metadata) > 0 {
			_ = json.Unmarshal(r.Metadata, &meta)
		}
		chunks[i] = document.DocumentChunk{
			ID:         r.ID,
			DocumentID: r.DocumentID,
			ChunkIndex: r.ChunkIndex,
			Content:    r.Content,
			Metadata:   meta,
		}
	}

	return chunks, nil
}

func (s *PGStore) LinkDocumentToAuthor(ctx context.Context, docID uuid.UUID, authorID uuid.UUID) error {
	query := `
	INSERT INTO document_authors (document_id, author_entity_id)
	VALUES ($1, $2)
	ON CONFLICT (document_id, author_entity_id) DO NOTHING
	`
	_, err := s.db.ExecContext(ctx, query, docID, authorID)
	if err != nil {
		return fmt.Errorf("failed to link document to author: %w", err)
	}
	return nil
}

func (s *PGStore) LinkDocumentToTopic(ctx context.Context, docID uuid.UUID, topicName string) error {
	query := `
	INSERT INTO document_topics (document_id, topic_name)
	VALUES ($1, $2)
	ON CONFLICT (document_id, topic_name) DO NOTHING
	`
	_, err := s.db.ExecContext(ctx, query, docID, topicName)
	if err != nil {
		return fmt.Errorf("failed to link document to topic: %w", err)
	}
	return nil
}

func (s *PGStore) LinkFactToSource(ctx context.Context, factID uuid.UUID, docID uuid.UUID, chunkID uuid.UUID) error {
	query := `
	INSERT INTO fact_document_provenance (fact_id, document_id, chunk_id)
	VALUES ($1, $2, $3)
	ON CONFLICT (fact_id) DO UPDATE SET document_id = EXCLUDED.document_id, chunk_id = EXCLUDED.chunk_id
	`
	var chunkIDVal interface{} = chunkID
	if chunkID == uuid.Nil {
		chunkIDVal = nil
	}
	_, err := s.db.ExecContext(ctx, query, factID, docID, chunkIDVal)
	if err != nil {
		return fmt.Errorf("failed to link fact to source document: %w", err)
	}
	return nil
}

