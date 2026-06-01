package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
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
	_, _ = s.db.ExecContext(ctx, "ALTER TABLE facts ALTER COLUMN embedding TYPE VECTOR(3072);")
	_, _ = s.db.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS facts_embedding_hnsw_idx ON facts USING hnsw (embedding vector_cosine_ops);")

	// 3. Start a transaction for core schema tables and index creations
	tx, err := s.db.BeginTxx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to start schema transaction: %w", err)
	}
	defer tx.Rollback()

	// Create facts table with forgetting curve fields (default stability to 30 days/1 month)
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
		source_agent VARCHAR(255) NOT NULL,
		memory_strength DOUBLE PRECISION NOT NULL DEFAULT 1.0,
		stability DOUBLE PRECISION NOT NULL DEFAULT 30.0,
		last_accessed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
	);
	`
	if _, err := tx.ExecContext(ctx, factsSchema); err != nil {
		return fmt.Errorf("failed to create facts table: %w", err)
	}

	if _, err := tx.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS facts_entity_idx ON facts (entity_id);"); err != nil {
		return fmt.Errorf("failed to create facts_entity_idx: %w", err)
	}

	// Dynamic column migrations for backward compatibility (facts)
	alterFacts := []string{
		"ALTER TABLE facts ADD COLUMN IF NOT EXISTS memory_strength DOUBLE PRECISION NOT NULL DEFAULT 1.0;",
		"ALTER TABLE facts ADD COLUMN IF NOT EXISTS stability DOUBLE PRECISION NOT NULL DEFAULT 30.0;",
		"ALTER TABLE facts ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP;",
	}
	for _, q := range alterFacts {
		if _, err := tx.ExecContext(ctx, q); err != nil {
			return fmt.Errorf("failed to alter facts table: %w", err)
		}
	}

	// Retro-migration: update any existing 1.0 stabilities to 30.0 so memories do not decay prematurely
	_, _ = tx.ExecContext(ctx, "UPDATE facts SET stability = 30.0 WHERE stability = 1.0;")

	// GIN Full-Text Search index for sparse lexical matching
	if _, err := tx.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS facts_val_fts_idx ON facts USING gin(to_tsvector('english', val));"); err != nil {
		return fmt.Errorf("failed to create facts_val_fts_idx: %w", err)
	}

	// Create relations table with forgetting curve fields
	relationsSchema := `
	CREATE TABLE IF NOT EXISTS relations (
		id UUID PRIMARY KEY,
		source_id UUID NOT NULL,
		target_id UUID NOT NULL,
		rel_type VARCHAR(255) NOT NULL,
		valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
		valid_until TIMESTAMP WITH TIME ZONE,
		memory_strength DOUBLE PRECISION NOT NULL DEFAULT 1.0,
		stability DOUBLE PRECISION NOT NULL DEFAULT 30.0,
		last_accessed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
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

	// Dynamic column migrations for backward compatibility (relations)
	alterRels := []string{
		"ALTER TABLE relations ADD COLUMN IF NOT EXISTS memory_strength DOUBLE PRECISION NOT NULL DEFAULT 1.0;",
		"ALTER TABLE relations ADD COLUMN IF NOT EXISTS stability DOUBLE PRECISION NOT NULL DEFAULT 30.0;",
		"ALTER TABLE relations ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP;",
	}
	for _, q := range alterRels {
		if _, err := tx.ExecContext(ctx, q); err != nil {
			return fmt.Errorf("failed to alter relations table: %w", err)
		}
	}

	// Retro-migration: update any existing 1.0 stabilities to 30.0
	_, _ = tx.ExecContext(ctx, "UPDATE relations SET stability = 30.0 WHERE stability = 1.0;")

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

	communitySummariesSchema := `
	CREATE TABLE IF NOT EXISTS community_summaries (
		id UUID PRIMARY KEY,
		name VARCHAR(255) NOT NULL,
		summary TEXT NOT NULL,
		embedding VECTOR(3072),
		entities UUID[] NOT NULL,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
	);
	`
	if _, err := docTx.ExecContext(ctx, communitySummariesSchema); err != nil {
		return fmt.Errorf("failed to create community_summaries table: %w", err)
	}

	if err := docTx.Commit(); err != nil {
		return fmt.Errorf("failed to commit document schema transaction: %w", err)
	}

	// 5. Create vector HNSW indexes outside transaction to avoid aborting on warnings/mismatches
	_, _ = s.db.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS doc_chunks_embedding_hnsw_idx ON document_chunks USING hnsw (embedding vector_cosine_ops);")
	_, _ = s.db.ExecContext(ctx, "CREATE INDEX IF NOT EXISTS community_summaries_embedding_hnsw_idx ON community_summaries USING hnsw (embedding vector_cosine_ops);")

	return nil
}

func (s *PGStore) InsertFact(ctx context.Context, fact *Fact, vector []float32) error {
	query := `
	INSERT INTO facts (id, entity_id, attribute, val, embedding, confidence_score, valid_from, valid_until, source_agent, memory_strength, stability, last_accessed)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
	`
	vecStr := vectorToString(vector)
	strength := fact.MemoryStrength
	if strength <= 0 {
		strength = 1.0
	}
	stability := fact.Stability
	if stability <= 0 && stability != -1.0 {
		stability = getDefaultStability()
	}
	lastAccessed := fact.LastAccessed
	if lastAccessed.IsZero() {
		lastAccessed = time.Now()
	}

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
		strength,
		stability,
		lastAccessed,
	)
	if err != nil {
		return fmt.Errorf("failed to insert fact: %w", err)
	}
	return nil
}

func (s *PGStore) InsertFactWithProvenance(ctx context.Context, fact *Fact, vector []float32, docID uuid.UUID, chunkID uuid.UUID) error {
	tx, err := s.db.BeginTxx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for fact with provenance: %w", err)
	}
	defer tx.Rollback()

	// 1. Insert Fact
	factQuery := `
	INSERT INTO facts (id, entity_id, attribute, val, embedding, confidence_score, valid_from, valid_until, source_agent, memory_strength, stability, last_accessed)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
	`
	vecStr := vectorToString(vector)
	strength := fact.MemoryStrength
	if strength <= 0 {
		strength = 1.0
	}
	stability := fact.Stability
	if stability <= 0 && stability != -1.0 {
		stability = getDefaultStability()
	}
	lastAccessed := fact.LastAccessed
	if lastAccessed.IsZero() {
		lastAccessed = time.Now()
	}

	_, err = tx.ExecContext(ctx, factQuery,
		fact.ID,
		fact.EntityID,
		fact.Attribute,
		fact.Value,
		vecStr,
		fact.ConfidenceScore,
		fact.ValidFrom,
		fact.ValidUntil,
		fact.SourceAgent,
		strength,
		stability,
		lastAccessed,
	)
	if err != nil {
		return fmt.Errorf("failed to insert fact in transaction: %w", err)
	}

	// 2. Insert Provenance Link
	provQuery := `
	INSERT INTO fact_document_provenance (fact_id, document_id, chunk_id)
	VALUES ($1, $2, $3)
	ON CONFLICT (fact_id) DO UPDATE SET document_id = EXCLUDED.document_id, chunk_id = EXCLUDED.chunk_id
	`
	var chunkIDVal interface{} = chunkID
	if chunkID == uuid.Nil {
		chunkIDVal = nil
	}
	_, err = tx.ExecContext(ctx, provQuery, fact.ID, docID, chunkIDVal)
	if err != nil {
		return fmt.Errorf("failed to link fact to source document in transaction: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit fact with provenance transaction: %w", err)
	}
	return nil
}

func (s *PGStore) ReinforceFact(ctx context.Context, factID uuid.UUID, currentStability float64) {
	if currentStability == -1.0 {
		query := `
		UPDATE facts
		SET last_accessed = $1, memory_strength = 1.0
		WHERE id = $2
		`
		go func() {
			_, _ = s.db.ExecContext(context.Background(), query, time.Now(), factID)
		}()
		return
	}

	newStability := currentStability * 1.5
	if newStability > 365.0 {
		newStability = 365.0
	}
	if newStability <= 0 {
		newStability = getDefaultStability() * 1.5
	}
	query := `
	UPDATE facts
	SET last_accessed = $1, stability = $2, memory_strength = 1.0
	WHERE id = $3
	`
	go func() {
		_, _ = s.db.ExecContext(context.Background(), query, time.Now(), newStability, factID)
	}()
}

func (s *PGStore) SearchHybrid(ctx context.Context, q *MemorySearchQuery) ([]Fact, error) {
	// If query vector is empty, perform a normal active search by Entity ID
	if len(q.QueryVector) == 0 {
		var facts []Fact
		query := `
		SELECT id, entity_id, attribute, val, confidence_score, valid_from, valid_until, source_agent, memory_strength, stability, last_accessed
		FROM facts
		WHERE entity_id = $1 AND valid_until IS NULL
		ORDER BY valid_from DESC
		LIMIT $2
		`
		err := s.db.SelectContext(ctx, &facts, query, q.TargetEntity, q.MaxResults)
		if err != nil {
			return nil, fmt.Errorf("failed to select active facts: %w", err)
		}

		// Filter active facts by temporal decay (Forgetting Curve) in Go for engine-agnostic consistency
		var activeFacts []Fact
		for _, f := range facts {
			t := time.Since(f.LastAccessed).Hours()
			stab := f.Stability
			if stab == -1.0 {
				f.MemoryStrength = 1.0
				activeFacts = append(activeFacts, f)
				continue
			}
			if stab <= 0 {
				stab = getDefaultStability()
			}
			retention := math.Exp(-t / (stab * 24.0))
			if retention < 0.1 {
				// Decayed completely, mark inactive in background
				go func(id uuid.UUID) {
					_ = s.DeactivateFact(context.Background(), id)
				}(f.ID)
				continue
			}
			f.MemoryStrength = retention
			activeFacts = append(activeFacts, f)
		}

		// Reinforce accessed facts in background
		for _, f := range activeFacts {
			s.ReinforceFact(ctx, f.ID, f.Stability)
		}

		return activeFacts, nil
	}

	// Calculate a limit for individual sub-searches (retrieve more than MaxResults to allow good blending)
	subLimit := q.MaxResults * 2
	if subLimit < 20 {
		subLimit = 20
	}
	if subLimit > 100 {
		subLimit = 100
	}

	// 1. Dense Vector Search
	vecStr := vectorToString(q.QueryVector)
	var denseFacts []Fact
	denseQuery := `
	SELECT id, entity_id, attribute, val, confidence_score, valid_from, valid_until, source_agent, memory_strength, stability, last_accessed
	FROM facts
	WHERE entity_id = $1 AND valid_until IS NULL
	ORDER BY embedding <=> $2
	LIMIT $3
	`
	err := s.db.SelectContext(ctx, &denseFacts, denseQuery, q.TargetEntity, vecStr, subLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to perform dense vector search: %w", err)
	}

	// 2. Sparse Lexical Search using PostgreSQL Full Text Search
	var sparseFacts []Fact
	cleanQueryText := strings.TrimSpace(q.QueryText)
	if cleanQueryText != "" {
		sparseQuery := `
		SELECT id, entity_id, attribute, val, confidence_score, valid_from, valid_until, source_agent, memory_strength, stability, last_accessed
		FROM facts
		WHERE entity_id = $1 AND valid_until IS NULL AND to_tsvector('english', val) @@ plainto_tsquery('english', $2)
		LIMIT $3
		`
		err = s.db.SelectContext(ctx, &sparseFacts, sparseQuery, q.TargetEntity, cleanQueryText, subLimit)
		if err != nil {
			// Log FTS search error but do not fail; fallback to dense search
			sparseFacts = nil
		}
	}

	// 3. Process Forgetting Curve temporal decay on all candidates before RRF
	processDecay := func(facts []Fact) []Fact {
		var active []Fact
		for _, f := range facts {
			t := time.Since(f.LastAccessed).Hours()
			stab := f.Stability
			if stab == -1.0 {
				f.MemoryStrength = 1.0
				active = append(active, f)
				continue
			}
			if stab <= 0 {
				stab = getDefaultStability()
			}
			retention := math.Exp(-t / (stab * 24.0))
			if retention < 0.1 {
				// Asynchronously deactivate completely decayed facts
				go func(id uuid.UUID) {
					_ = s.DeactivateFact(context.Background(), id)
				}(f.ID)
				continue
			}
			f.MemoryStrength = retention
			active = append(active, f)
		}
		return active
	}

	denseFacts = processDecay(denseFacts)
	sparseFacts = processDecay(sparseFacts)

	// 4. Reciprocal Rank Fusion (RRF)
	// Constant k = 60.0 stabilizes ranks for outlier results
	const k = 60.0
	scores := make(map[uuid.UUID]float64)
	factMap := make(map[uuid.UUID]Fact)

	applyRanks := func(facts []Fact) {
		for rank, fact := range facts {
			rank1 := float64(rank + 1)
			scores[fact.ID] += 1.0 / (k + rank1)
			factMap[fact.ID] = fact
		}
	}

	applyRanks(denseFacts)
	applyRanks(sparseFacts)

	type scoredFact struct {
		fact  Fact
		score float64
	}

	var scored []scoredFact
	for id, score := range scores {
		scored = append(scored, scoredFact{
			fact:  factMap[id],
			score: score,
		})
	}

	// Sort scored facts by RRF score descending, then by confidence score and timestamp as tie-breakers
	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score > scored[j].score
		}
		if scored[i].fact.ConfidenceScore != scored[j].fact.ConfidenceScore {
			return scored[i].fact.ConfidenceScore > scored[j].fact.ConfidenceScore
		}
		return scored[i].fact.ValidFrom.After(scored[j].fact.ValidFrom)
	})

	resultSize := len(scored)
	if resultSize > q.MaxResults {
		resultSize = q.MaxResults
	}

	result := make([]Fact, resultSize)
	for i := 0; i < resultSize; i++ {
		result[i] = scored[i].fact
		// Reinforce accessed fact in background
		s.ReinforceFact(ctx, scored[i].fact.ID, scored[i].fact.Stability)
	}

	return result, nil
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
	INSERT INTO relations (id, source_id, target_id, rel_type, valid_from, valid_until, memory_strength, stability, last_accessed)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`
	strength := relation.MemoryStrength
	if strength <= 0 {
		strength = 1.0
	}
	stability := relation.Stability
	if stability <= 0 {
		stability = getDefaultStability()
	}
	lastAccessed := relation.LastAccessed
	if lastAccessed.IsZero() {
		lastAccessed = time.Now()
	}

	_, err := s.db.ExecContext(ctx, query,
		relation.ID,
		relation.SourceID,
		relation.TargetID,
		relation.Type,
		relation.ValidFrom,
		relation.ValidUntil,
		strength,
		stability,
		lastAccessed,
	)
	if err != nil {
		return fmt.Errorf("failed to insert relation: %w", err)
	}
	return nil
}

func (s *PGStore) GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]Relation, error) {
	var relations []Relation
	query := `
	SELECT id, source_id, target_id, rel_type, valid_from, valid_until, memory_strength, stability, last_accessed
	FROM relations
	WHERE (source_id = $1 OR target_id = $1) AND valid_until IS NULL
	`
	err := s.db.SelectContext(ctx, &relations, query, entityID)
	if err != nil {
		return nil, fmt.Errorf("failed to get active relations: %w", err)
	}
	return relations, nil
}

func (s *PGStore) GetActiveRelationsBatch(ctx context.Context, entityIDs []uuid.UUID) ([]Relation, error) {
	if len(entityIDs) == 0 {
		return nil, nil
	}
	var relations []Relation
	query, args, err := sqlx.In(`
	SELECT id, source_id, target_id, rel_type, valid_from, valid_until, memory_strength, stability, last_accessed
	FROM relations
	WHERE (source_id IN (?) OR target_id IN (?)) AND valid_until IS NULL
	`, entityIDs, entityIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to build IN query for relations: %w", err)
	}
	query = s.db.Rebind(query)
	err = s.db.SelectContext(ctx, &relations, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to get active relations batch: %w", err)
	}
	return relations, nil
}

func (s *PGStore) GetEntityNamesBatch(ctx context.Context, entityIDs []uuid.UUID) (map[uuid.UUID]string, error) {
	if len(entityIDs) == 0 {
		return nil, nil
	}
	nameMap := make(map[uuid.UUID]string)

	query, args, err := sqlx.In(`
	SELECT entity_id, attribute, val
	FROM facts
	WHERE entity_id IN (?) AND valid_until IS NULL
	ORDER BY entity_id, (case when attribute = 'name' then 0 else 1 end), valid_from DESC
	`, entityIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to build IN query for entity names: %w", err)
	}
	query = s.db.Rebind(query)

	type factRow struct {
		EntityID  uuid.UUID `db:"entity_id"`
		Attribute string    `db:"attribute"`
		Val       string    `db:"val"`
	}
	var rows []factRow
	err = s.db.SelectContext(ctx, &rows, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to select entity names batch: %w", err)
	}

	for _, row := range rows {
		if _, exists := nameMap[row.EntityID]; !exists {
			nameMap[row.EntityID] = row.Val
		}
	}
	return nameMap, nil
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

// GetAllEntities recupera todas las entidades activas del grafo representadas en la tabla facts.
func (s *PGStore) GetAllEntities(ctx context.Context) ([]EntityNode, error) {
	type row struct {
		EntityID  uuid.UUID `db:"entity_id"`
		Val       string    `db:"val"`
		Embedding string    `db:"embedding"`
	}

	query := `
	SELECT entity_id, val, embedding::text
	FROM facts
	WHERE attribute = 'name' AND valid_until IS NULL
	`
	var rows []row
	err := s.db.SelectContext(ctx, &rows, query)
	if err != nil {
		return nil, fmt.Errorf("failed to select entities: %w", err)
	}

	entities := make([]EntityNode, len(rows))
	for i, r := range rows {
		vec, _ := parseVectorString(r.Embedding)
		entities[i] = EntityNode{
			ID:        r.EntityID,
			Name:      r.Val,
			Embedding: vec,
		}
	}
	return entities, nil
}

// MergeEntities unifica de forma transaccional una entidad duplicada en su contraparte canónica.
func (s *PGStore) MergeEntities(ctx context.Context, canonicalID, duplicateID uuid.UUID) error {
	tx, err := s.db.BeginTxx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for merge: %w", err)
	}
	defer tx.Rollback()

	// 1. Migrar todos los hechos activos de la entidad duplicada a la canónica
	updateFactsQuery := `
	UPDATE facts
	SET entity_id = $1
	WHERE entity_id = $2
	`
	if _, err := tx.ExecContext(ctx, updateFactsQuery, canonicalID, duplicateID); err != nil {
		return fmt.Errorf("failed to migrate facts: %w", err)
	}

	// 2. Migrar aristas de relaciones de origen
	updateSourceRelsQuery := `
	UPDATE relations
	SET source_id = $1
	WHERE source_id = $2
	`
	if _, err := tx.ExecContext(ctx, updateSourceRelsQuery, canonicalID, duplicateID); err != nil {
		return fmt.Errorf("failed to migrate source relations: %w", err)
	}

	// 3. Migrar aristas de relaciones de destino
	updateTargetRelsQuery := `
	UPDATE relations
	SET target_id = $1
	WHERE target_id = $2
	`
	if _, err := tx.ExecContext(ctx, updateTargetRelsQuery, canonicalID, duplicateID); err != nil {
		return fmt.Errorf("failed to migrate target relations: %w", err)
	}

	// 4. Desactivar hechos duplicados de nombre de la entidad importada para evitar redundancia
	deactivateObsoleteQuery := `
	UPDATE facts
	SET valid_until = $1
	WHERE entity_id = $2 AND attribute = 'name' AND id <> $3 AND valid_until IS NULL
	`
	_, _ = tx.ExecContext(ctx, deactivateObsoleteQuery, time.Now(), canonicalID, canonicalID)

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit merge: %w", err)
	}
	return nil
}

// parseVectorString deserializa el formato texto de pgvector "[v1,v2,...]" a un slice de float32.
func parseVectorString(s string) ([]float32, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == "[]" || !strings.HasPrefix(s, "[") || !strings.HasSuffix(s, "]") {
		return nil, nil
	}

	trimmed := s[1 : len(s)-1]
	parts := strings.Split(trimmed, ",")
	res := make([]float32, len(parts))
	for i, p := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, err
		}
		res[i] = float32(val)
	}
	return res, nil
}

// getDefaultStability lee el valor DEFAULT_MEMORY_STABILITY_DAYS del archivo .env,
// cayendo de vuelta a 30.0 días si no está definido o es inválido.
func getDefaultStability() float64 {
	val := os.Getenv("DEFAULT_MEMORY_STABILITY_DAYS")
	if val != "" {
		if parsed, err := strconv.ParseFloat(val, 64); err == nil && parsed > 0 {
			return parsed
		}
	}
	return 30.0 // Default fallback a 30 días
}

func (s *PGStore) InsertCommunitySummary(ctx context.Context, id uuid.UUID, name string, summary string, embedding []float32, entities []uuid.UUID) error {
	query := `
	INSERT INTO community_summaries (id, name, summary, embedding, entities)
	VALUES ($1, $2, $3, $4, $5)
	ON CONFLICT (id) DO UPDATE SET
		name = EXCLUDED.name,
		summary = EXCLUDED.summary,
		embedding = EXCLUDED.embedding,
		entities = EXCLUDED.entities
	`
	vecStr := vectorToString(embedding)
	arrStr := formatUUIDArray(entities)

	_, err := s.db.ExecContext(ctx, query, id, name, summary, vecStr, arrStr)
	if err != nil {
		return fmt.Errorf("failed to insert community summary: %w", err)
	}
	return nil
}

func (s *PGStore) SearchCommunitySummaries(ctx context.Context, queryVector []float32, limit int) ([]CommunitySummary, error) {
	if len(queryVector) == 0 {
		return nil, nil
	}

	query := `
	SELECT id, name, summary, entities::text, created_at
	FROM community_summaries
	ORDER BY embedding <=> $1
	LIMIT $2
	`
	vecStr := vectorToString(queryVector)

	type row struct {
		ID        uuid.UUID `db:"id"`
		Name      string    `db:"name"`
		Summary   string    `db:"summary"`
		Entities  string    `db:"entities"`
		CreatedAt time.Time `db:"created_at"`
	}

	var dbRows []row
	err := s.db.SelectContext(ctx, &dbRows, query, vecStr, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to search community summaries: %w", err)
	}

	results := make([]CommunitySummary, len(dbRows))
	for i, r := range dbRows {
		results[i] = CommunitySummary{
			ID:        r.ID,
			Name:      r.Name,
			Summary:   r.Summary,
			Entities:  parseUUIDArray(r.Entities),
			CreatedAt: r.CreatedAt,
		}
	}
	return results, nil
}

func formatUUIDArray(uuids []uuid.UUID) string {
	strs := make([]string, len(uuids))
	for i, u := range uuids {
		strs[i] = u.String()
	}
	return "{" + strings.Join(strs, ",") + "}"
}

func parseUUIDArray(pgArray string) []uuid.UUID {
	pgArray = strings.Trim(pgArray, "{}")
	if pgArray == "" {
		return nil
	}
	parts := strings.Split(pgArray, ",")
	uuids := make([]uuid.UUID, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if u, err := uuid.Parse(p); err == nil {
			uuids = append(uuids, u)
		}
	}
	return uuids
}


