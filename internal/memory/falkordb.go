package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/google/uuid"
	redisgraph "github.com/falkordb/falkordb-go"
	"github.com/gomodule/redigo/redis"
	"pulse/internal/document"
)

type FalkorDBStore struct {
	pool      *redis.Pool
	graphName string
}

// NewFalkorDBStore initializes a FalkorDB connection using a Redis connection pool.
func NewFalkorDBStore(address, graphName string) (*FalkorDBStore, error) {
	pool := &redis.Pool{
		MaxIdle:     5,
		MaxActive:   20,
		IdleTimeout: 240 * time.Second,
		Dial: func() (redis.Conn, error) {
			return redis.Dial("tcp", address, redis.DialConnectTimeout(5*time.Second))
		},
	}

	// Verify connectivity
	conn := pool.Get()
	defer conn.Close()

	if _, err := conn.Do("PING"); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to ping Redis/FalkorDB instance: %w", err)
	}

	return &FalkorDBStore{
		pool:      pool,
		graphName: graphName,
	}, nil
}

// Close closes the Redis connection pool.
func (s *FalkorDBStore) Close() error {
	return s.pool.Close()
}

// InitSchema prepares FalkorDB schema assets.
func (s *FalkorDBStore) InitSchema(ctx context.Context) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	// Attempt to create the vector node index on Facts
	query := "CALL db.idx.vector.createNodeIndex('Fact', 'embedding', 3072, 'COSINE')"
	_, _ = graph.Query(query) // Ignore error if index already exists

	// Attempt to create the vector node index on DocumentChunks
	chunkQuery := "CALL db.idx.vector.createNodeIndex('DocumentChunk', 'embedding', 3072, 'COSINE')"
	_, _ = graph.Query(chunkQuery) // Ignore error if index already exists

	return nil
}

// InsertFact writes a fact node and HAS_FACT edge to the graph.
func (s *FalkorDBStore) InsertFact(ctx context.Context, fact *Fact, vector []float32) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	// FalkorDB maps float slices via arrays of interfaces (float64)
	embedding := make([]interface{}, len(vector))
	for i, v := range vector {
		embedding[i] = float64(v)
	}

	var validUntilStr string = ""
	if fact.ValidUntil != nil {
		validUntilStr = fact.ValidUntil.Format(time.RFC3339Nano)
	}

	query := `
	MERGE (e:Entity {id: $entity_id})
	CREATE (f:Fact {
		id: $id,
		attribute: $attribute,
		val: $val,
		embedding: $embedding,
		confidence_score: $confidence_score,
		valid_from: $valid_from,
		valid_until: $valid_until,
		source_agent: $source_agent
	})
	CREATE (e)-[:HAS_FACT]->(f)
	`
	params := map[string]interface{}{
		"entity_id":        fact.EntityID.String(),
		"id":              fact.ID.String(),
		"attribute":       fact.Attribute,
		"val":             fact.Value,
		"embedding":       embedding,
		"confidence_score": fact.ConfidenceScore,
		"valid_from":       fact.ValidFrom.Format(time.RFC3339Nano),
		"valid_until":      validUntilStr,
		"source_agent":     fact.SourceAgent,
	}

	_, err := graph.ParameterizedQuery(query, params)
	if err != nil {
		return fmt.Errorf("failed to insert fact: %w", err)
	}
	return nil
}

// SearchHybrid queries active facts, calculating exact cosine similarity on Go side.
func (s *FalkorDBStore) SearchHybrid(ctx context.Context, q *MemorySearchQuery) ([]Fact, error) {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	// 1. Exact active keyword query by entity
	if len(q.QueryVector) == 0 {
		query := `
		MATCH (e:Entity {id: $entity_id})-[:HAS_FACT]->(f:Fact)
		WHERE f.valid_until = "" OR f.valid_until IS NULL
		RETURN f.id AS id, e.id AS entity_id, f.attribute AS attribute, f.val AS val, 
		       f.confidence_score AS confidence_score, f.valid_from AS valid_from, 
		       f.valid_until AS valid_until, f.source_agent AS source_agent
		ORDER BY f.valid_from DESC
		LIMIT $limit
		`
		res, err := graph.ParameterizedQuery(query, map[string]interface{}{
			"entity_id": q.TargetEntity.String(),
			"limit":     q.MaxResults,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to perform keyword search: %w", err)
		}
		return parseFalkorFacts(res)
	}

	// 2. Dynamic Hybrid Search using exact in-memory cosine similarity calculation
	query := `
	MATCH (e:Entity {id: $entity_id})-[:HAS_FACT]->(f:Fact)
	WHERE f.valid_until = "" OR f.valid_until IS NULL
	RETURN f.id AS id, e.id AS entity_id, f.attribute AS attribute, f.val AS val, 
	       f.confidence_score AS confidence_score, f.valid_from AS valid_from, 
	       f.valid_until AS valid_until, f.source_agent AS source_agent, f.embedding AS embedding
	`
	res, err := graph.ParameterizedQuery(query, map[string]interface{}{
		"entity_id": q.TargetEntity.String(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve active facts for vector query: %w", err)
	}

	type scoredFact struct {
		fact  Fact
		score float32
	}

	var scored []scoredFact

	for res.Next() {
		record := res.Record()

		idVal, _ := record.Get("id")
		id, err := uuid.Parse(idVal.(string))
		if err != nil {
			continue
		}

		entityVal, _ := record.Get("entity_id")
		entityID, err := uuid.Parse(entityVal.(string))
		if err != nil {
			continue
		}

		attr, _ := record.Get("attribute")
		val, _ := record.Get("val")
		confidence, _ := record.Get("confidence_score")

		var confFloat float64
		if cf, ok := confidence.(float64); ok {
			confFloat = cf
		} else if ci, ok := confidence.(int64); ok {
			confFloat = float64(ci)
		}

		validFromStr, _ := record.Get("valid_from")
		validFrom, err := time.Parse(time.RFC3339Nano, validFromStr.(string))
		if err != nil {
			validFrom, _ = time.Parse(time.RFC3339, validFromStr.(string))
		}

		var validUntil *time.Time
		if vuVal, ok := record.Get("valid_until"); ok && vuVal != nil {
			if vuStr, ok := vuVal.(string); ok && vuStr != "" {
				t, err := time.Parse(time.RFC3339Nano, vuStr)
				if err != nil {
					t, _ = time.Parse(time.RFC3339, vuStr)
				}
				validUntil = &t
			}
		}

		srcAgent, _ := record.Get("source_agent")

		fact := Fact{
			ID:              id,
			EntityID:        entityID,
			Attribute:       attr.(string),
			Value:           val.(string),
			ConfidenceScore: confFloat,
			ValidFrom:       validFrom,
			ValidUntil:      validUntil,
			SourceAgent:     srcAgent.(string),
		}

		// Read embedding property
		var factEmb []float32
		if embVal, ok := record.Get("embedding"); ok && embVal != nil {
			if embList, ok := embVal.([]interface{}); ok {
				factEmb = make([]float32, len(embList))
				for i, v := range embList {
					if fv, ok := v.(float64); ok {
						factEmb[i] = float32(fv)
					} else if fv, ok := v.(float32); ok {
						factEmb[i] = fv
					} else if fi, ok := v.(int64); ok {
						factEmb[i] = float32(fi)
					}
				}
			}
		}

		score := falkorCosineSimilarity(q.QueryVector, factEmb)

		scored = append(scored, scoredFact{
			fact:  fact,
			score: score,
		})
	}

	// Sort by descending similarity score
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	limit := q.MaxResults
	if limit > len(scored) {
		limit = len(scored)
	}

	facts := make([]Fact, limit)
	for i := 0; i < limit; i++ {
		facts[i] = scored[i].fact
	}

	return facts, nil
}

// DeactivateFact sets the valid_until timestamp to mark fact as deactivated.
func (s *FalkorDBStore) DeactivateFact(ctx context.Context, factID uuid.UUID) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	query := `
	MATCH (f:Fact {id: $id})
	WHERE f.valid_until = "" OR f.valid_until IS NULL
	SET f.valid_until = $valid_until
	`
	params := map[string]interface{}{
		"id":          factID.String(),
		"valid_until": time.Now().Format(time.RFC3339Nano),
	}

	_, err := graph.ParameterizedQuery(query, params)
	if err != nil {
		return fmt.Errorf("failed to deactivate fact: %w", err)
	}
	return nil
}

// InsertRelation writes a RELATION edge between two entity nodes in FalkorDB.
func (s *FalkorDBStore) InsertRelation(ctx context.Context, relation *Relation) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	var validUntilStr string = ""
	if relation.ValidUntil != nil {
		validUntilStr = relation.ValidUntil.Format(time.RFC3339Nano)
	}

	query := `
	MERGE (s:Entity {id: $source_id})
	MERGE (t:Entity {id: $target_id})
	CREATE (s)-[r:RELATION {
		id: $id,
		type: $type,
		valid_from: $valid_from,
		valid_until: $valid_until
	}]->(t)
	`
	params := map[string]interface{}{
		"source_id":   relation.SourceID.String(),
		"target_id":   relation.TargetID.String(),
		"id":          relation.ID.String(),
		"type":        relation.Type,
		"valid_from":  relation.ValidFrom.Format(time.RFC3339Nano),
		"valid_until": validUntilStr,
	}

	_, err := graph.ParameterizedQuery(query, params)
	if err != nil {
		return fmt.Errorf("failed to insert relation: %w", err)
	}
	return nil
}

// GetActiveRelations retrieves all active relationship edges connected to the target entity.
func (s *FalkorDBStore) GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]Relation, error) {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	query := `
	MATCH (s:Entity)-[r:RELATION]->(t:Entity)
	WHERE (s.id = $entity_id OR t.id = $entity_id) AND (r.valid_until = "" OR r.valid_until IS NULL)
	RETURN r.id AS id, s.id AS source_id, t.id AS target_id, r.type AS rel_type, 
	       r.valid_from AS valid_from, r.valid_until AS valid_until
	`
	res, err := graph.ParameterizedQuery(query, map[string]interface{}{
		"entity_id": entityID.String(),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to fetch active relations: %w", err)
	}

	var relations []Relation
	for res.Next() {
		record := res.Record()

		idVal, _ := record.Get("id")
		id, err := uuid.Parse(idVal.(string))
		if err != nil {
			continue
		}

		srcVal, _ := record.Get("source_id")
		srcID, err := uuid.Parse(srcVal.(string))
		if err != nil {
			continue
		}

		tgtVal, _ := record.Get("target_id")
		tgtID, err := uuid.Parse(tgtVal.(string))
		if err != nil {
			continue
		}

		relType, _ := record.Get("rel_type")
		validFromStr, _ := record.Get("valid_from")
		validFrom, err := time.Parse(time.RFC3339Nano, validFromStr.(string))
		if err != nil {
			validFrom, _ = time.Parse(time.RFC3339, validFromStr.(string))
		}

		var validUntil *time.Time
		if vuVal, ok := record.Get("valid_until"); ok && vuVal != nil {
			if vuStr, ok := vuVal.(string); ok && vuStr != "" {
				t, err := time.Parse(time.RFC3339Nano, vuStr)
				if err != nil {
					t, _ = time.Parse(time.RFC3339, vuStr)
				}
				validUntil = &t
			}
		}

		relations = append(relations, Relation{
			ID:         id,
			SourceID:   srcID,
			TargetID:   tgtID,
			Type:       relType.(string),
			ValidFrom:  validFrom,
			ValidUntil: validUntil,
		})
	}
	return relations, nil
}

// parseFalkorFacts is a helper function to deserialize records into Fact structs.
func parseFalkorFacts(res *redisgraph.QueryResult) ([]Fact, error) {
	var facts []Fact
	for res.Next() {
		record := res.Record()

		idVal, _ := record.Get("id")
		id, err := uuid.Parse(idVal.(string))
		if err != nil {
			continue
		}

		entityVal, _ := record.Get("entity_id")
		entityID, err := uuid.Parse(entityVal.(string))
		if err != nil {
			continue
		}

		attr, _ := record.Get("attribute")
		val, _ := record.Get("val")
		confidence, _ := record.Get("confidence_score")

		var confFloat float64
		if cf, ok := confidence.(float64); ok {
			confFloat = cf
		} else if ci, ok := confidence.(int64); ok {
			confFloat = float64(ci)
		}

		validFromStr, _ := record.Get("valid_from")
		validFrom, err := time.Parse(time.RFC3339Nano, validFromStr.(string))
		if err != nil {
			validFrom, _ = time.Parse(time.RFC3339, validFromStr.(string))
		}

		var validUntil *time.Time
		if vuVal, ok := record.Get("valid_until"); ok && vuVal != nil {
			if vuStr, ok := vuVal.(string); ok && vuStr != "" {
				t, err := time.Parse(time.RFC3339Nano, vuStr)
				if err != nil {
					t, _ = time.Parse(time.RFC3339, vuStr)
				}
				validUntil = &t
			}
		}

		srcAgent, _ := record.Get("source_agent")

		facts = append(facts, Fact{
			ID:              id,
			EntityID:        entityID,
			Attribute:       attr.(string),
			Value:           val.(string),
			ConfidenceScore: confFloat,
			ValidFrom:       validFrom,
			ValidUntil:      validUntil,
			SourceAgent:     srcAgent.(string),
		})
	}
	return facts, nil
}

// falkorCosineSimilarity calculates exact cosine distance similarity between two vectors.
func falkorCosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func (s *FalkorDBStore) InsertDocument(ctx context.Context, doc *document.Document) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	metadataBytes, err := json.Marshal(doc.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal document metadata: %w", err)
	}

	query := `
	CREATE (d:Document {
		id: $id,
		title: $title,
		source_type: $source_type,
		source_url: $source_url,
		file_path: $file_path,
		status: $status,
		error_message: $error_message,
		metadata_json: $metadata_json,
		created_at: $created_at,
		updated_at: $updated_at
	})
	`
	params := map[string]interface{}{
		"id":            doc.ID.String(),
		"title":         doc.Title,
		"source_type":   string(doc.SourceType),
		"source_url":    doc.SourceURL,
		"file_path":     doc.FilePath,
		"status":        string(doc.Status),
		"error_message": doc.ErrorMessage,
		"metadata_json": string(metadataBytes),
		"created_at":    doc.CreatedAt.Format(time.RFC3339Nano),
		"updated_at":    doc.UpdatedAt.Format(time.RFC3339Nano),
	}

	_, err = graph.ParameterizedQuery(query, params)
	if err != nil {
		return fmt.Errorf("failed to insert document node: %w", err)
	}
	return nil
}

func (s *FalkorDBStore) GetDocument(ctx context.Context, id uuid.UUID) (*document.Document, error) {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	query := `
	MATCH (d:Document {id: $id})
	RETURN d.id AS id, d.title AS title, d.source_type AS source_type, 
	       d.source_url AS source_url, d.file_path AS file_path, d.status AS status, 
	       d.error_message AS error_message, d.metadata_json AS metadata_json, 
	       d.created_at AS created_at, d.updated_at AS updated_at
	`
	res, err := graph.ParameterizedQuery(query, map[string]interface{}{"id": id.String()})
	if err != nil {
		return nil, err
	}
	if !res.Next() {
		return nil, fmt.Errorf("document not found")
	}
	record := res.Record()
	idStr, _ := record.Get("id")
	docID, _ := uuid.Parse(idStr.(string))
	title, _ := record.Get("title")
	srcType, _ := record.Get("source_type")
	
	var srcURL, filePath, errMsg string
	if val, ok := record.Get("source_url"); ok && val != nil { srcURL = val.(string) }
	if val, ok := record.Get("file_path"); ok && val != nil { filePath = val.(string) }
	if val, ok := record.Get("error_message"); ok && val != nil { errMsg = val.(string) }

	status, _ := record.Get("status")
	
	var metadata map[string]string
	if metaVal, ok := record.Get("metadata_json"); ok && metaVal != nil {
		if metaStr, ok := metaVal.(string); ok && metaStr != "" {
			_ = json.Unmarshal([]byte(metaStr), &metadata)
		}
	}

	createdStr, _ := record.Get("created_at")
	updatedStr, _ := record.Get("updated_at")
	createdAt, _ := time.Parse(time.RFC3339Nano, createdStr.(string))
	updatedAt, _ := time.Parse(time.RFC3339Nano, updatedStr.(string))

	return &document.Document{
		ID:           docID,
		Title:        title.(string),
		SourceType:   document.SourceType(srcType.(string)),
		SourceURL:    srcURL,
		FilePath:     filePath,
		Status:       document.IngestionStatus(status.(string)),
		ErrorMessage: errMsg,
		Metadata:     metadata,
		CreatedAt:    createdAt,
		UpdatedAt:    updatedAt,
	}, nil
}

func (s *FalkorDBStore) UpdateDocumentStatus(ctx context.Context, docID uuid.UUID, status document.IngestionStatus, errMsg string) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	query := `
	MATCH (d:Document {id: $id})
	SET d.status = $status, d.error_message = $error_message, d.updated_at = $updated_at
	`
	params := map[string]interface{}{
		"id":            docID.String(),
		"status":        string(status),
		"error_message": errMsg,
		"updated_at":    time.Now().Format(time.RFC3339Nano),
	}

	_, err := graph.ParameterizedQuery(query, params)
	if err != nil {
		return fmt.Errorf("failed to update document status: %w", err)
	}
	return nil
}

func (s *FalkorDBStore) InsertDocumentChunks(ctx context.Context, chunks []document.DocumentChunk, embeddings [][]float32) error {
	if len(chunks) == 0 {
		return nil
	}

	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	for i, chunk := range chunks {
		metadataBytes, _ := json.Marshal(chunk.Metadata)
		var embedding []interface{}
		if len(embeddings) > i && len(embeddings[i]) > 0 {
			embedding = make([]interface{}, len(embeddings[i]))
			for j, v := range embeddings[i] {
				embedding[j] = float64(v)
			}
		}

		query := `
		MATCH (d:Document {id: $document_id})
		CREATE (c:DocumentChunk {
			id: $id,
			chunk_index: $chunk_index,
			content: $content,
			embedding: $embedding,
			metadata_json: $metadata_json
		})
		CREATE (d)-[:HAS_CHUNK]->(c)
		`
		params := map[string]interface{}{
			"id":            chunk.ID.String(),
			"document_id":   chunk.DocumentID.String(),
			"chunk_index":   chunk.ChunkIndex,
			"content":       chunk.Content,
			"embedding":     embedding,
			"metadata_json": string(metadataBytes),
		}

		_, err := graph.ParameterizedQuery(query, params)
		if err != nil {
			return fmt.Errorf("failed to insert document chunk at index %d: %w", i, err)
		}
	}
	return nil
}

func (s *FalkorDBStore) SearchDocumentChunks(ctx context.Context, queryVector []float32, limit int) ([]document.DocumentChunk, error) {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	query := `
	MATCH (d:Document)-[:HAS_CHUNK]->(c:DocumentChunk)
	RETURN c.id AS id, d.id AS document_id, c.chunk_index AS chunk_index, 
	       c.content AS content, c.metadata_json AS metadata_json, c.embedding AS embedding
	`
	res, err := graph.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve document chunks: %w", err)
	}

	type scoredChunk struct {
		chunk document.DocumentChunk
		score float32
	}

	var scored []scoredChunk

	for res.Next() {
		record := res.Record()

		idVal, _ := record.Get("id")
		id, err := uuid.Parse(idVal.(string))
		if err != nil {
			continue
		}

		docVal, _ := record.Get("document_id")
		docID, err := uuid.Parse(docVal.(string))
		if err != nil {
			continue
		}

		cIdx, _ := record.Get("chunk_index")
		content, _ := record.Get("content")

		var metadata map[string]string
		if metaVal, ok := record.Get("metadata_json"); ok && metaVal != nil {
			if metaStr, ok := metaVal.(string); ok && metaStr != "" {
				_ = json.Unmarshal([]byte(metaStr), &metadata)
			}
		}

		var chunkIndex int
		if ci, ok := cIdx.(int64); ok {
			chunkIndex = int(ci)
		} else if cf, ok := cIdx.(float64); ok {
			chunkIndex = int(cf)
		}

		var chunkVec []float32
		if embVal, ok := record.Get("embedding"); ok && embVal != nil {
			if list, ok := embVal.([]interface{}); ok {
				chunkVec = make([]float32, len(list))
				for j, item := range list {
					if f, ok := item.(float64); ok {
						chunkVec[j] = float32(f)
					}
				}
			}
		}

		chunk := document.DocumentChunk{
			ID:         id,
			DocumentID: docID,
			ChunkIndex: chunkIndex,
			Content:    content.(string),
			Metadata:   metadata,
		}

		score := falkorCosineSimilarity(queryVector, chunkVec)
		scored = append(scored, scoredChunk{
			chunk: chunk,
			score: score,
		})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	n := limit
	if len(scored) < limit {
		n = len(scored)
	}

	chunks := make([]document.DocumentChunk, n)
	for i := 0; i < n; i++ {
		chunks[i] = scored[i].chunk
	}

	return chunks, nil
}

func (s *FalkorDBStore) LinkDocumentToAuthor(ctx context.Context, docID uuid.UUID, authorID uuid.UUID) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	query := `
	MERGE (d:Document {id: $doc_id})
	MERGE (e:Entity {id: $author_id})
	MERGE (e)-[:AUTHORED]->(d)
	`
	params := map[string]interface{}{
		"doc_id":    docID.String(),
		"author_id": authorID.String(),
	}

	_, err := graph.ParameterizedQuery(query, params)
	if err != nil {
		return fmt.Errorf("failed to link document to author: %w", err)
	}
	return nil
}

func (s *FalkorDBStore) LinkDocumentToTopic(ctx context.Context, docID uuid.UUID, topicName string) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	query := `
	MERGE (d:Document {id: $doc_id})
	MERGE (e:Entity {id: $topic_id})
	SET e.attribute = "topic", e.val = $topic_name
	MERGE (d)-[:DISCUSSES]->(e)
	`
	topicID := uuid.NewMD5(uuid.NameSpaceDNS, []byte("topic-"+topicName))
	params := map[string]interface{}{
		"doc_id":     docID.String(),
		"topic_id":   topicID.String(),
		"topic_name": topicName,
	}

	_, err := graph.ParameterizedQuery(query, params)
	if err != nil {
		return fmt.Errorf("failed to link document to topic: %w", err)
	}
	return nil
}

func (s *FalkorDBStore) LinkFactToSource(ctx context.Context, factID uuid.UUID, docID uuid.UUID, chunkID uuid.UUID) error {
	conn := s.pool.Get()
	defer conn.Close()

	graph := redisgraph.GraphNew(s.graphName, conn)

	query := `
	MATCH (f:Fact {id: $fact_id})
	MATCH (d:Document {id: $doc_id})
	MERGE (f)-[:EXTRACTED_FROM]->(d)
	`
	params := map[string]interface{}{
		"fact_id": factID.String(),
		"doc_id":  docID.String(),
	}

	if _, err := graph.ParameterizedQuery(query, params); err != nil {
		return fmt.Errorf("failed to link fact to document: %w", err)
	}

	if chunkID != uuid.Nil {
		chunkQuery := `
		MATCH (f:Fact {id: $fact_id})
		MATCH (c:DocumentChunk {id: $chunk_id})
		MERGE (f)-[:MENTIONED_IN]->(c)
		`
		params["chunk_id"] = chunkID.String()
		if _, err := graph.ParameterizedQuery(chunkQuery, params); err != nil {
			return fmt.Errorf("failed to link fact to chunk: %w", err)
		}
	}

	return nil
}

func (s *FalkorDBStore) GetAllEntities(ctx context.Context) ([]EntityNode, error) {
	return nil, nil
}

func (s *FalkorDBStore) MergeEntities(ctx context.Context, canonicalID, duplicateID uuid.UUID) error {
	return nil
}

func (s *FalkorDBStore) InsertFactWithProvenance(ctx context.Context, fact *Fact, vector []float32, docID uuid.UUID, chunkID uuid.UUID) error {
	if err := s.InsertFact(ctx, fact, vector); err != nil {
		return err
	}
	return s.LinkFactToSource(ctx, fact.ID, docID, chunkID)
}

func (s *FalkorDBStore) InsertCommunitySummary(ctx context.Context, id uuid.UUID, name string, summary string, embedding []float32, entities []uuid.UUID) error {
	return nil
}

func (s *FalkorDBStore) SearchCommunitySummaries(ctx context.Context, queryVector []float32, limit int) ([]CommunitySummary, error) {
	return nil, nil
}



