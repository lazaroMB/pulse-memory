package memory

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/google/uuid"
	redisgraph "github.com/falkordb/falkordb-go"
	"github.com/gomodule/redigo/redis"
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
