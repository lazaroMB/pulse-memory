package memory

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type Neo4jStore struct {
	driver neo4j.DriverWithContext
}

// NewNeo4jStore establishes a connection with a Neo4j graph database.
func NewNeo4jStore(uri, username, password string) (*Neo4jStore, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = driver.VerifyConnectivity(ctx)
	if err != nil {
		driver.Close(ctx)
		return nil, fmt.Errorf("failed to verify Neo4j connectivity: %w", err)
	}

	return &Neo4jStore{driver: driver}, nil
}

// Close releases Neo4j driver connection pools.
func (s *Neo4jStore) Close() error {
	return s.driver.Close(context.Background())
}

// InitSchema sets up graph indices and unique ID constraints.
func (s *Neo4jStore) InitSchema(ctx context.Context) error {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		queries := []string{
			"CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
			"CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
			`CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
			 FOR (f:Fact) ON (f.embedding)
			 OPTIONS {
			   indexProvider: 'vector-2.0',
			   nodeProperties: ['embedding'],
			   vectorDimension: 3072,
			   vectorSimilarityFunction: 'cosine'
			 }`,
		}
		for _, q := range queries {
			if _, err := tx.Run(ctx, q, nil); err != nil {
				return nil, err
			}
		}
		return nil, nil
	})
	if err != nil {
		return fmt.Errorf("failed to initialize Neo4j schema: %w", err)
	}
	return nil
}

// InsertFact stores a semantic fact node and merges it into the graph.
func (s *Neo4jStore) InsertFact(ctx context.Context, fact *Fact, vector []float32) error {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)

	// Neo4j maps float64 slices to lists of floating-point values
	embedding := make([]float64, len(vector))
	for i, v := range vector {
		embedding[i] = float64(v)
	}

	var validUntilStr *string
	if fact.ValidUntil != nil {
		s := fact.ValidUntil.Format(time.RFC3339Nano)
		validUntilStr = &s
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
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
		return tx.Run(ctx, query, params)
	})
	if err != nil {
		return fmt.Errorf("failed to insert fact: %w", err)
	}
	return nil
}

// SearchHybrid searches for active facts utilizing either temporal sorting or vector cosine similarity.
func (s *Neo4jStore) SearchHybrid(ctx context.Context, q *MemorySearchQuery) ([]Fact, error) {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer session.Close(ctx)

	if len(q.QueryVector) == 0 {
		result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
			query := `
			MATCH (e:Entity {id: $entity_id})-[:HAS_FACT]->(f:Fact)
			WHERE f.valid_until IS NULL
			RETURN f.id AS id, e.id AS entity_id, f.attribute AS attribute, f.val AS val, 
			       f.confidence_score AS confidence_score, f.valid_from AS valid_from, 
			       f.valid_until AS valid_until, f.source_agent AS source_agent
			ORDER BY f.valid_from DESC
			LIMIT $limit
			`
			res, err := tx.Run(ctx, query, map[string]interface{}{
				"entity_id": q.TargetEntity.String(),
				"limit":     q.MaxResults,
			})
			if err != nil {
				return nil, err
			}
			return parseFacts(ctx, res)
		})
		if err != nil {
			return nil, fmt.Errorf("failed to perform keyword search: %w", err)
		}
		return result.([]Fact), nil
	}

	// Compute native vector similarity
	embedding := make([]float64, len(q.QueryVector))
	for i, v := range q.QueryVector {
		embedding[i] = float64(v)
	}

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		query := `
		MATCH (e:Entity {id: $entity_id})-[:HAS_FACT]->(f:Fact)
		WHERE f.valid_until IS NULL AND f.embedding IS NOT NULL
		RETURN f.id AS id, e.id AS entity_id, f.attribute AS attribute, f.val AS val, 
		       f.confidence_score AS confidence_score, f.valid_from AS valid_from, 
		       f.valid_until AS valid_until, f.source_agent AS source_agent,
		       vector.similarity.cosine(f.embedding, $query_vector) AS score
		ORDER BY score DESC
		LIMIT $limit
		`
		res, err := tx.Run(ctx, query, map[string]interface{}{
			"entity_id":    q.TargetEntity.String(),
			"query_vector": embedding,
			"limit":        q.MaxResults,
		})
		if err != nil {
			return nil, err
		}
		return parseFacts(ctx, res)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to perform hybrid vector search: %w", err)
	}
	return result.([]Fact), nil
}

// DeactivateFact sets the valid_until timestamp of a fact to deactivate it dynamically.
func (s *Neo4jStore) DeactivateFact(ctx context.Context, factID uuid.UUID) error {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		query := `
		MATCH (f:Fact {id: $id})
		WHERE f.valid_until IS NULL
		SET f.valid_until = $valid_until
		`
		params := map[string]interface{}{
			"id":          factID.String(),
			"valid_until": time.Now().Format(time.RFC3339Nano),
		}
		return tx.Run(ctx, query, params)
	})
	if err != nil {
		return fmt.Errorf("failed to deactivate fact: %w", err)
	}
	return nil
}

// InsertRelation adds a directed relationship edge between two entity nodes.
func (s *Neo4jStore) InsertRelation(ctx context.Context, relation *Relation) error {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer session.Close(ctx)

	var validUntilStr *string
	if relation.ValidUntil != nil {
		s := relation.ValidUntil.Format(time.RFC3339Nano)
		validUntilStr = &s
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
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
		return tx.Run(ctx, query, params)
	})
	if err != nil {
		return fmt.Errorf("failed to insert relation: %w", err)
	}
	return nil
}

// GetActiveRelations retrieves all active relationship edges connected to the target entity node.
func (s *Neo4jStore) GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]Relation, error) {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer session.Close(ctx)

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (interface{}, error) {
		query := `
		MATCH (s:Entity)-[r:RELATION]->(t:Entity)
		WHERE (s.id = $entity_id OR t.id = $entity_id) AND r.valid_until IS NULL
		RETURN r.id AS id, s.id AS source_id, t.id AS target_id, r.type AS rel_type, 
		       r.valid_from AS valid_from, r.valid_until AS valid_until
		`
		res, err := tx.Run(ctx, query, map[string]interface{}{
			"entity_id": entityID.String(),
		})
		if err != nil {
			return nil, err
		}

		var relations []Relation
		for res.Next(ctx) {
			record := res.Record()

			idStr, _ := record.Get("id")
			id, err := uuid.Parse(idStr.(string))
			if err != nil {
				continue
			}

			srcStr, _ := record.Get("source_id")
			srcID, err := uuid.Parse(srcStr.(string))
			if err != nil {
				continue
			}

			tgtStr, _ := record.Get("target_id")
			tgtID, err := uuid.Parse(tgtStr.(string))
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
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get active relations: %w", err)
	}
	return result.([]Relation), nil
}

// parseFacts is a helper function to deserialize Cypher query results into fact slices.
func parseFacts(ctx context.Context, res neo4j.ResultWithContext) ([]Fact, error) {
	var facts []Fact
	for res.Next(ctx) {
		record := res.Record()

		idStr, _ := record.Get("id")
		id, err := uuid.Parse(idStr.(string))
		if err != nil {
			continue
		}

		entityStr, _ := record.Get("entity_id")
		entityID, err := uuid.Parse(entityStr.(string))
		if err != nil {
			continue
		}

		attr, _ := record.Get("attribute")
		val, _ := record.Get("val")
		confidence, _ := record.Get("confidence_score")

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
			ConfidenceScore: confidence.(float64),
			ValidFrom:       validFrom,
			ValidUntil:      validUntil,
			SourceAgent:     srcAgent.(string),
		})
	}
	return facts, nil
}
