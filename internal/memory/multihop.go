package memory

import (
	"context"
	"fmt"
	"log"

	"github.com/google/uuid"
)

type MultiHopFact struct {
	Fact Fact
	Hop  int
	Path string // e.g. "Seed -[FRIEND_OF]-> Alice -[WORKS_AT]-> Google"
}

// MultiHopRetriever performs recursive relation traversal to retrieve rich multi-hop graph context.
type MultiHopRetriever struct {
	Store MemoryStore
}

// NewMultiHopRetriever instantiates a concrete MultiHopRetriever
func NewMultiHopRetriever(store MemoryStore) *MultiHopRetriever {
	return &MultiHopRetriever{Store: store}
}

// Retrieve performs up to maxHops recursive traversals to collect deep relationship-based facts.
func (mhr *MultiHopRetriever) Retrieve(ctx context.Context, queryText string, queryVector []float32, startEntity uuid.UUID, maxHops int) ([]MultiHopFact, error) {
	if maxHops <= 0 {
		maxHops = 2
	}

	log.Printf("[Multi-Hop GraphRAG] Starting recursive traversal from %s (Max Hops: %d)", startEntity, maxHops)
	
	visitedEntities := make(map[uuid.UUID]bool)
	var resultFacts []MultiHopFact

	// Queue for BFS traversal
	type queueItem struct {
		EntityID uuid.UUID
		Hop      int
		Path     string
	}
	queue := []queueItem{{EntityID: startEntity, Hop: 0, Path: "Seed"}}

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if visitedEntities[curr.EntityID] || curr.Hop > maxHops {
			continue
		}
		visitedEntities[curr.EntityID] = true

		// 1. Fetch active facts for the current entity
		facts, err := mhr.Store.SearchHybrid(ctx, &MemorySearchQuery{
			QueryText:    queryText,
			QueryVector:  queryVector,
			TargetEntity: curr.EntityID,
			MaxResults:   10,
		})
		if err == nil {
			for _, f := range facts {
				resultFacts = append(resultFacts, MultiHopFact{
					Fact: f,
					Hop:  curr.Hop,
					Path: curr.Path,
				})
			}
		}

		// If we reached the maximum hop limit, don't expand neighbors
		if curr.Hop >= maxHops {
			continue
		}

		// 2. Fetch active relations for the current entity to find connected neighbors
		rels, err := mhr.Store.GetActiveRelations(ctx, curr.EntityID)
		if err != nil {
			continue
		}

		for _, r := range rels {
			neighborID := r.TargetID
			if neighborID == curr.EntityID {
				neighborID = r.SourceID
			}

			if visitedEntities[neighborID] {
				continue
			}

			nextPath := fmt.Sprintf("%s -[%s]-> %s", curr.Path, r.Type, neighborID.String()[:8])
			queue = append(queue, queueItem{
				EntityID: neighborID,
				Hop:      curr.Hop + 1,
				Path:     nextPath,
			})
		}
	}

	log.Printf("[Multi-Hop GraphRAG] Traversal completed. Retrieved %d multi-hop context facts.", len(resultFacts))
	return resultFacts, nil
}
