package consolidation

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/google/uuid"
	"pulse/internal/agent"
	"pulse/internal/memory"
)

// RunCommunityDetection clusters the graph of entities into communities and generates macro narrative summaries using LLM.
func RunCommunityDetection(ctx context.Context, store memory.MemoryStore, llm agent.LLMClient) error {
	log.Println("[Community GraphRAG] Starting community detection and summarization...")

	// 1. Get all active entities
	entities, err := store.GetAllEntities(ctx)
	if err != nil {
		return fmt.Errorf("failed to retrieve entities for clustering: %w", err)
	}

	if len(entities) < 2 {
		log.Println("[Community GraphRAG] Not enough entities to detect communities.")
		return nil
	}

	// 2. Build graph structure from relations
	adj := make(map[uuid.UUID][]uuid.UUID)
	entityMap := make(map[uuid.UUID]memory.EntityNode)
	for _, e := range entities {
		entityMap[e.ID] = e
	}

	// Build undirected adjacency lists from relations
	for _, e := range entities {
		rels, err := store.GetActiveRelations(ctx, e.ID)
		if err != nil {
			continue
		}
		for _, r := range rels {
			adj[r.SourceID] = append(adj[r.SourceID], r.TargetID)
			adj[r.TargetID] = append(adj[r.TargetID], r.SourceID)
		}
	}

	// 3. Run Label Propagation Algorithm (LPA)
	labels := make(map[uuid.UUID]uuid.UUID)
	for _, e := range entities {
		labels[e.ID] = e.ID
	}

	iterations := 5
	for iter := 0; iter < iterations; iter++ {
		changed := false
		for _, e := range entities {
			neighbors := adj[e.ID]
			if len(neighbors) == 0 {
				continue
			}

			// Find the most frequent label among neighbors
			counts := make(map[uuid.UUID]int)
			maxCount := 0
			var bestLabel uuid.UUID

			for _, n := range neighbors {
				l := labels[n]
				counts[l]++
				if counts[l] > maxCount {
					maxCount = counts[l]
					bestLabel = l
				}
			}

			if labels[e.ID] != bestLabel {
				labels[e.ID] = bestLabel
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	// 4. Group entities by community label
	communities := make(map[uuid.UUID][]uuid.UUID)
	for entityID, label := range labels {
		communities[label] = append(communities[label], entityID)
	}

	log.Printf("[Community GraphRAG] Detected %d unique communities across %d entities", len(communities), len(entities))

	// 5. Generate and store narrative summaries for each community with >= 2 entities
	for label, members := range communities {
		if len(members) < 2 {
			continue // Skip single isolated entities to manage LLM API costs
		}

		// Gather facts and relations inside this community
		var factsBuilder strings.Builder
		var relsBuilder strings.Builder

		memberSet := make(map[uuid.UUID]bool)
		memberNames := make([]string, 0, len(members))
		for _, m := range members {
			memberSet[m] = true
			if nameEnt, ok := entityMap[m]; ok {
				memberNames = append(memberNames, nameEnt.Name)
			}
		}

		factsBuilder.WriteString(fmt.Sprintf("Community Members: %s\n\n", strings.Join(memberNames, ", ")))
		factsBuilder.WriteString("Active Facts:\n")

		for _, m := range members {
			facts, err := store.SearchHybrid(ctx, &memory.MemorySearchQuery{
				TargetEntity: m,
				MaxResults:   20,
			})
			if err == nil {
				for _, f := range facts {
					factsBuilder.WriteString(fmt.Sprintf("- Entity '%s': %s: %s\n", entityMap[m].Name, f.Attribute, f.Value))
				}
			}

			rels, err := store.GetActiveRelations(ctx, m)
			if err == nil {
				for _, r := range rels {
					if memberSet[r.TargetID] {
						srcName := entityMap[r.SourceID].Name
						tgtName := entityMap[r.TargetID].Name
						relsBuilder.WriteString(fmt.Sprintf("- %s -> %s (%s)\n", srcName, tgtName, r.Type))
					}
				}
			}
		}

		communityContext := fmt.Sprintf("%s\nRelations:\n%s", factsBuilder.String(), relsBuilder.String())

		// Generate summary narrative
		prompt := fmt.Sprintf(`You are an advanced knowledge graph analysis engine.
Analyze the following community of connected entities, facts, and relations.
Synthesize this information into a cohesive, high-quality "Macro Narrative Summary".
Your summary should act as a high-level overview explaining how these entities relate, their shared context, and their main activities.

Community Data:
%s

Your output must contain only the clean synthesized summary. Be descriptive, objective, and clear. Avoid metadata, conversational intros, or markdown headers.`, communityContext)

		summary, err := llm.GenerateAnswer(ctx, prompt, nil, nil)
		if err != nil {
			log.Printf("[Community GraphRAG] Error generating summary for community %s: %v", label, err)
			continue
		}

		summary = strings.TrimSpace(summary)
		if summary == "" {
			continue
		}

		embedding, err := llm.GenerateEmbedding(ctx, summary)
		if err != nil {
			log.Printf("[Community GraphRAG] Error generating embedding for community summary %s: %v", label, err)
			continue
		}

		communityName := fmt.Sprintf("Community of %s", entityMap[members[0]].Name)
		if len(memberNames) > 1 {
			communityName = fmt.Sprintf("Community of %s and %s", entityMap[members[0]].Name, entityMap[members[1]].Name)
		}

		err = store.InsertCommunitySummary(ctx, label, communityName, summary, embedding, members)
		if err != nil {
			log.Printf("[Community GraphRAG] Error saving community summary %s: %v", label, err)
		} else {
			log.Printf("[Community GraphRAG] Generated and saved community summary: '%s' (%d members)", communityName, len(members))
		}
	}

	return nil
}
