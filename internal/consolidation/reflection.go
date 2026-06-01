package consolidation

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/google/uuid"
	"pulse/internal/agent"
	"pulse/internal/memory"
)

type ReflectedInsight struct {
	Attribute   string  `json:"attribute"`
	Value       string  `json:"value"`
	Explanation string  `json:"explanation"`
	Confidence  float64 `json:"confidence"`
}

// RunCognitiveReflection scans active facts and relations, deduces implicit insights, and consolidates them back to storage.
func RunCognitiveReflection(ctx context.Context, store memory.MemoryStore, llm agent.LLMClient) error {
	log.Println("[Reflection Engine] Starting cognitive reflection cycle...")

	// 1. Get all active entities
	entities, err := store.GetAllEntities(ctx)
	if err != nil {
		return fmt.Errorf("failed to retrieve entities: %w", err)
	}

	if len(entities) == 0 {
		return nil
	}

	entityMap := make(map[uuid.UUID]memory.EntityNode)
	for _, e := range entities {
		entityMap[e.ID] = e
	}

	reflectedCount := 0

	// 2. Perform reflection per entity to discover hidden attributes/connections
	for _, ent := range entities {
		// Fetch active facts
		facts, err := store.SearchHybrid(ctx, &memory.MemorySearchQuery{
			TargetEntity: ent.ID,
			MaxResults:   30,
		})
		if err != nil || len(facts) < 2 {
			continue // Skip entities with insufficient known context
		}

		// Fetch active relations
		rels, err := store.GetActiveRelations(ctx, ent.ID)
		if err != nil {
			rels = []memory.Relation{}
		}

		var factsStr strings.Builder
		for _, f := range facts {
			factsStr.WriteString(fmt.Sprintf("- Attribute '%s': %s\n", f.Attribute, f.Value))
		}

		var relsStr strings.Builder
		for _, r := range rels {
			targetName := r.TargetID.String()
			if tNode, ok := entityMap[r.TargetID]; ok {
				targetName = tNode.Name
			}
			sourceName := r.SourceID.String()
			if sNode, ok := entityMap[r.SourceID]; ok {
				sourceName = sNode.Name
			}
			relsStr.WriteString(fmt.Sprintf("- %s -> %s (%s)\n", sourceName, targetName, r.Type))
		}

		prompt := fmt.Sprintf(`You are a deep cognitive reflection engine.
Analyze the following active facts and relationships established for the entity "%s".
Your goal is to perform logical deduction and synthesis to identify at least one implicit, new factual claim or connection that is logically derived from this context but not explicitly stated.

CRITICAL RULES:
- Do NOT repeat any fact that is already explicitly listed.
- Do NOT make wild speculations; the deduction must be a logical inference.
- The attribute must be a clean snake_case string representing the specific deduced category (e.g., "inferred_interest", "implied_professional_scope", "logical_conflict").

Active Facts:
%s

Relationships:
%s

Format the output strictly as a JSON array of objects. Do not include markdown code block formatting (like json). Just output raw JSON.
Each object must contain:
- "attribute": the deduced attribute category in snake_case (string)
- "value": the deduced implicit fact value (string)
- "explanation": brief logical reasoning of how you deduced this (string)
- "confidence": decimal value between 0.0 and 1.0

If no valid logical deductions can be made, output an empty JSON array [].`, ent.Name, factsStr.String(), relsStr.String())

		// Generate reflection using the LLM client
		responseStr, err := llm.GenerateAnswer(ctx, prompt, nil, nil)
		if err != nil {
			log.Printf("[Reflection Engine] LLM generation failed for entity %s: %v", ent.Name, err)
			continue
		}

		cleanedJSON := strings.TrimSpace(responseStr)
		cleanedJSON = strings.TrimPrefix(cleanedJSON, "```json")
		cleanedJSON = strings.TrimPrefix(cleanedJSON, "```")
		cleanedJSON = strings.TrimSuffix(cleanedJSON, "```")
		cleanedJSON = strings.TrimSpace(cleanedJSON)

		if cleanedJSON == "" || cleanedJSON == "[]" {
			continue
		}

		var insights []ReflectedInsight
		if err := json.Unmarshal([]byte(cleanedJSON), &insights); err != nil {
			log.Printf("[Reflection Engine] Failed to unmarshal insights (raw: %s): %v", cleanedJSON, err)
			continue
		}

		for _, ins := range insights {
			if ins.Attribute == "" || ins.Value == "" || ins.Confidence < 0.7 {
				continue
			}

			// Generate embedding for the new derived fact representation
			rep := fmt.Sprintf("%s: %s", ins.Attribute, ins.Value)
			emb, err := llm.GenerateEmbedding(ctx, rep)
			if err != nil {
				continue
			}

			factID := uuid.New()
			newFact := &memory.Fact{
				ID:              factID,
				EntityID:        ent.ID,
				Attribute:       ins.Attribute,
				Value:           ins.Value,
				ConfidenceScore: ins.Confidence,
				ValidFrom:       time.Now(),
				SourceAgent:     "reflection_engine",
				Stability:       30.0, // Default stability for new reflections
			}

			err = store.InsertFact(ctx, newFact, emb)
			if err != nil {
				log.Printf("[Reflection Engine] Error inserting reflected fact: %v", err)
			} else {
				log.Printf("[Reflection Engine] Deduced new implicit fact for '%s': [%s: %s] (Reason: %s)", 
					ent.Name, ins.Attribute, ins.Value, ins.Explanation)
				reflectedCount++
			}
		}
	}

	if reflectedCount > 0 {
		log.Printf("[Reflection Engine] Cognitive reflection cycle completed. Consolidated %d new derived insights.", reflectedCount)
	} else {
		log.Printf("[Reflection Engine] Cognitive reflection cycle completed. No new insights deduced.")
	}

	return nil
}
