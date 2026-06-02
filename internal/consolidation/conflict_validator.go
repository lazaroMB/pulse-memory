package consolidation

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/google/uuid"
	"pulse/internal/agent"
	"pulse/internal/memory"
)

type ConflictResult struct {
	HasConflict     bool   `json:"has_conflict"`
	ConflictingFact string `json:"conflicting_fact"`
	Reason          string `json:"reason"`
}

// ConflictValidator implements semantic logic verification using LLM ValidateConflict interface
type ConflictValidator struct {
	LLM   agent.LLMClient
	Store memory.MemoryStore
}

// NewConflictValidator instantiates a concrete ConflictValidator
func NewConflictValidator(llm agent.LLMClient, store memory.MemoryStore) *ConflictValidator {
	return &ConflictValidator{
		LLM:   llm,
		Store: store,
	}
}

// CheckConflict validates if a candidate fact has a logical contradiction with existing active facts of an entity.
// It returns the validation result, and if a conflict is found, it attempts to match it to an existing memory.Fact.
func (cv *ConflictValidator) CheckConflict(ctx context.Context, entityID uuid.UUID, candidateFact agent.ExtractedFact, activeFacts []memory.Fact) (*ConflictResult, *memory.Fact, error) {
	if len(activeFacts) == 0 {
		return &ConflictResult{HasConflict: false}, nil, nil
	}

	candidateStr := fmt.Sprintf("%s: %s", candidateFact.Attribute, candidateFact.Value)

	// Collect string representations of all active facts
	existingStrs := make([]string, len(activeFacts))
	factMap := make(map[string]memory.Fact)
	for i, f := range activeFacts {
		rep := fmt.Sprintf("%s: %s", f.Attribute, f.Value)
		existingStrs[i] = rep
		factMap[rep] = f
	}

	// Call LLM logic layer
	jsonResp, err := cv.LLM.ValidateConflict(ctx, candidateStr, existingStrs)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute LLM ValidateConflict: %w", err)
	}

	// Parse JSON
	var result ConflictResult
	if err := json.Unmarshal([]byte(jsonResp), &result); err != nil {
		log.Printf("[Conflict Validator] Warning: failed to parse JSON response from ValidateConflict (raw: %s): %v", jsonResp, err)
		return &ConflictResult{HasConflict: false}, nil, nil
	}

	if !result.HasConflict {
		return &result, nil, nil
	}

	// Try to find the exact existing Fact struct that matches the conflicting_fact returned by the LLM
	var matchedFact *memory.Fact
	conflictingFactTrimmed := strings.TrimSpace(strings.ToLower(result.ConflictingFact))

	for rep, f := range factMap {
		repTrimmed := strings.TrimSpace(strings.ToLower(rep))
		valTrimmed := strings.TrimSpace(strings.ToLower(f.Value))

		// Try exact representation match, or substring match, or value match
		if repTrimmed == conflictingFactTrimmed ||
			strings.Contains(repTrimmed, conflictingFactTrimmed) ||
			strings.Contains(conflictingFactTrimmed, repTrimmed) ||
			valTrimmed == conflictingFactTrimmed ||
			strings.Contains(conflictingFactTrimmed, valTrimmed) {
			matchedFact = &f
			break
		}
	}

	// Fallback lookup: if the LLM reason or conflicting_fact mentions an attribute that matches, match it
	if matchedFact == nil {
		for _, f := range activeFacts {
			attrLower := strings.ToLower(f.Attribute)
			if strings.Contains(conflictingFactTrimmed, attrLower) || strings.Contains(strings.ToLower(result.Reason), attrLower) {
				matchedFact = &f
				break
			}
		}
	}

	return &result, matchedFact, nil
}
