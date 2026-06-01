package consolidation

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"pulse/internal/agent"
	"pulse/internal/memory"
)

func TestIsSingularAttribute(t *testing.T) {
	tests := []struct {
		attr     string
		expected bool
	}{
		// Singular / Mutually Exclusive Attributes
		{"user_name", true},
		{"company", true},
		{"company_city", true},
		{"company_country", true},
		{"email", true},
		{"preferred_programming_language", true},
		{"current_city", true},

		// Cumulative / List-like Attributes (with former_/past_/visited_ prefixes)
		{"former_company", false},
		{"former_company_city", false},
		{"former_company_country", false},
		{"past_injury", false},
		{"past_hospitalization", false},
		{"visited_city", false},
		{"visited_country", false},

		// Cumulative / List-like Attributes (with suffix patterns)
		{"travel_history", false},
		{"programming_list", false},
		{"reading_hobbies", false},
		{"scientific_interests", false},

		// Specific cumulative keywords and composite attributes
		{"hospitalization", false},
		{"injury_history", false},
		{"hobby", false},
		{"interest", false},
		{"user_preference_hobby", false},
		{"user_preference_interest", false},
		{"allergy", false},
		{"medication", false},
	}

	for _, tt := range tests {
		t.Run(tt.attr, func(t *testing.T) {
			result := isSingularAttribute(tt.attr)
			if result != tt.expected {
				t.Errorf("expected isSingularAttribute(%q) to be %v, got %v", tt.attr, tt.expected, result)
			}
		})
	}
}

type dummyMemoryStore struct {
	memory.MemoryStore
}

func (d *dummyMemoryStore) SearchHybrid(ctx context.Context, query *memory.MemorySearchQuery) ([]memory.Fact, error) {
	return nil, nil
}

func (d *dummyMemoryStore) GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]memory.Relation, error) {
	return nil, nil
}

type recordingLLMClient struct {
	agent.LLMClient
	LastFactMessage     string
	LastRelationMessage string
}

func (r *recordingLLMClient) ExtractFacts(ctx context.Context, message string) ([]agent.ExtractedFact, error) {
	r.LastFactMessage = message
	return nil, nil
}

func (r *recordingLLMClient) ExtractRelations(ctx context.Context, message string) ([]agent.ExtractedRelation, error) {
	r.LastRelationMessage = message
	return nil, nil
}

func TestProcessJobHistoryFiltering(t *testing.T) {
	ctx := context.Background()
	store := &dummyMemoryStore{}
	chatMemory := memory.NewInMemoryChatMemory()
	llm := &recordingLLMClient{}

	wp := NewWorkerPool(store, chatMemory, llm, 10, 1)

	sessionID := "test-session"
	entityID := uuid.New()

	// 1. Add historical message
	err := chatMemory.AppendMessage(ctx, sessionID, memory.ChatMessage{
		Role:      "developer_agent",
		Content:   "I love chocolate ice cream",
		Timestamp: time.Now().Add(-10 * time.Minute),
	})
	if err != nil {
		t.Fatalf("failed to append message: %v", err)
	}

	err = chatMemory.AppendMessage(ctx, sessionID, memory.ChatMessage{
		Role:      "assistant",
		Content:   "That is nice!",
		Timestamp: time.Now().Add(-9 * time.Minute),
	})
	if err != nil {
		t.Fatalf("failed to append message: %v", err)
	}

	// 2. Add current message and assistant reply to simulate main.go timing
	currentMsg := "Trabajo en google, me gusta el helado"
	err = chatMemory.AppendMessage(ctx, sessionID, memory.ChatMessage{
		Role:      "developer_agent",
		Content:   currentMsg,
		Timestamp: time.Now(),
	})
	if err != nil {
		t.Fatalf("failed to append message: %v", err)
	}

	err = chatMemory.AppendMessage(ctx, sessionID, memory.ChatMessage{
		Role:      "assistant",
		Content:   "Qué interesante...",
		Timestamp: time.Now(),
	})
	if err != nil {
		t.Fatalf("failed to append message: %v", err)
	}

	// 3. Process job
	job := InteractionLog{
		SessionID: sessionID,
		EntityID:  entityID,
		Sender:    "developer_agent",
		Message:   currentMsg,
		Timestamp: time.Now(),
	}

	wp.processJob(ctx, 0, job)

	// Verify that history used for extraction contains the old history but excludes the current message and assistant response
	expectedContextPrefix := "Conversation Context:\ndeveloper_agent: I love chocolate ice cream\nassistant: That is nice!\n"
	expectedMessage := "\nMessage to process:\ndeveloper_agent: Trabajo en google, me gusta el helado"

	if !strings.HasPrefix(llm.LastFactMessage, expectedContextPrefix) {
		t.Errorf("expected context prefix %q, got %q", expectedContextPrefix, llm.LastFactMessage)
	}

	if !strings.HasSuffix(llm.LastFactMessage, expectedMessage) {
		t.Errorf("expected suffix %q, got %q", expectedMessage, llm.LastFactMessage)
	}
}

