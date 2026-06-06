package chat_test

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"pulse/internal/domain/entity"
	"pulse/internal/usecase/chat"
	"pulse/internal/usecase/ports"
)

// Define mock structures using Go's interface embedding pattern
type mockMemoryRepo struct {
	ports.MemoryRepository
	searchHybridFunc             func(ctx context.Context, query *entity.MemorySearchQuery) ([]entity.Fact, error)
	searchDocumentChunksFunc     func(ctx context.Context, queryVector []float32, limit int) ([]entity.DocumentChunk, error)
	searchCommunitySummariesFunc func(ctx context.Context, queryVector []float32, limit int) ([]entity.CommunitySummary, error)
	getActiveRelationsFunc       func(ctx context.Context, entityID uuid.UUID) ([]entity.Relation, error)
	getActiveRelationsBatchFunc  func(ctx context.Context, entityIDs []uuid.UUID) ([]entity.Relation, error)
	getEntityNamesBatchFunc      func(ctx context.Context, entityIDs []uuid.UUID) (map[uuid.UUID]string, error)
}

func (m *mockMemoryRepo) SearchHybrid(ctx context.Context, query *entity.MemorySearchQuery) ([]entity.Fact, error) {
	if m.searchHybridFunc != nil {
		return m.searchHybridFunc(ctx, query)
	}
	return nil, nil
}

func (m *mockMemoryRepo) SearchDocumentChunks(ctx context.Context, queryVector []float32, limit int) ([]entity.DocumentChunk, error) {
	if m.searchDocumentChunksFunc != nil {
		return m.searchDocumentChunksFunc(ctx, queryVector, limit)
	}
	return nil, nil
}

func (m *mockMemoryRepo) SearchCommunitySummaries(ctx context.Context, queryVector []float32, limit int) ([]entity.CommunitySummary, error) {
	if m.searchCommunitySummariesFunc != nil {
		return m.searchCommunitySummariesFunc(ctx, queryVector, limit)
	}
	return nil, nil
}

func (m *mockMemoryRepo) GetActiveRelations(ctx context.Context, entityID uuid.UUID) ([]entity.Relation, error) {
	if m.getActiveRelationsFunc != nil {
		return m.getActiveRelationsFunc(ctx, entityID)
	}
	return nil, nil
}

func (m *mockMemoryRepo) GetActiveRelationsBatch(ctx context.Context, entityIDs []uuid.UUID) ([]entity.Relation, error) {
	if m.getActiveRelationsBatchFunc != nil {
		return m.getActiveRelationsBatchFunc(ctx, entityIDs)
	}
	return nil, nil
}

func (m *mockMemoryRepo) GetEntityNamesBatch(ctx context.Context, entityIDs []uuid.UUID) (map[uuid.UUID]string, error) {
	if m.getEntityNamesBatchFunc != nil {
		return m.getEntityNamesBatchFunc(ctx, entityIDs)
	}
	return nil, nil
}

type mockChatMemoryRepo struct {
	ports.ChatMemoryRepository
	appendMessageFunc     func(ctx context.Context, sessionID string, msg entity.ChatMessage) error
	getSessionHistoryFunc func(ctx context.Context, sessionID string, limit int) ([]entity.ChatMessage, error)
}

func (m *mockChatMemoryRepo) AppendMessage(ctx context.Context, sessionID string, msg entity.ChatMessage) error {
	if m.appendMessageFunc != nil {
		return m.appendMessageFunc(ctx, sessionID, msg)
	}
	return nil
}

func (m *mockChatMemoryRepo) GetSessionHistory(ctx context.Context, sessionID string, limit int) ([]entity.ChatMessage, error) {
	if m.getSessionHistoryFunc != nil {
		return m.getSessionHistoryFunc(ctx, sessionID, limit)
	}
	return nil, nil
}

type mockLLMService struct {
	ports.LLMService
	generateEmbeddingFunc func(ctx context.Context, text string) ([]float32, error)
	generateAnswerFunc     func(ctx context.Context, message string, history []entity.ChatMessage, facts []entity.Fact) (string, error)
}

func (m *mockLLMService) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	if m.generateEmbeddingFunc != nil {
		return m.generateEmbeddingFunc(ctx, text)
	}
	return []float32{0.1, 0.2}, nil
}

func (m *mockLLMService) GenerateAnswer(ctx context.Context, message string, history []entity.ChatMessage, facts []entity.Fact) (string, error) {
	if m.generateAnswerFunc != nil {
		return m.generateAnswerFunc(ctx, message, history, facts)
	}
	return "Mocked answer", nil
}

type mockSemanticCache struct {
	ports.SemanticCache
	getFunc        func(ctx context.Context, queryVector []float32) (string, bool, error)
	setFunc        func(ctx context.Context, queryText string, queryVector []float32, replyText string) error
	invalidateFunc func(ctx context.Context, entityID uuid.UUID) error
}

func (m *mockSemanticCache) Get(ctx context.Context, queryVector []float32) (string, bool, error) {
	if m.getFunc != nil {
		return m.getFunc(ctx, queryVector)
	}
	return "", false, nil
}

func (m *mockSemanticCache) Set(ctx context.Context, queryText string, queryVector []float32, replyText string) error {
	if m.setFunc != nil {
		return m.setFunc(ctx, queryText, queryVector, replyText)
	}
	return nil
}

type mockPrivacyService struct {
	ports.PrivacyService
	scrubTextFunc      func(ctx context.Context, text string) (string, error)
	validateAccessFunc func(ctx context.Context, agentRole string, fact *entity.Fact) bool
}

func (m *mockPrivacyService) ScrubText(ctx context.Context, text string) (string, error) {
	if m.scrubTextFunc != nil {
		return m.scrubTextFunc(ctx, text)
	}
	return text, nil
}

func (m *mockPrivacyService) ValidateAccess(ctx context.Context, agentRole string, fact *entity.Fact) bool {
	if m.validateAccessFunc != nil {
		return m.validateAccessFunc(ctx, agentRole, fact)
	}
	return true
}

type mockConsolidationService struct {
	ports.ConsolidationService
	queueInteractionFunc func(log entity.InteractionLog)
	queueDocumentFunc    func(job entity.DocumentJob)
}

func (m *mockConsolidationService) QueueInteraction(log entity.InteractionLog) {
	if m.queueInteractionFunc != nil {
		m.queueInteractionFunc(log)
	}
}

func (m *mockConsolidationService) QueueDocument(job entity.DocumentJob) {
	if m.queueDocumentFunc != nil {
		m.queueDocumentFunc(job)
	}
}

func TestChatUseCase_Execute_CacheHit(t *testing.T) {
	// Setup mocks
	cacheCalled := false
	cache := &mockSemanticCache{
		getFunc: func(ctx context.Context, queryVector []float32) (string, bool, error) {
			cacheCalled = true
			return "Cached response", true, nil
		},
	}

	privacy := &mockPrivacyService{}
	llm := &mockLLMService{}
	store := &mockMemoryRepo{}
	chatMem := &mockChatMemoryRepo{}
	worker := &mockConsolidationService{}

	uc := chat.NewChatUseCase(store, chatMem, llm, cache, privacy, worker)

	input := chat.ChatInput{
		SessionID: "session-1",
		EntityID:  uuid.New(),
		AgentRole: "user",
		Message:   "Hello",
	}

	output, err := uc.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if !cacheCalled {
		t.Error("Expected cache.Get to be called")
	}

	if output.ResponseMessage != "Cached response" {
		t.Errorf("Expected response 'Cached response', got: %s", output.ResponseMessage)
	}
}

func TestChatUseCase_Execute_CacheMiss(t *testing.T) {
	// Setup mocks
	llmAnswerCalled := false
	llm := &mockLLMService{
		generateAnswerFunc: func(ctx context.Context, message string, history []entity.ChatMessage, facts []entity.Fact) (string, error) {
			llmAnswerCalled = true
			return "Generated answer", nil
		},
	}

	cache := &mockSemanticCache{}
	privacy := &mockPrivacyService{
		scrubTextFunc: func(ctx context.Context, text string) (string, error) {
			return text + " scrubbed", nil
		},
	}

	storeCalled := false
	store := &mockMemoryRepo{
		searchHybridFunc: func(ctx context.Context, query *entity.MemorySearchQuery) ([]entity.Fact, error) {
			storeCalled = true
			return []entity.Fact{
				{
					ID:        uuid.New(),
					Attribute: "name",
					Value:     "Alice",
				},
			}, nil
		},
	}

	chatMem := &mockChatMemoryRepo{}
	worker := &mockConsolidationService{}

	uc := chat.NewChatUseCase(store, chatMem, llm, cache, privacy, worker)

	input := chat.ChatInput{
		SessionID: "session-2",
		EntityID:  uuid.New(),
		AgentRole: "user",
		Message:   "Who am I?",
	}

	output, err := uc.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if !storeCalled {
		t.Error("Expected store.SearchHybrid to be called")
	}

	if !llmAnswerCalled {
		t.Error("Expected llm.GenerateAnswer to be called")
	}

	if output.ResponseMessage != "Generated answer" {
		t.Errorf("Expected response 'Generated answer', got: %s", output.ResponseMessage)
	}
}
