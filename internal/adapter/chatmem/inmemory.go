package chatmem

import (
	"context"
	"sync"

	"pulse/internal/domain/entity"
)

// InMemoryChatMemory implements ChatMemoryRepository using an in-memory map protected by a mutex.
type InMemoryChatMemory struct {
	mu       sync.RWMutex
	sessions map[string][]entity.ChatMessage
}

// NewInMemoryChatMemory instantiates a new InMemoryChatMemory provider.
func NewInMemoryChatMemory() *InMemoryChatMemory {
	return &InMemoryChatMemory{
		sessions: make(map[string][]entity.ChatMessage),
	}
}

// AppendMessage appends a message to the session's conversational history and caps it at 50 messages.
func (m *InMemoryChatMemory) AppendMessage(ctx context.Context, sessionID string, msg entity.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	history := m.sessions[sessionID]
	history = append(history, msg)
	if len(history) > 50 {
		history = history[len(history)-50:]
	}
	m.sessions[sessionID] = history
	return nil
}

// GetSessionHistory retrieves up to the last 'limit' messages in chronological order.
func (m *InMemoryChatMemory) GetSessionHistory(ctx context.Context, sessionID string, limit int) ([]entity.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	history, exists := m.sessions[sessionID]
	if !exists {
		return []entity.ChatMessage{}, nil
	}

	if len(history) <= limit {
		// Return copy to prevent slice mutation race conditions
		res := make([]entity.ChatMessage, len(history))
		copy(res, history)
		return res, nil
	}

	res := make([]entity.ChatMessage, limit)
	copy(res, history[len(history)-limit:])
	return res, nil
}

// ClearSession purges all history for a session ID.
func (m *InMemoryChatMemory) ClearSession(ctx context.Context, sessionID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.sessions, sessionID)
	return nil
}

// Close implements cleanup interface method.
func (m *InMemoryChatMemory) Close() error {
	return nil
}
