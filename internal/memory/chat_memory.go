package memory

import (
	"context"
	"fmt"
	"time"
)

// ChatMessage represents a single interaction turn in a chat session.
type ChatMessage struct {
	Role      string    `json:"role"`      // E.g., "user", "assistant", or specific agent role
	Content   string    `json:"content"`   // The scrubbed message content
	Timestamp time.Time `json:"timestamp"` // Interaction timestamp
}

// ChatMemory defines the boundary for storing and retrieving conversational short-term memory.
type ChatMemory interface {
	// AppendMessage adds a message to the end of the session history.
	AppendMessage(ctx context.Context, sessionID string, msg ChatMessage) error

	// GetSessionHistory retrieves the last N messages of the session history in chronological order.
	GetSessionHistory(ctx context.Context, sessionID string, limit int) ([]ChatMessage, error)

	// ClearSession removes all history associated with the session.
	ClearSession(ctx context.Context, sessionID string) error

	// Close terminates any open connections or resources associated with the memory store.
	Close() error
}

// ChatMemoryConfig holds parameters needed to instantiate a short-term chat memory provider.
type ChatMemoryConfig struct {
	Provider string // "redis", "valkey", "in-memory"
	URL      string // E.g., "localhost:6379"
}

// NewChatMemory instantiates a concrete ChatMemory provider matching the configuration.
func NewChatMemory(cfg ChatMemoryConfig) (ChatMemory, error) {
	switch cfg.Provider {
	case "redis", "valkey":
		if cfg.URL == "" {
			cfg.URL = "localhost:6379"
		}
		return NewRedisChatMemory(cfg.URL)
	case "in-memory":
		return NewInMemoryChatMemory(), nil
	default:
		return nil, fmt.Errorf("unknown chat memory provider: %s", cfg.Provider)
	}
}
