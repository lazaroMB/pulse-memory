package entity

import (
	"time"

	"github.com/google/uuid"
)

// ChatMessage represents a single interaction turn in a chat session.
type ChatMessage struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

// InteractionLog represents the payload passed to the asynchronous consolidation worker pool.
type InteractionLog struct {
	SessionID string
	EntityID  uuid.UUID
	Sender    string
	Message   string
	Timestamp time.Time
}
