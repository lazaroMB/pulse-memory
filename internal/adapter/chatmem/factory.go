package chatmem

import (
	"fmt"

	"pulse/internal/usecase/ports"
)

// ChatMemoryConfig holds parameters needed to instantiate a short-term chat memory provider.
type ChatMemoryConfig struct {
	Provider string // "redis", "valkey", "in-memory"
	URL      string // E.g., "localhost:6379"
}

// NewChatMemory instantiates a concrete ChatMemoryRepository provider matching the configuration.
func NewChatMemory(cfg ChatMemoryConfig) (ports.ChatMemoryRepository, error) {
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
