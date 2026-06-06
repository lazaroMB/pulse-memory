package llm

import (
	"context"

	"pulse/internal/usecase/ports"
)

// Config contains the configuration variables for all supported LLM API providers.
type Config struct {
	Provider       string // "gemini", "openai"
	APIKey         string // API credential key
	GenModelName   string // Model name for generation (e.g. gemini-2.5-flash or gpt-4o-mini)
	EmbedModelName string // Model name for embeddings (e.g. text-embedding-004 or text-embedding-3-small)
}

// NewLLMService instantiates a concrete LLM provider matching the configured provider.
func NewLLMService(ctx context.Context, cfg Config) (ports.LLMService, error) {
	switch cfg.Provider {
	case "gemini":
		return NewGeminiClient(ctx, cfg.APIKey, cfg.GenModelName, cfg.EmbedModelName)
	case "openai":
		return NewOpenAIClient(ctx, cfg.APIKey, cfg.GenModelName, cfg.EmbedModelName)
	default:
		// Default to gemini if not specified or unrecognized for backward compatibility
		return NewGeminiClient(ctx, cfg.APIKey, cfg.GenModelName, cfg.EmbedModelName)
	}
}
