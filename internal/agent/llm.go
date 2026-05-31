package agent

import (
	"context"
	"pulse/internal/memory"
)

// ExtractedFact holds the intermediate structured output from the LLM.
type ExtractedFact struct {
	Attribute       string  `json:"attribute" db:"attribute"`
	Value           string  `json:"value" db:"val"`
	ConfidenceScore float64 `json:"confidence_score" db:"confidence_score"`
}

// ExtractedRelation holds structured relationship details extracted by the LLM.
type ExtractedRelation struct {
	SourceEntity string  `json:"source_entity"`
	TargetEntity string  `json:"target_entity"`
	RelationType string  `json:"relation_type"`
	Confidence   float64 `json:"confidence"`
}

// Config contains the configuration variables for all supported LLM API providers.
type Config struct {
	Provider       string // "gemini", "openai"
	APIKey         string // API credential key
	GenModelName   string // Model name for generation (e.g. gemini-2.5-flash or gpt-4o-mini)
	EmbedModelName string // Model name for embeddings (e.g. text-embedding-004 or text-embedding-3-small)
}

// LLMClient defines the boundary for LLM interaction, including vector embeddings,
// conversational answer generation, and structured factual claim extraction.
type LLMClient interface {
	// GenerateEmbedding creates a dense vector representation of the input text
	GenerateEmbedding(ctx context.Context, text string) ([]float32, error)

	// GenerateEmbeddings creates dense vector representations for multiple inputs in a single batch request
	GenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error)

	// GenerateAnswer responds to the user by combining their message with retrieved long-term facts and short-term chat history
	GenerateAnswer(ctx context.Context, message string, history []memory.ChatMessage, facts []memory.Fact) (string, error)

	// ExtractFacts parses raw conversation text to extract new atomic factual claims
	ExtractFacts(ctx context.Context, message string) ([]ExtractedFact, error)

	// ExtractRelations parses raw conversation text to extract structural relationships to other entities
	ExtractRelations(ctx context.Context, message string) ([]ExtractedRelation, error)

	// Close terminates the client and cleans up any open connections/resources
	Close()
}

// NewLLMClient instantiates a concrete LLM provider matching the configured provider.
func NewLLMClient(ctx context.Context, cfg Config) (LLMClient, error) {
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
