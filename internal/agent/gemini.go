package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
	"pulse/internal/memory"
)

type GeminiClient struct {
	client     *genai.Client
	genModel   *genai.GenerativeModel
	embedModel *genai.EmbeddingModel
}

func NewGeminiClient(ctx context.Context, apiKey, genModelName, embedModelName string) (*GeminiClient, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	genModel := client.GenerativeModel(genModelName)
	embedModel := client.EmbeddingModel(embedModelName)

	return &GeminiClient{
		client:     client,
		genModel:   genModel,
		embedModel: embedModel,
	}, nil
}

func (c *GeminiClient) Close() {
	c.client.Close()
}

// GenerateEmbedding creates a 768-dimensional dense vector representation of the input text
func (c *GeminiClient) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	res, err := c.embedModel.EmbedContent(ctx, genai.Text(text))
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}
	if res == nil || res.Embedding == nil {
		return nil, fmt.Errorf("empty embedding returned")
	}
	return res.Embedding.Values, nil
}

// GenerateAnswer responds to the user by combining their message with retrieved long-term facts
func (c *GeminiClient) GenerateAnswer(ctx context.Context, message string, facts []memory.Fact) (string, error) {
	// Construct the context block from active facts
	var contextBuilder strings.Builder
	if len(facts) > 0 {
		contextBuilder.WriteString("The following relevant long-term facts about the user and context are known:\n")
		for _, f := range facts {
			contextBuilder.WriteString(fmt.Sprintf("- %s: %s (Confidence: %.2f)\n", f.Attribute, f.Value, f.ConfidenceScore))
		}
		contextBuilder.WriteString("\nUse the facts above to personalize your answer if relevant. Do not mention the facts explicitly unless asked.\n\n")
	}

	contextBuilder.WriteString(fmt.Sprintf("User Message: %s\nAnswer:", message))

	resp, err := c.genModel.GenerateContent(ctx, genai.Text(contextBuilder.String()))
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %w", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil || len(resp.Candidates[0].Content.Parts) == 0 {
		return "No response generated.", nil
	}

	// Extract the text content from parts
	var replyBuilder strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if textPart, ok := part.(genai.Text); ok {
			replyBuilder.WriteString(string(textPart))
		}
	}

	return replyBuilder.String(), nil
}

// ExtractedFact holds the intermediate structured output from the LLM
type ExtractedFact struct {
	Attribute       string  `json:"attribute"`
	Value           string  `json:"value"`
	ConfidenceScore float64 `json:"confidence_score"`
}

// ExtractFacts parses raw conversation text to extract new atomic factual claims in JSON format
func (c *GeminiClient) ExtractFacts(ctx context.Context, message string) ([]ExtractedFact, error) {
	prompt := fmt.Sprintf(`Analyze the following message and extract any new, explicit, long-term facts about the user's preferences, role, company, or state. 
Do not extract transient details (like "user is saying hello"). Focus only on structural facts that are useful for long-term memory.

Format the output strictly as a JSON array of objects. Do not include markdown code block formatting (like `+"`"+`json). Just output raw JSON.
Each object must contain:
- "attribute": short string (snake_case, e.g. "programming_language", "office_location")
- "value": the actual value (string, e.g. "Go", "Denver")
- "confidence_score": decimal value between 0.0 and 1.0

If no factual claims are present, output an empty JSON array [].

Message: "%s"`, message)

	resp, err := c.genModel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return nil, fmt.Errorf("failed to extract facts: %w", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, nil
	}

	var jsonBuilder strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if textPart, ok := part.(genai.Text); ok {
			jsonBuilder.WriteString(string(textPart))
		}
	}

	// Clean any potential markdown wrapper around JSON
	cleanedJSON := strings.TrimSpace(jsonBuilder.String())
	cleanedJSON = strings.TrimPrefix(cleanedJSON, "```json")
	cleanedJSON = strings.TrimPrefix(cleanedJSON, "```")
	cleanedJSON = strings.TrimSuffix(cleanedJSON, "```")
	cleanedJSON = strings.TrimSpace(cleanedJSON)

	if cleanedJSON == "" || cleanedJSON == "[]" {
		return nil, nil
	}

	var extracted []ExtractedFact
	if err := json.Unmarshal([]byte(cleanedJSON), &extracted); err != nil {
		return nil, fmt.Errorf("failed to unmarshal extracted facts (raw: %s): %w", cleanedJSON, err)
	}

	return extracted, nil
}
