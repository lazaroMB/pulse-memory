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

// GenerateEmbeddings creates dense vector representations for multiple inputs in a single batch request
func (c *GeminiClient) GenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	batch := c.embedModel.NewBatch()
	for _, text := range texts {
		batch.AddContent(genai.Text(text))
	}
	res, err := c.embedModel.BatchEmbedContents(ctx, batch)
	if err != nil {
		return nil, fmt.Errorf("failed to generate batch embeddings: %w", err)
	}
	if res == nil || len(res.Embeddings) != len(texts) {
		return nil, fmt.Errorf("unexpected batch embedding response length: got %d, expected %d", len(res.Embeddings), len(texts))
	}
	embeddings := make([][]float32, len(res.Embeddings))
	for i, emb := range res.Embeddings {
		if emb == nil {
			return nil, fmt.Errorf("nil embedding at index %d", i)
		}
		embeddings[i] = emb.Values
	}
	return embeddings, nil
}

// GenerateAnswer responds to the user by combining their message with retrieved long-term facts and short-term chat history
func (c *GeminiClient) GenerateAnswer(ctx context.Context, message string, history []memory.ChatMessage, facts []memory.Fact) (string, error) {
	// Construct the context block from active facts
	var contextBuilder strings.Builder
	if len(facts) > 0 {
		contextBuilder.WriteString("The following relevant long-term facts about the user and context are known:\n")
		for _, f := range facts {
			contextBuilder.WriteString(fmt.Sprintf("- %s: %s (Confidence: %.2f)\n", f.Attribute, f.Value, f.ConfidenceScore))
		}
		contextBuilder.WriteString("\nUse the facts above to personalize your answer if relevant. Do not mention the facts explicitly unless asked.\n\n")
	}

	// Inject short-term chat history
	if len(history) > 0 {
		contextBuilder.WriteString("Conversation History:\n")
		for _, msg := range history {
			contextBuilder.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
		}
		contextBuilder.WriteString("\n")
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

// ExtractFacts parses raw conversation text to extract new atomic factual claims in JSON format
func (c *GeminiClient) ExtractFacts(ctx context.Context, message string) ([]ExtractedFact, error) {
	prompt := fmt.Sprintf(`You will be provided with a message to analyze, which may optionally be accompanied by recent "Conversation Context".

If "Conversation Context" is provided, use it ONLY to resolve references (such as pronouns like "she", "he", "it", "they" or relative references) to their actual entities. 
CRITICAL: Do NOT extract any facts that were already established or discussed in the "Conversation Context". Focus EXCLUSIVELY on extracting NEW facts mentioned in the "Message to process".

Analyze the message and extract any new, explicit, long-term facts about the user's preferences, role, company, state, or history. 
Do not extract transient details (like "user is saying hello"). Focus only on structural facts that are useful for long-term memory.

Ensure that facts about past events, history, or previous states preserve their temporal context (e.g. dates, years, or "former" status) in the attribute or value, so they are not misconstrued as current states. 
Attributes related to past or former states (like previous jobs, past locations, former companies) MUST be strictly prefixed with "former_" or "past_" (e.g., "former_company", "former_company_city", "former_company_country") so they do not conflict with or overwrite current active states (like "company", "company_city", "company_country") in the database.
For example:
- If the user says "In 2019 I broke my leg", extract attribute "past_injury" or "injury_history" with value "broke leg in 2019" (do NOT extract attribute "injury" with value "broken leg" which implies a currently active injury).
- If the user says "I worked at Envato in Melbourne in 2016", extract "former_company" with value "Envato", "former_company_city" with value "Melbourne (2016)", and "former_company_country" with value "Australia (2016)" (do NOT use generic "company_city" or "company_country" as that would overwrite current company details).
- If the user says "I used to work at Google", extract attribute "former_employer" with value "Google".

Format the output strictly as a JSON array of objects. Do not include markdown code block formatting (like `+"`"+`json). Just output raw JSON.
Each object must contain:
- "attribute": short string (snake_case, e.g. "programming_language", "past_injury", "former_company")
- "value": the actual value (string, e.g. "Go", "broke leg in 2019")
- "confidence_score": decimal value between 0.0 and 1.0

If no new factual claims are present in the latest message, output an empty JSON array [].

Input:
%s`, message)

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

// ExtractRelations parses raw conversation text to extract relationships between the subject and other entities
func (c *GeminiClient) ExtractRelations(ctx context.Context, message string) ([]ExtractedRelation, error) {
	prompt := fmt.Sprintf(`You will be provided with a message to analyze, which may optionally be accompanied by recent "Conversation Context".

If "Conversation Context" is provided, use it ONLY to resolve references (such as pronouns like "she", "he", "it", "they" or relative references) to their actual entities.
CRITICAL: Do NOT extract any relationships that were already established or discussed in the "Conversation Context". Focus EXCLUSIVELY on extracting NEW structural relationships mentioned in the "Message to process".

Analyze the message and extract any structural relationships between the speaker (user/subject) and other distinct entities mentioned, OR between any of the entities mentioned in the text.
Only extract relationships that represent a long-term connection. Do not extract transient actions.

Format the output strictly as a JSON array of objects. Do not include markdown code block formatting (like `+"`"+`json). Just output raw JSON.
Each object must contain:
- "source_entity": the name of the source entity (string, use "user" if it refers to the speaker/user, e.g., "user", "Pepe")
- "target_entity": the name of the target entity (string, e.g. "Google", "Python", "New York", "Alice")
- "relation_type": the type of relationship in UPPERCASE snake_case (string, e.g. "WORKS_AT", "DEVELOPED_IN", "LIVES_IN", "KNOWS", "USES", "LOVES", "FRIEND_OF")
- "confidence": decimal value between 0.0 and 1.0

If no new relationships are present, output an empty JSON array [].

Input:
%s`, message)

	resp, err := c.genModel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return nil, fmt.Errorf("failed to extract relations: %w", err)
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

	cleanedJSON := strings.TrimSpace(jsonBuilder.String())
	cleanedJSON = strings.TrimPrefix(cleanedJSON, "```json")
	cleanedJSON = strings.TrimPrefix(cleanedJSON, "```")
	cleanedJSON = strings.TrimSuffix(cleanedJSON, "```")
	cleanedJSON = strings.TrimSpace(cleanedJSON)

	if cleanedJSON == "" || cleanedJSON == "[]" {
		return nil, nil
	}

	var extracted []ExtractedRelation
	if err := json.Unmarshal([]byte(cleanedJSON), &extracted); err != nil {
		return nil, fmt.Errorf("failed to unmarshal extracted relations (raw: %s): %w", cleanedJSON, err)
	}

	return extracted, nil
}
