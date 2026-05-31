package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"pulse/internal/memory"
	"strings"
	"time"
)

// OpenAIClient implements the LLMClient interface using OpenAI's public REST APIs.
type OpenAIClient struct {
	apiKey         string
	genModelName   string
	embedModelName string
	httpClient     *http.Client
}

// NewOpenAIClient initializes an OpenAI API client.
func NewOpenAIClient(ctx context.Context, apiKey, genModelName, embedModelName string) (*OpenAIClient, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("openai API key is required")
	}
	if genModelName == "" {
		genModelName = "gpt-4o-mini"
	}
	if embedModelName == "" {
		embedModelName = "text-embedding-3-small"
	}

	return &OpenAIClient{
		apiKey:         apiKey,
		genModelName:   genModelName,
		embedModelName: embedModelName,
		httpClient:     &http.Client{Timeout: 30 * time.Second},
	}, nil
}

// Close is a no-op as the standard net/http client does not require manual closing.
func (c *OpenAIClient) Close() {}

// GenerateEmbedding creates a dense vector representation of the input text using OpenAI's Embeddings API.
func (c *OpenAIClient) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	reqBody, err := json.Marshal(map[string]interface{}{
		"input": text,
		"model": c.embedModelName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embeddings request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/embeddings", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create embeddings request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embeddings request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai embeddings API returned status %d: %s", resp.StatusCode, string(body))
	}

	var res struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode embeddings response: %w", err)
	}

	if len(res.Data) == 0 {
		return nil, fmt.Errorf("openai embeddings API returned empty data")
	}

	return res.Data[0].Embedding, nil
}

// GenerateEmbeddings creates dense vector representations for multiple inputs in a single batch request using OpenAI's Embeddings API.
func (c *OpenAIClient) GenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	reqBody, err := json.Marshal(map[string]interface{}{
		"input": texts,
		"model": c.embedModelName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embeddings request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/embeddings", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create embeddings request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embeddings request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai embeddings API returned status %d: %s", resp.StatusCode, string(body))
	}

	var res struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode embeddings response: %w", err)
	}

	if len(res.Data) != len(texts) {
		return nil, fmt.Errorf("openai embeddings API returned %d embeddings, expected %d", len(res.Data), len(texts))
	}

	embeddings := make([][]float32, len(res.Data))
	for i, d := range res.Data {
		embeddings[i] = d.Embedding
	}

	return embeddings, nil
}

// GenerateAnswer responds to the user by combining their message with retrieved long-term facts and short-term chat history using OpenAI's Chat Completions API.
func (c *OpenAIClient) GenerateAnswer(ctx context.Context, message string, history []memory.ChatMessage, facts []memory.Fact) (string, error) {
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

	reqBody, err := json.Marshal(map[string]interface{}{
		"model": c.genModelName,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": contextBuilder.String(),
			},
		},
	})
	if err != nil {
		return "", fmt.Errorf("failed to marshal chat completion request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create chat completion request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("chat completion request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("openai chat completion API returned status %d: %s", resp.StatusCode, string(body))
	}

	var res struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", fmt.Errorf("failed to decode chat completion response: %w", err)
	}

	if len(res.Choices) == 0 {
		return "", fmt.Errorf("openai chat completion API returned zero choices")
	}

	return res.Choices[0].Message.Content, nil
}

// ExtractFacts parses raw conversation text to extract new atomic factual claims in JSON format.
func (c *OpenAIClient) ExtractFacts(ctx context.Context, message string) ([]ExtractedFact, error) {
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

	reqBody, err := json.Marshal(map[string]interface{}{
		"model": c.genModelName,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": prompt,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal fact extraction request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create fact extraction request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fact extraction request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai fact extraction API returned status %d: %s", resp.StatusCode, string(body))
	}

	var res struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode fact extraction response: %w", err)
	}

	if len(res.Choices) == 0 {
		return nil, fmt.Errorf("openai fact extraction API returned zero choices")
	}

	cleanedJSON := strings.TrimSpace(res.Choices[0].Message.Content)
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

// ExtractRelations parses raw conversation text to extract relationships between the subject and other entities.
func (c *OpenAIClient) ExtractRelations(ctx context.Context, message string) ([]ExtractedRelation, error) {
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

	reqBody, err := json.Marshal(map[string]interface{}{
		"model": c.genModelName,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": prompt,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal relation extraction request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create relation extraction request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("relation extraction request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai relation extraction API returned status %d: %s", resp.StatusCode, string(body))
	}

	var res struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("failed to decode relation extraction response: %w", err)
	}

	if len(res.Choices) == 0 {
		return nil, fmt.Errorf("openai relation extraction API returned zero choices")
	}

	cleanedJSON := strings.TrimSpace(res.Choices[0].Message.Content)
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
