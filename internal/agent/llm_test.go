package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"pulse/internal/memory"
	"testing"
)

func TestNewLLMClientFactory(t *testing.T) {
	ctx := context.Background()

	t.Run("gemini provider initialization", func(t *testing.T) {
		cfg := Config{
			Provider:       "gemini",
			APIKey:         "dummy-gemini-key",
			GenModelName:   "gemini-2.5-flash",
			EmbedModelName: "text-embedding-004",
		}
		client, err := NewLLMClient(ctx, cfg)
		if err != nil {
			t.Fatalf("expected no error initializing gemini client, got %v", err)
		}
		defer client.Close()

		if _, ok := client.(*GeminiClient); !ok {
			t.Errorf("expected client to be of type *GeminiClient")
		}
	})

	t.Run("openai provider initialization", func(t *testing.T) {
		cfg := Config{
			Provider:       "openai",
			APIKey:         "dummy-openai-key",
			GenModelName:   "gpt-4o-mini",
			EmbedModelName: "text-embedding-3-small",
		}
		client, err := NewLLMClient(ctx, cfg)
		if err != nil {
			t.Fatalf("expected no error initializing openai client, got %v", err)
		}
		defer client.Close()

		if _, ok := client.(*OpenAIClient); !ok {
			t.Errorf("expected client to be of type *OpenAIClient")
		}
	})
}

func TestOpenAIClient(t *testing.T) {
	ctx := context.Background()

	t.Run("GenerateEmbedding success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				t.Errorf("expected POST request, got %s", r.Method)
			}
			if r.Header.Get("Authorization") != "Bearer test-key" {
				t.Errorf("expected Bearer authorization header, got %s", r.Header.Get("Authorization"))
			}

			var req struct {
				Input string `json:"input"`
				Model string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("failed to decode request body: %v", err)
			}

			if req.Input != "hello world" {
				t.Errorf("expected input 'hello world', got '%s'", req.Input)
			}
			if req.Model != "text-embedding-3-small" {
				t.Errorf("expected model 'text-embedding-3-small', got '%s'", req.Model)
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			resp := map[string]interface{}{
				"data": []map[string]interface{}{
					{
						"embedding": []float32{0.1, 0.2, 0.3},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		client := &OpenAIClient{
			apiKey:         "test-key",
			genModelName:   "gpt-4o-mini",
			embedModelName: "text-embedding-3-small",
			httpClient:     server.Client(),
		}

		// Inject mock URL
		originalURL := "https://api.openai.com/v1/embeddings"
		// To route request to our local test server, we temporarily redefine the URL inside test scope
		// (We can use a helper or write custom request builder. Let's make it hit the server directly).
		// Wait, in OpenAIClient we hardcode the URL. Let's see: how can we route it?
		// We can easily inject a transport that overrides the host!
		// Yes, Go's standard http.Client has a Transport field where we can rewrite the URL host.
		client.httpClient.Transport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			req.URL.Scheme = "http"
			req.URL.Host = server.Listener.Addr().String()
			return http.DefaultTransport.RoundTrip(req)
		})

		emb, err := client.GenerateEmbedding(ctx, "hello world")
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		if len(emb) != 3 || emb[0] != 0.1 || emb[1] != 0.2 || emb[2] != 0.3 {
			t.Errorf("unexpected embedding response: %v", emb)
		}
		_ = originalURL
	})

	t.Run("GenerateAnswer success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Model    string `json:"model"`
				Messages []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("failed to decode request body: %v", err)
			}

			if len(req.Messages) == 0 {
				t.Fatalf("expected messages, got none")
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			resp := map[string]interface{}{
				"choices": []map[string]interface{}{
					{
						"message": map[string]interface{}{
							"content": "this is the answer",
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		client := &OpenAIClient{
			apiKey:         "test-key",
			genModelName:   "gpt-4o-mini",
			embedModelName: "text-embedding-3-small",
			httpClient:     server.Client(),
		}
		client.httpClient.Transport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			req.URL.Scheme = "http"
			req.URL.Host = server.Listener.Addr().String()
			return http.DefaultTransport.RoundTrip(req)
		})

		answer, err := client.GenerateAnswer(ctx, "hello", []memory.Fact{
			{Attribute: "name", Value: "Alice", ConfidenceScore: 0.95},
		})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		if answer != "this is the answer" {
			t.Errorf("expected answer 'this is the answer', got '%s'", answer)
		}
	})

	t.Run("ExtractFacts success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			resp := map[string]interface{}{
				"choices": []map[string]interface{}{
					{
						"message": map[string]interface{}{
							"content": `[{"attribute":"fav_color","value":"blue","confidence_score":0.9}]`,
						},
					},
				},
			}
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		client := &OpenAIClient{
			apiKey:         "test-key",
			genModelName:   "gpt-4o-mini",
			embedModelName: "text-embedding-3-small",
			httpClient:     server.Client(),
		}
		client.httpClient.Transport = roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			req.URL.Scheme = "http"
			req.URL.Host = server.Listener.Addr().String()
			return http.DefaultTransport.RoundTrip(req)
		})

		facts, err := client.ExtractFacts(ctx, "I like blue")
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		if len(facts) != 1 || facts[0].Attribute != "fav_color" || facts[0].Value != "blue" || facts[0].ConfidenceScore != 0.9 {
			t.Errorf("unexpected facts extracted: %+v", facts)
		}
	})
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
