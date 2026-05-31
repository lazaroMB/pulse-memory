package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"pulse/internal/memory"
	"sync"
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

	t.Run("GenerateEmbeddings success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Input []string `json:"input"`
				Model string   `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("failed to decode request body: %v", err)
			}

			if len(req.Input) != 2 || req.Input[0] != "hello" || req.Input[1] != "world" {
				t.Errorf("expected inputs ['hello', 'world'], got %v", req.Input)
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			resp := map[string]interface{}{
				"data": []map[string]interface{}{
					{"embedding": []float32{0.1, 0.2, 0.3}},
					{"embedding": []float32{0.4, 0.5, 0.6}},
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

		embs, err := client.GenerateEmbeddings(ctx, []string{"hello", "world"})
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		if len(embs) != 2 {
			t.Fatalf("expected 2 embeddings, got %d", len(embs))
		}
		if len(embs[0]) != 3 || embs[0][0] != 0.1 || embs[1][0] != 0.4 {
			t.Errorf("unexpected embeddings response: %v", embs)
		}
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

		answer, err := client.GenerateAnswer(ctx, "hello", nil, []memory.Fact{
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

type mockLLMClient struct {
	LLMClient
	embeddingsFunc func(ctx context.Context, texts []string) ([][]float32, error)
	closeCalled    bool
}

func (m *mockLLMClient) GenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	return m.embeddingsFunc(ctx, texts)
}

func (m *mockLLMClient) Close() {
	m.closeCalled = true
}

func (m *mockLLMClient) ExtractRelations(ctx context.Context, message string) ([]ExtractedRelation, error) {
	return nil, nil
}

func TestBatchingLLMClient(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var mu sync.Mutex
	var capturedCalls [][]string

	mock := &mockLLMClient{
		embeddingsFunc: func(ctx context.Context, texts []string) ([][]float32, error) {
			mu.Lock()
			capturedCalls = append(capturedCalls, texts)
			mu.Unlock()

			res := make([][]float32, len(texts))
			for i := range texts {
				res[i] = []float32{float32(i) * 1.0}
			}
			return res, nil
		},
	}

	// Create batching client with 2 items max batch size and 50ms linger
	t.Setenv("EMBEDDING_BATCH_SIZE", "2")
	t.Setenv("EMBEDDING_LINGER_MS", "50")
	t.Setenv("EMBEDDING_QUEUE_CAPACITY", "10")

	client := NewBatchingLLMClient(ctx, mock)
	defer client.Close()

	// 1. Test batch size trigger (should immediately flush when 2 items are queued)
	var wg sync.WaitGroup
	wg.Add(2)

	var res1, res2 []float32
	var err1, err2 error

	go func() {
		defer wg.Done()
		res1, err1 = client.GenerateEmbedding(ctx, "hello")
	}()

	go func() {
		defer wg.Done()
		res2, err2 = client.GenerateEmbedding(ctx, "world")
	}()

	wg.Wait()

	if err1 != nil || err2 != nil {
		t.Fatalf("expected no errors, got: %v, %v", err1, err2)
	}

	if len(res1) == 0 || len(res2) == 0 {
		t.Errorf("expected non-empty embeddings, got res1=%v, res2=%v", res1, res2)
	}

	mu.Lock()
	if len(capturedCalls) != 1 {
		t.Errorf("expected exactly 1 batch call, got %d", len(capturedCalls))
	} else {
		call := capturedCalls[0]
		if len(call) != 2 || (call[0] != "hello" && call[1] != "hello") {
			t.Errorf("unexpected batch items: %v", call)
		}
	}
	mu.Unlock()

	// 2. Test close propagation
	client.Close()
	if !mock.closeCalled {
		t.Errorf("expected close call to propagate to underlying client")
	}
}
