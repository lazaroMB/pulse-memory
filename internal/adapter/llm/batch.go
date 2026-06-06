package llm

import (
	"context"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"pulse/internal/usecase/ports"
)

type embedRequest struct {
	text   string
	result chan embedResult
}

type embedResult struct {
	embedding []float32
	err       error
}

// BatchingLLMClient wraps an existing ports.LLMService and intercepts GenerateEmbedding calls
// to route them through a high-throughput dynamic batching queue.
type BatchingLLMClient struct {
	ports.LLMService
	queue      chan embedRequest
	batchSize  int
	lingerTime time.Duration
	stopChan   chan struct{}
	wg         sync.WaitGroup
	closeOnce  sync.Once
}

// NewBatchingLLMClient creates and starts a new BatchingLLMClient decorator.
func NewBatchingLLMClient(ctx context.Context, client ports.LLMService) *BatchingLLMClient {
	batchSize := 100
	if val := os.Getenv("EMBEDDING_BATCH_SIZE"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			batchSize = parsed
		}
	}

	lingerMs := 10
	if val := os.Getenv("EMBEDDING_LINGER_MS"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			lingerMs = parsed
		}
	}

	queueCap := 5000
	if val := os.Getenv("EMBEDDING_QUEUE_CAPACITY"); val != "" {
		if parsed, err := strconv.Atoi(val); err == nil && parsed > 0 {
			queueCap = parsed
		}
	}

	bc := &BatchingLLMClient{
		LLMService: client,
		queue:      make(chan embedRequest, queueCap),
		batchSize:  batchSize,
		lingerTime: time.Duration(lingerMs) * time.Millisecond,
		stopChan:   make(chan struct{}),
	}

	bc.wg.Add(1)
	go bc.run(ctx)

	log.Printf("[BatchingLLMClient] Started dynamic batcher queue (batchSize=%d, linger=%s, capacity=%d)",
		bc.batchSize, bc.lingerTime, queueCap)

	return bc
}

// Close gracefully stops the batcher worker and closes the underlying LLMService.
func (bc *BatchingLLMClient) Close() {
	bc.closeOnce.Do(func() {
		close(bc.stopChan)
		bc.wg.Wait()
		bc.LLMService.Close()
	})
}

// GenerateEmbedding queues the embedding request and waits for the batch processor to return the result.
func (bc *BatchingLLMClient) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	resChan := make(chan embedResult, 1)
	req := embedRequest{
		text:   text,
		result: resChan,
	}

	select {
	case bc.queue <- req:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	select {
	case res := <-resChan:
		return res.embedding, res.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (bc *BatchingLLMClient) run(ctx context.Context) {
	defer bc.wg.Done()

	var activeBatch []embedRequest
	ticker := time.NewTicker(bc.lingerTime)
	defer ticker.Stop()

	for {
		select {
		case req := <-bc.queue:
			activeBatch = append(activeBatch, req)
			if len(activeBatch) >= bc.batchSize {
				bc.processBatch(ctx, activeBatch)
				activeBatch = nil
			}

		case <-ticker.C:
			if len(activeBatch) > 0 {
				bc.processBatch(ctx, activeBatch)
				activeBatch = nil
			}

		case <-bc.stopChan:
			// Process any remaining items before exiting
			if len(activeBatch) > 0 {
				bc.processBatch(ctx, activeBatch)
			}
			// Drain queue
			for {
				select {
				case req := <-bc.queue:
					activeBatch = append(activeBatch, req)
					if len(activeBatch) >= bc.batchSize {
						bc.processBatch(ctx, activeBatch)
						activeBatch = nil
					}
				default:
					if len(activeBatch) > 0 {
						bc.processBatch(ctx, activeBatch)
					}
					return
				}
			}
		}
	}
}

func (bc *BatchingLLMClient) processBatch(ctx context.Context, batch []embedRequest) {
	texts := make([]string, len(batch))
	for i, req := range batch {
		texts[i] = req.text
	}

	// We use a background context or a timeout context to ensure embeddings are written even if the individual request context is cancelled
	callCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	embeddings, err := bc.LLMService.GenerateEmbeddings(callCtx, texts)
	if err != nil {
		for _, req := range batch {
			req.result <- embedResult{err: err}
		}
		return
	}

	for i, req := range batch {
		req.result <- embedResult{embedding: embeddings[i]}
	}
}
