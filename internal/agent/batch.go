package agent

import (
	"context"
	"log"
	"os"
	"strconv"
	"sync"
	"time"
)

type embedRequest struct {
	text   string
	result chan embedResult
}

type embedResult struct {
	embedding []float32
	err       error
}

// BatchingLLMClient wraps an existing LLMClient and intercepts GenerateEmbedding calls
// to route them through a high-throughput dynamic batching queue.
type BatchingLLMClient struct {
	LLMClient
	queue      chan embedRequest
	batchSize  int
	lingerTime time.Duration
	stopChan   chan struct{}
	wg         sync.WaitGroup
	closeOnce  sync.Once
}

// NewBatchingLLMClient creates and starts a new BatchingLLMClient decorator.
func NewBatchingLLMClient(ctx context.Context, client LLMClient) *BatchingLLMClient {
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
		LLMClient:  client,
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

// Close gracefully stops the batcher worker and closes the underlying LLMClient.
func (bc *BatchingLLMClient) Close() {
	bc.closeOnce.Do(func() {
		close(bc.stopChan)
		bc.wg.Wait()
		bc.LLMClient.Close()
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

	flush := func() {
		if len(activeBatch) == 0 {
			return
		}

		texts := make([]string, len(activeBatch))
		for i, req := range activeBatch {
			texts[i] = req.text
		}

		// Run the batch embedding API call
		embeddings, err := bc.LLMClient.GenerateEmbeddings(ctx, texts)

		// Distribute results back to waiting channels
		for i, req := range activeBatch {
			if err != nil {
				req.result <- embedResult{err: err}
			} else {
				req.result <- embedResult{embedding: embeddings[i]}
			}
		}

		activeBatch = nil
	}

	for {
		select {
		case req := <-bc.queue:
			activeBatch = append(activeBatch, req)
			if len(activeBatch) >= bc.batchSize {
				flush()
				ticker.Reset(bc.lingerTime)
			}
		case <-ticker.C:
			flush()
		case <-bc.stopChan:
			flush()
			return
		case <-ctx.Done():
			flush()
			return
		}
	}
}

// ExtractRelations forwards relationship extraction calls to the underlying LLMClient.
func (bc *BatchingLLMClient) ExtractRelations(ctx context.Context, message string) ([]ExtractedRelation, error) {
	return bc.LLMClient.ExtractRelations(ctx, message)
}
