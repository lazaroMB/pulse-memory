package consolidation

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	"pulse/internal/agent"
	"pulse/internal/memory"
)

type InteractionLog struct {
	SessionID   string
	EntityID    uuid.UUID
	Sender      string
	Message     string
	Timestamp   time.Time
}

type WorkerPool struct {
	JobQueue    chan InteractionLog
	Store       memory.MemoryStore
	Gemini      *agent.GeminiClient
	MaxWorkers  int
	stopChan    chan struct{}
}

func NewWorkerPool(store memory.MemoryStore, gemini *agent.GeminiClient, queueSize int, maxWorkers int) *WorkerPool {
	return &WorkerPool{
		JobQueue:   make(chan InteractionLog, queueSize),
		Store:      store,
		Gemini:     gemini,
		MaxWorkers: maxWorkers,
		stopChan:   make(chan struct{}),
	}
}

func (wp *WorkerPool) Start(ctx context.Context) {
	for i := 0; i < wp.MaxWorkers; i++ {
		go wp.worker(ctx, i)
	}
}

func (wp *WorkerPool) Stop() {
	close(wp.stopChan)
}

func (wp *WorkerPool) worker(ctx context.Context, id int) {
	log.Printf("[Worker %d] Started background memory consolidation worker", id)
	for {
		select {
		case job := <-wp.JobQueue:
			wp.processJob(ctx, id, job)
		case <-wp.stopChan:
			log.Printf("[Worker %d] Stopped", id)
			return
		case <-ctx.Done():
			log.Printf("[Worker %d] Context cancelled, stopping", id)
			return
		}
	}
}

func (wp *WorkerPool) processJob(ctx context.Context, workerID int, job InteractionLog) {
	log.Printf("[Worker %d] Ingesting message from session %s for fact extraction", workerID, job.SessionID)

	// 1. Run Gemini fact extraction
	extracted, err := wp.Gemini.ExtractFacts(ctx, job.Message)
	if err != nil {
		log.Printf("[Worker %d] Error extracting facts: %v", workerID, err)
		return
	}

	if len(extracted) == 0 {
		log.Printf("[Worker %d] No long-term facts found in message.", workerID)
		return
	}

	// 2. Fetch active facts to reconcile and resolve conflicts
	activeFacts, err := wp.Store.SearchHybrid(ctx, &memory.MemorySearchQuery{
		TargetEntity: job.EntityID,
		MaxResults:   100, // Fetch all active facts for comparison
	})
	if err != nil {
		log.Printf("[Worker %d] Error fetching active facts for entity: %v", workerID, err)
		return
	}

	// Create a lookup map of active facts by attribute for fast conflict resolution
	activeMap := make(map[string]memory.Fact)
	for _, f := range activeFacts {
		activeMap[f.Attribute] = f
	}

	for _, ext := range extracted {
		existing, exists := activeMap[ext.Attribute]

		// Scenario A: Fact already exists with the SAME value
		if exists && existing.Value == ext.Value {
			log.Printf("[Worker %d] Fact already exists and matches: [%s: %s]. Skipping.", workerID, ext.Attribute, ext.Value)
			continue
		}

		// Scenario B: Fact already exists but has a DIFFERENT value (Conflict detected!)
		if exists && existing.Value != ext.Value {
			log.Printf("[Worker %d] Conflict detected for attribute '%s'. Old: '%s', New: '%s'. Resolving...", 
				workerID, ext.Attribute, existing.Value, ext.Value)
			
			// Deactivate the old, stale fact
			if err := wp.Store.DeactivateFact(ctx, existing.ID); err != nil {
				log.Printf("[Worker %d] Error deactivating stale fact %s: %v", workerID, existing.ID, err)
				continue
			}
		}

		// Scenario C: New fact, or resolved conflict (insert new active fact)
		// Generate the embedding for the semantic representation: "attribute: value"
		representation := fmt.Sprintf("%s: %s", ext.Attribute, ext.Value)
		
		// Rate Limit Defense: Add a 500ms delay before each embedding call
		// to prevent hitting 429 Rate Limits on Google Gemini Free Tier keys
		time.Sleep(500 * time.Millisecond)

		embedding, err := wp.Gemini.GenerateEmbedding(ctx, representation)
		if err != nil {
			log.Printf("[Worker %d] Error generating embedding for fact [%s: %s]: %v", workerID, ext.Attribute, ext.Value, err)
			continue
		}

		newFact := &memory.Fact{
			ID:              uuid.New(),
			EntityID:        job.EntityID,
			Attribute:       ext.Attribute,
			Value:           ext.Value,
			ConfidenceScore: ext.ConfidenceScore,
			ValidFrom:       time.Now(),
			ValidUntil:      nil,
			SourceAgent:     job.Sender,
		}

		if err := wp.Store.InsertFact(ctx, newFact, embedding); err != nil {
			log.Printf("[Worker %d] Error inserting new fact [%s: %s]: %v", workerID, ext.Attribute, ext.Value, err)
			continue
		}

		log.Printf("[Worker %d] Consolidated and stored new fact: [%s: %s]", workerID, ext.Attribute, ext.Value)
	}
}
