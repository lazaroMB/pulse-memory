package consolidation

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
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
	JobQueue      chan InteractionLog
	DocumentQueue chan DocumentJob
	Store         memory.MemoryStore
	ChatMemory    memory.ChatMemory
	LLM           agent.LLMClient
	Cache         memory.SemanticCache
	MaxWorkers    int
	stopChan      chan struct{}
}

func NewWorkerPool(store memory.MemoryStore, chatMemory memory.ChatMemory, llm agent.LLMClient, cache memory.SemanticCache, queueSize int, maxWorkers int) *WorkerPool {
	return &WorkerPool{
		JobQueue:      make(chan InteractionLog, queueSize),
		DocumentQueue: make(chan DocumentJob, queueSize),
		Store:         store,
		ChatMemory:    chatMemory,
		LLM:           llm,
		Cache:         cache,
		MaxWorkers:    maxWorkers,
		stopChan:      make(chan struct{}),
	}
}

func (wp *WorkerPool) Start(ctx context.Context) {
	for i := 0; i < wp.MaxWorkers; i++ {
		go wp.worker(ctx, i)
	}

	// Start background Entity Resolution loop
	go wp.entityResolutionLoop(ctx)

	// Start background Community GraphRAG Detection loop
	go wp.communityDetectionLoop(ctx)

	// Start background Cognitive Reflection loop
	go wp.reflectionLoop(ctx)
}

func (wp *WorkerPool) entityResolutionLoop(ctx context.Context) {
	interval := 5 * time.Minute
	if val := os.Getenv("ENTITY_RESOLUTION_INTERVAL"); val != "" {
		if parsed, err := time.ParseDuration(val); err == nil && parsed > 0 {
			interval = parsed
		}
	}

	log.Printf("[Worker Pool] Started background Entity Resolution loop (Interval: %v)", interval)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Initial warmup delay before running the first scan
	select {
	case <-time.After(30 * time.Second):
		_ = RunEntityResolution(ctx, wp.Store)
	case <-ctx.Done():
		return
	case <-wp.stopChan:
		return
	}

	for {
		select {
		case <-ticker.C:
			_ = RunEntityResolution(ctx, wp.Store)
		case <-ctx.Done():
			log.Println("[Worker Pool] Stopping background Entity Resolution loop (Context cancelled)")
			return
		case <-wp.stopChan:
			log.Println("[Worker Pool] Stopping background Entity Resolution loop (Pool stopped)")
			return
		}
	}
}

func (wp *WorkerPool) communityDetectionLoop(ctx context.Context) {
	interval := 15 * time.Minute
	if val := os.Getenv("COMMUNITY_RESOLUTION_INTERVAL"); val != "" {
		if parsed, err := time.ParseDuration(val); err == nil && parsed > 0 {
			interval = parsed
		}
	}

	log.Printf("[Worker Pool] Started background Community GraphRAG Detection loop (Interval: %v)", interval)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Initial warmup delay before running the first scan
	select {
	case <-time.After(1 * time.Minute):
		_ = RunCommunityDetection(ctx, wp.Store, wp.LLM)
	case <-ctx.Done():
		return
	case <-wp.stopChan:
		return
	}

	for {
		select {
		case <-ticker.C:
			_ = RunCommunityDetection(ctx, wp.Store, wp.LLM)
		case <-ctx.Done():
			log.Println("[Worker Pool] Stopping background Community GraphRAG Detection loop (Context cancelled)")
			return
		case <-wp.stopChan:
			log.Println("[Worker Pool] Stopping background Community GraphRAG Detection loop (Pool stopped)")
			return
		}
	}
}

func (wp *WorkerPool) reflectionLoop(ctx context.Context) {
	interval := 30 * time.Minute
	if val := os.Getenv("REFLECTION_INTERVAL"); val != "" {
		if parsed, err := time.ParseDuration(val); err == nil && parsed > 0 {
			interval = parsed
		}
	}

	log.Printf("[Worker Pool] Started background Cognitive Reflection loop (Interval: %v)", interval)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Initial warmup delay before running the first scan (2 minutes to let entities settle)
	select {
	case <-time.After(2 * time.Minute):
		_ = RunCognitiveReflection(ctx, wp.Store, wp.LLM)
	case <-ctx.Done():
		return
	case <-wp.stopChan:
		return
	}

	for {
		select {
		case <-ticker.C:
			_ = RunCognitiveReflection(ctx, wp.Store, wp.LLM)
		case <-ctx.Done():
			log.Println("[Worker Pool] Stopping background Cognitive Reflection loop (Context cancelled)")
			return
		case <-wp.stopChan:
			log.Println("[Worker Pool] Stopping background Cognitive Reflection loop (Pool stopped)")
			return
		}
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
		case docJob := <-wp.DocumentQueue:
			wp.processDocumentJob(ctx, id, docJob)
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
	log.Printf("[Worker %d] Ingesting message from session %s for fact & relation extraction", workerID, job.SessionID)

	ctx = memory.WithAgentOwner(ctx, job.EntityID)

	// Fetch conversational context from short-term memory
	history, err := wp.ChatMemory.GetSessionHistory(ctx, job.SessionID, 10)
	if err != nil {
		log.Printf("[Worker %d] Error retrieving chat history for session %s: %v", workerID, job.SessionID, err)
		history = []memory.ChatMessage{}
	}

	// Filter out the current message and any subsequent messages (like the assistant's response) from the history context
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == job.Sender && history[i].Content == job.Message {
			history = history[:i]
			break
		}
	}

	// Format conversational transcript context to allow pronoun resolution (e.g. "she" -> Emily)
	var dialogBuilder strings.Builder
	if len(history) > 0 {
		dialogBuilder.WriteString("Conversation Context:\n")
		for _, msg := range history {
			dialogBuilder.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
		}
		dialogBuilder.WriteString("\nMessage to process:\n")
	}
	dialogBuilder.WriteString(fmt.Sprintf("%s: %s", job.Sender, job.Message))
	contextMsg := dialogBuilder.String()

	// 1. Run LLM fact and relation extraction concurrently
	var (
		extractedFacts []agent.ExtractedFact
		factsErr       error
		extractedRels  []agent.ExtractedRelation
		relsErr        error
		wgExtract      sync.WaitGroup
	)

	wgExtract.Add(2)
	go func() {
		defer wgExtract.Done()
		extractedFacts, factsErr = wp.LLM.ExtractFacts(ctx, contextMsg)
	}()
	go func() {
		defer wgExtract.Done()
		extractedRels, relsErr = wp.LLM.ExtractRelations(ctx, contextMsg)
	}()
	wgExtract.Wait()

	if factsErr != nil {
		log.Printf("[Worker %d] Error extracting facts: %v", workerID, factsErr)
	}
	if relsErr != nil {
		log.Printf("[Worker %d] Error extracting relations: %v", workerID, relsErr)
	}

	// 2. Process Facts
	if len(extractedFacts) > 0 {
		activeFacts, err := wp.Store.SearchHybrid(ctx, &memory.MemorySearchQuery{
			TargetEntity: job.EntityID,
			MaxResults:   100, // Fetch all active facts for comparison
		})
		if err != nil {
			log.Printf("[Worker %d] Error fetching active facts for entity: %v", workerID, err)
		} else {
			// Create a lookup map of active facts by attribute for fast conflict resolution
			activeMap := make(map[string]memory.Fact)
			for _, f := range activeFacts {
				activeMap[f.Attribute] = f
			}

			var wgFacts sync.WaitGroup
			for _, ext := range extractedFacts {
				existing, exists := activeMap[ext.Attribute]

				// Scenario A: Fact already exists with the SAME value
				if exists && existing.Value == ext.Value {
					log.Printf("[Worker %d] Fact already exists and matches: [%s: %s]. Skipping.", workerID, ext.Attribute, ext.Value)
					continue
				}

				// Scenario B: Fact already exists but has a DIFFERENT value (Conflict detected!)
				if exists && existing.Value != ext.Value {
					if isSingularAttribute(ext.Attribute) {
						log.Printf("[Worker %d] Conflict detected for singular attribute '%s'. Old: '%s', New: '%s'. Resolving...", 
							workerID, ext.Attribute, existing.Value, ext.Value)
						
						// Deactivate the old, stale fact
						if err := wp.Store.DeactivateFact(ctx, existing.ID); err != nil {
							log.Printf("[Worker %d] Error deactivating stale fact %s: %v", workerID, existing.ID, err)
							continue
						}
					} else {
						log.Printf("[Worker %d] Non-exclusive attribute '%s' has new value '%s'. Preserving existing value '%s' to allow coexistence.", 
							workerID, ext.Attribute, ext.Value, existing.Value)
					}
				}

				// Scenario C: New fact, or resolved conflict (insert new active fact)
				// Process embedding and storage concurrently to leverage the dynamic embedding queue
				wgFacts.Add(1)
				go func(ext agent.ExtractedFact) {
					defer wgFacts.Done()

					// Run logical conflict validator against other active facts of this entity
					validator := NewConflictValidator(wp.LLM, wp.Store)
					conflictRes, conflictingFact, err := validator.CheckConflict(ctx, job.EntityID, ext, activeFacts)
					if err == nil && conflictRes != nil && conflictRes.HasConflict {
						log.Printf("[Worker %d] Logical conflict detected for candidate [%s: %s] against active facts. Reason: %s", 
							workerID, ext.Attribute, ext.Value, conflictRes.Reason)
						if conflictingFact != nil {
							log.Printf("[Worker %d] Deactivating conflicting fact [%s: %s] (ID: %s) to resolve contradiction",
								workerID, conflictingFact.Attribute, conflictingFact.Value, conflictingFact.ID)
							if err := wp.Store.DeactivateFact(ctx, conflictingFact.ID); err != nil {
								log.Printf("[Worker %d] Error deactivating conflicting fact: %v", workerID, err)
							}
						}
					}

					representation := fmt.Sprintf("%s: %s", ext.Attribute, ext.Value)

					embedding, err := wp.LLM.GenerateEmbedding(ctx, representation)
					if err != nil {
						log.Printf("[Worker %d] Error generating embedding for fact [%s: %s]: %v", workerID, ext.Attribute, ext.Value, err)
						return
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
						AgentOwner:      job.EntityID,
					}

					if err := wp.Store.InsertFact(ctx, newFact, embedding); err != nil {
						log.Printf("[Worker %d] Error inserting new fact [%s: %s]: %v", workerID, ext.Attribute, ext.Value, err)
						return
					}

					log.Printf("[Worker %d] Consolidated and stored new fact: [%s: %s]", workerID, ext.Attribute, ext.Value)
				}(ext)
			}
			wgFacts.Wait()
		}
	} else {
		log.Printf("[Worker %d] No long-term facts found in message.", workerID)
	}

	// 3. Process Relations
	if len(extractedRels) > 0 {
		var wgRels sync.WaitGroup
		for _, r := range extractedRels {
			wgRels.Add(1)
			go func(rel agent.ExtractedRelation) {
				defer wgRels.Done()

				senderName := strings.TrimSpace(strings.ToLower(job.Sender))

				// 1. Resolve target ID
				targetName := strings.TrimSpace(strings.ToLower(rel.TargetEntity))
				if targetName == "" {
					return
				}
				var targetID uuid.UUID
				if targetName == "user" || targetName == "subject" || targetName == "john" || targetName == "speaker" || targetName == senderName {
					targetID = job.EntityID
				} else {
					targetID = uuid.NewMD5(uuid.NameSpaceDNS, []byte(targetName))

					// Insert target name fact to resolve UUID -> Name later
					targetRep := fmt.Sprintf("name: %s", rel.TargetEntity)
					targetEmb, err := wp.LLM.GenerateEmbedding(ctx, targetRep)
					if err == nil {
						_ = wp.Store.InsertFact(ctx, &memory.Fact{
							ID:              uuid.New(),
							EntityID:        targetID,
							Attribute:       "name",
							Value:           rel.TargetEntity,
							ConfidenceScore: 1.0,
							ValidFrom:       time.Now(),
							SourceAgent:     job.Sender,
							AgentOwner:      job.EntityID,
						}, targetEmb)
					}
				}

				// 2. Resolve source ID
				var sourceID uuid.UUID
				sourceName := strings.TrimSpace(strings.ToLower(rel.SourceEntity))
				if sourceName == "" || sourceName == "user" || sourceName == "subject" || sourceName == "john" || sourceName == "speaker" || sourceName == senderName {
					sourceID = job.EntityID
				} else {
					sourceID = uuid.NewMD5(uuid.NameSpaceDNS, []byte(sourceName))

					// Insert source name fact to resolve UUID -> Name later
					sourceRep := fmt.Sprintf("name: %s", rel.SourceEntity)
					sourceEmb, err := wp.LLM.GenerateEmbedding(ctx, sourceRep)
					if err == nil {
						_ = wp.Store.InsertFact(ctx, &memory.Fact{
							ID:              uuid.New(),
							EntityID:        sourceID,
							Attribute:       "name",
							Value:           rel.SourceEntity,
							ConfidenceScore: 1.0,
							ValidFrom:       time.Now(),
							SourceAgent:     job.Sender,
							AgentOwner:      job.EntityID,
						}, sourceEmb)
					}
				}

				// 2. Check if relation already exists in the database to prevent duplicate writes
				existingRels, err := wp.Store.GetActiveRelations(ctx, sourceID)
				if err == nil {
					for _, extRel := range existingRels {
						if extRel.TargetID == targetID && strings.ToUpper(extRel.Type) == strings.ToUpper(rel.RelationType) {
							log.Printf("[Worker %d] Relation already exists in database: [%s -> %s (%s)]. Skipping write.",
								workerID, sourceName, targetName, extRel.Type)
							return
						}
					}
				}

				relation := &memory.Relation{
					ID:         uuid.New(),
					SourceID:   sourceID,
					TargetID:   targetID,
					Type:       strings.ToUpper(strings.TrimSpace(rel.RelationType)),
					AgentOwner: job.EntityID,
					ValidFrom:  time.Now(),
				}

				if err := wp.Store.InsertRelation(ctx, relation); err != nil {
					log.Printf("[Worker %d] Error inserting relation [%s -> %s (%s)]: %v",
						workerID, sourceID, targetID, relation.Type, err)
					return
				}

				log.Printf("[Worker %d] Extracted and stored relationship: [%s -> %s (%s)]",
					workerID, sourceName, targetName, relation.Type)
			}(r)
		}
		wgRels.Wait()
	} else {
		log.Printf("[Worker %d] No relationships found in message.", workerID)
	}

	// Invalidate semantic cache for the agent owner as new facts/relations have been consolidated
	if wp.Cache != nil {
		if err := wp.Cache.Invalidate(ctx, job.EntityID); err != nil {
			log.Printf("[Worker %d] Error invalidating semantic cache for owner %s: %v", workerID, job.EntityID, err)
		} else {
			log.Printf("[Worker %d] Semantic cache invalidated for owner %s", workerID, job.EntityID)
		}
	}
}

// isSingularAttribute returns true if the attribute represents a mutually exclusive state
// (e.g. user name, current company, preferred programming language) that should be
// deactivated and overwritten when a new value is specified.
// Cumulative or historical attributes (e.g. past injuries, former companies, hospitalizations)
// are non-exclusive, allowing multiple facts to coexist and form a list/history of events.
func isSingularAttribute(attr string) bool {
	attrLower := strings.ToLower(attr)

	// Historical, list-like, or plural patterns are cumulative
	if strings.HasPrefix(attrLower, "former_") ||
		strings.HasPrefix(attrLower, "past_") ||
		strings.HasPrefix(attrLower, "visited_") ||
		strings.HasSuffix(attrLower, "_history") ||
		strings.HasSuffix(attrLower, "_list") ||
		strings.HasSuffix(attrLower, "_hobbies") ||
		strings.HasSuffix(attrLower, "_interests") {
		return false
	}

	// Specific cumulative substrings
	if strings.Contains(attrLower, "hobb") ||        // Matches "hobby", "hobbies", "user_preference_hobby"
		strings.Contains(attrLower, "interest") ||    // Matches "interest", "interests", "special_interests"
		strings.Contains(attrLower, "injury") ||      // Matches "injury", "injuries", "past_injury"
		strings.Contains(attrLower, "hospital") ||    // Matches "hospitalization"
		strings.Contains(attrLower, "allergy") ||     // Matches "allergy", "allergies"
		strings.Contains(attrLower, "medication") {   // Matches "medication", "medications"
		return false
	}

	// Default to mutually exclusive singular state (e.g. user_name, preferred_programming_language)
	return true
}
