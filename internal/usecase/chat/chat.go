package chat

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/google/uuid"
	"pulse/internal/domain/entity"
	"pulse/internal/usecase/ports"
)

type ChatUseCase struct {
	store      ports.MemoryRepository
	chatMemory ports.ChatMemoryRepository
	llm        ports.LLMService
	cache      ports.SemanticCache
	privacy    ports.PrivacyService
	worker     ports.ConsolidationService
}

func NewChatUseCase(
	store ports.MemoryRepository,
	chatMemory ports.ChatMemoryRepository,
	llm ports.LLMService,
	cache ports.SemanticCache,
	privacy ports.PrivacyService,
	worker ports.ConsolidationService,
) *ChatUseCase {
	return &ChatUseCase{
		store:      store,
		chatMemory: chatMemory,
		llm:        llm,
		cache:      cache,
		privacy:    privacy,
		worker:     worker,
	}
}

type ChatInput struct {
	SessionID    string
	EntityID     uuid.UUID
	AgentRole    string
	Message      string
	IncludeFacts bool
}

type ChatOutput struct {
	ResponseMessage string
	EntityFacts     []entity.Fact
	DocumentFacts   []entity.Fact
}

func (u *ChatUseCase) Execute(ctx context.Context, req ChatInput) (*ChatOutput, error) {
	ctx = entity.WithAgentOwner(ctx, req.EntityID)

	// 1. Apply Local Privacy Filter (PII Scrubbing Proxy)
	cleanMessage, err := u.privacy.ScrubText(ctx, req.Message)
	if err != nil {
		return nil, fmt.Errorf("privacy scrubbing failed: %w", err)
	}

	// 2. Generate Vector Embedding of the Query Text for similarity search
	queryVector, err := u.llm.GenerateEmbedding(ctx, cleanMessage)
	if err != nil {
		log.Printf("Embedding generation failed, falling back to basic metadata retrieval: %v", err)
		queryVector = nil
	}

	// 3. Check Semantic Cache for hits (Cos similarity >= 0.95)
	if len(queryVector) > 0 {
		cachedReply, found, err := u.cache.Get(ctx, queryVector)
		if err == nil && found {
			log.Printf("[Semantic Cache HIT] Returning cached response directly for query: %s", cleanMessage)
			// Append cache turn to short-term chat memory to keep session state updated
			userMsg := entity.ChatMessage{
				Role:      req.AgentRole,
				Content:   cleanMessage,
				Timestamp: time.Now(),
			}
			_ = u.chatMemory.AppendMessage(ctx, req.SessionID, userMsg)

			assistantMsg := entity.ChatMessage{
				Role:      "assistant",
				Content:   cachedReply,
				Timestamp: time.Now(),
			}
			_ = u.chatMemory.AppendMessage(ctx, req.SessionID, assistantMsg)

			return &ChatOutput{
				ResponseMessage: cachedReply,
			}, nil
		}
	}

	// 4. Search database for active facts (Hybrid Search)
	searchQuery := &entity.MemorySearchQuery{
		QueryText:     cleanMessage,
		QueryVector:   queryVector,
		TargetEntity:  req.EntityID,
		RequiredScope: req.AgentRole,
		AgentOwner:    req.EntityID,
		MaxResults:    5,
	}

	retrievedFacts, err := u.store.SearchHybrid(ctx, searchQuery)
	if err != nil {
		log.Printf("Database retrieval error: %v", err)
		retrievedFacts = []entity.Fact{}
	}

	// 4b. Search database for active general/shared knowledge facts extracted from documents
	sharedEntityID := uuid.NewMD5(uuid.NameSpaceDNS, []byte("shared-general-knowledge"))
	sharedSearchQuery := &entity.MemorySearchQuery{
		QueryText:     cleanMessage,
		QueryVector:   queryVector,
		TargetEntity:  sharedEntityID,
		RequiredScope: req.AgentRole,
		AgentOwner:    req.EntityID,
		MaxResults:    5,
	}

	sharedFacts, err := u.store.SearchHybrid(ctx, sharedSearchQuery)
	if err != nil {
		log.Printf("Shared database retrieval error: %v", err)
	} else {
		retrievedFacts = append(retrievedFacts, sharedFacts...)
	}

	// 5. Validate Access (Role-Based Access Control)
	var filteredFacts []entity.Fact
	for _, fact := range retrievedFacts {
		if u.privacy.ValidateAccess(ctx, req.AgentRole, &fact) {
			filteredFacts = append(filteredFacts, fact)
		}
	}

	// 6. Perform vector similarity search over raw document chunks (Standard RAG)
	if len(queryVector) > 0 {
		chunks, err := u.store.SearchDocumentChunks(ctx, queryVector, 3)
		if err == nil {
			for _, chunk := range chunks {
				docTitle := "Document"
				if chunk.Metadata != nil && chunk.Metadata["title"] != "" {
					docTitle = chunk.Metadata["title"]
				} else if chunk.Metadata != nil && chunk.Metadata["source_url"] != "" {
					docTitle = chunk.Metadata["source_url"]
				}
				filteredFacts = append(filteredFacts, entity.Fact{
					ID:              chunk.ID,
					EntityID:        sharedEntityID,
					Attribute:       "document_chunk",
					Value:           fmt.Sprintf("[Source: %s] %s", docTitle, chunk.Content),
					ConfidenceScore: 1.0,
					SourceAgent:     "document",
				})
			}
		} else {
			log.Printf("Document chunks retrieval error: %v", err)
		}

		// 6b. Retrieve community summaries for Macro-Narrative GraphRAG context
		communities, err := u.store.SearchCommunitySummaries(ctx, queryVector, 2)
		if err == nil {
			for _, comm := range communities {
				filteredFacts = append(filteredFacts, entity.Fact{
					ID:              comm.ID,
					EntityID:        sharedEntityID,
					Attribute:       "macro_narrative",
					Value:           fmt.Sprintf("[Macro Narrative Summary: %s] %s", comm.Name, comm.Summary),
					ConfidenceScore: 1.0,
					SourceAgent:     "community_graphrag",
				})
			}
		} else {
			log.Printf("Community summaries retrieval error: %v", err)
		}
	}

	// 7. Retrieve 1-hop and 2-hop active relationships for Graph Context (Graph RAG)
	rels, err := u.store.GetActiveRelations(ctx, req.EntityID)
	if err == nil && len(rels) > 0 {
		entityIDs := make(map[uuid.UUID]bool)
		entityIDs[req.EntityID] = true
		for _, r := range rels {
			entityIDs[r.SourceID] = true
			entityIDs[r.TargetID] = true
		}

		// Optimized path: batch fetch 2-hop relations in 1 query
		otherIDsMap := make(map[uuid.UUID]bool)
		for _, r := range rels {
			otherID := r.TargetID
			if otherID != req.EntityID {
				otherIDsMap[otherID] = true
			}
		}
		var otherIDs []uuid.UUID
		for oid := range otherIDsMap {
			otherIDs = append(otherIDs, oid)
		}
		if len(otherIDs) > 0 {
			otherRels, err := u.store.GetActiveRelationsBatch(ctx, otherIDs)
			if err == nil {
				for _, or := range otherRels {
					entityIDs[or.SourceID] = true
					entityIDs[or.TargetID] = true
					rels = append(rels, or)
				}
			} else {
				log.Printf("Failed to batch get 2-hop active relations: %v", err)
			}
		}

		// Batch resolve names in 1 query
		var eIDs []uuid.UUID
		for eID := range entityIDs {
			if eID != req.EntityID {
				eIDs = append(eIDs, eID)
			}
		}
		nameMap, err := u.store.GetEntityNamesBatch(ctx, eIDs)
		if err != nil {
			log.Printf("Failed to batch get entity names: %v", err)
			nameMap = make(map[uuid.UUID]string)
		}

		if nameMap == nil {
			nameMap = make(map[uuid.UUID]string)
		}
		nameMap[req.EntityID] = "user" // Speaker is mapped to "user"

		// Deduplicate and format relations as artificial Graph Relation facts
		seenRels := make(map[string]bool)
		for _, r := range rels {
			if strings.ToUpper(r.Type) == "HAS_NAME" || strings.ToUpper(r.Type) == "NAME" {
				continue
			}
			srcName := nameMap[r.SourceID]
			tgtName := nameMap[r.TargetID]
			if srcName == "" {
				srcName = r.SourceID.String()[:8]
			}
			if tgtName == "" {
				tgtName = r.TargetID.String()[:8]
			}
			if srcName == tgtName {
				continue
			}
			relKey := fmt.Sprintf("%s-%s-%s", srcName, r.Type, tgtName)
			if !seenRels[relKey] {
				seenRels[relKey] = true
				filteredFacts = append(filteredFacts, entity.Fact{
					Attribute:       "graph_relation",
					Value:           fmt.Sprintf("Graph Relation: %s is linked to %s via %s", srcName, tgtName, r.Type),
					ConfidenceScore: 1.0,
				})
			}
		}
	}

	// 8. Retrieve short-term chat history
	history, err := u.chatMemory.GetSessionHistory(ctx, req.SessionID, 15) // Fetch last 15 messages for context
	if err != nil {
		log.Printf("Failed to retrieve chat history for session %s: %v", req.SessionID, err)
		history = []entity.ChatMessage{}
	}

	// 9. Generate Answer via LLM using the retrieved facts and short-term history
	reply, err := u.llm.GenerateAnswer(ctx, cleanMessage, history, filteredFacts)
	if err != nil {
		return nil, fmt.Errorf("failed to generate answer from model: %w", err)
	}

	// 10. Register response in semantic cache asynchronously
	if len(queryVector) > 0 {
		go func(qText string, qVec []float32, rText string) {
			_ = u.cache.Set(context.Background(), qText, qVec, rText)
		}(cleanMessage, queryVector, reply)
	}

	// 11. Append interaction turn to short-term chat memory
	userMsg := entity.ChatMessage{
		Role:      req.AgentRole,
		Content:   cleanMessage,
		Timestamp: time.Now(),
	}
	if err := u.chatMemory.AppendMessage(ctx, req.SessionID, userMsg); err != nil {
		log.Printf("Failed to append user message to chat memory: %v", err)
	}

	assistantMsg := entity.ChatMessage{
		Role:      "assistant",
		Content:   reply,
		Timestamp: time.Now(),
	}
	if err := u.chatMemory.AppendMessage(ctx, req.SessionID, assistantMsg); err != nil {
		log.Printf("Failed to append assistant message to chat memory: %v", err)
	}

	// 12. Push event to the Asynchronous Consolidation queue (The Sleep-Time Pattern)
	u.worker.QueueInteraction(entity.InteractionLog{
		SessionID: req.SessionID,
		EntityID:  req.EntityID,
		Sender:    req.AgentRole,
		Message:   cleanMessage,
		Timestamp: time.Now(),
	})

	var entityFacts []entity.Fact
	var documentFacts []entity.Fact

	if req.IncludeFacts {
		entityFacts = []entity.Fact{}
		documentFacts = []entity.Fact{}
		for _, fact := range filteredFacts {
			if fact.SourceAgent == "document" || fact.Attribute == "document_chunk" {
				documentFacts = append(documentFacts, fact)
			} else {
				entityFacts = append(entityFacts, fact)
			}
		}
	}

	return &ChatOutput{
		ResponseMessage: reply,
		EntityFacts:     entityFacts,
		DocumentFacts:   documentFacts,
	}, nil
}
