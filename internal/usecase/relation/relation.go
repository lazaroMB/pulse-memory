package relation

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"pulse/internal/domain/entity"
	"pulse/internal/usecase/ports"
)

type RelationUseCase struct {
	store ports.MemoryRepository
	cache ports.SemanticCache
}

func NewRelationUseCase(store ports.MemoryRepository, cache ports.SemanticCache) *RelationUseCase {
	return &RelationUseCase{
		store: store,
		cache: cache,
	}
}

type RelationInput struct {
	SourceID   uuid.UUID
	TargetID   uuid.UUID
	Type       string
	AgentOwner uuid.UUID
}

func (u *RelationUseCase) Execute(ctx context.Context, req RelationInput) error {
	agentOwner := req.SourceID
	if req.AgentOwner != uuid.Nil {
		agentOwner = req.AgentOwner
	}

	ctx = entity.WithAgentOwner(ctx, agentOwner)
	relation := &entity.Relation{
		ID:         uuid.New(),
		SourceID:   req.SourceID,
		TargetID:   req.TargetID,
		Type:       req.Type,
		AgentOwner: agentOwner,
		ValidFrom:  time.Now(),
	}

	if err := u.store.InsertRelation(ctx, relation); err != nil {
		return fmt.Errorf("failed to store relation: %w", err)
	}

	if u.cache != nil {
		_ = u.cache.Invalidate(ctx, agentOwner)
	}

	return nil
}
