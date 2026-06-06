package document

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"pulse/internal/domain/entity"
	"pulse/internal/usecase/ports"
)

type DocumentUseCase struct {
	store  ports.MemoryRepository
	worker ports.ConsolidationService
}

func NewDocumentUseCase(store ports.MemoryRepository, worker ports.ConsolidationService) *DocumentUseCase {
	return &DocumentUseCase{
		store:  store,
		worker: worker,
	}
}

type IngestFileInput struct {
	Title      string
	SourceType entity.SourceType
	FilePath   string
	EntityID   string
}

type IngestLinkInput struct {
	URL        string
	Title      string
	SourceType entity.SourceType
	EntityID   string
}

func (u *DocumentUseCase) IngestFile(ctx context.Context, req IngestFileInput) (uuid.UUID, error) {
	docID := uuid.New()
	doc := &entity.Document{
		ID:         docID,
		Title:      req.Title,
		SourceType: req.SourceType,
		FilePath:   req.FilePath,
		Status:     entity.StatusPending,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Metadata:   make(map[string]string),
	}

	if err := u.store.InsertDocument(ctx, doc); err != nil {
		return uuid.Nil, fmt.Errorf("failed to insert document record: %w", err)
	}

	if req.EntityID != "" {
		if entityUUID, err := uuid.Parse(req.EntityID); err == nil {
			_ = u.store.LinkDocumentToAuthor(ctx, docID, entityUUID)
			doc.Metadata["target_entity"] = req.EntityID
		}
	}

	u.worker.QueueDocument(entity.DocumentJob{
		DocumentID: docID,
		FilePath:   req.FilePath,
		SourceType: req.SourceType,
		Metadata:   doc.Metadata,
	})

	return docID, nil
}

func (u *DocumentUseCase) IngestLink(ctx context.Context, req IngestLinkInput) (uuid.UUID, error) {
	docID := uuid.New()
	doc := &entity.Document{
		ID:         docID,
		Title:      req.Title,
		SourceType: req.SourceType,
		SourceURL:  req.URL,
		Status:     entity.StatusPending,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Metadata:   make(map[string]string),
	}

	if err := u.store.InsertDocument(ctx, doc); err != nil {
		return uuid.Nil, fmt.Errorf("failed to insert document record: %w", err)
	}

	if req.EntityID != "" {
		if entityUUID, err := uuid.Parse(req.EntityID); err == nil {
			_ = u.store.LinkDocumentToAuthor(ctx, docID, entityUUID)
			doc.Metadata["target_entity"] = req.EntityID
		}
	}

	u.worker.QueueDocument(entity.DocumentJob{
		DocumentID: docID,
		URL:        req.URL,
		SourceType: req.SourceType,
		Metadata:   doc.Metadata,
	})

	return docID, nil
}

func (u *DocumentUseCase) GetDocument(ctx context.Context, id uuid.UUID) (*entity.Document, error) {
	return u.store.GetDocument(ctx, id)
}

func (u *DocumentUseCase) SearchDocuments(ctx context.Context, queryVector []float32, limit int) ([]entity.DocumentChunk, error) {
	return u.store.SearchDocumentChunks(ctx, queryVector, limit)
}
