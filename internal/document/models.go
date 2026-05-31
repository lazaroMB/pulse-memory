package document

import (
	"time"

	"github.com/google/uuid"
)

type SourceType string

const (
	SourcePDF        SourceType = "pdf"
	SourceMarkdown   SourceType = "markdown"
	SourceGoogleDocs SourceType = "google_docs"
	SourceWebPage    SourceType = "web_page"
)

type IngestionStatus string

const (
	StatusPending    IngestionStatus = "pending"
	StatusProcessing IngestionStatus = "processing"
	StatusCompleted  IngestionStatus = "completed"
	StatusFailed     IngestionStatus = "failed"
)

type Document struct {
	ID           uuid.UUID         `db:"id" json:"id"`
	Title        string            `db:"title" json:"title"`
	SourceType   SourceType        `db:"source_type" json:"source_type"`
	SourceURL    string            `db:"source_url" json:"source_url,omitempty"`
	FilePath     string            `db:"file_path" json:"file_path,omitempty"`
	Status       IngestionStatus   `db:"status" json:"status"`
	ErrorMessage string            `db:"error_message" json:"error_message,omitempty"`
	Metadata     map[string]string `db:"metadata" json:"metadata"`
	CreatedAt    time.Time         `db:"created_at" json:"created_at"`
	UpdatedAt    time.Time         `db:"updated_at" json:"updated_at"`
}

type DocumentChunk struct {
	ID         uuid.UUID         `db:"id" json:"id"`
	DocumentID uuid.UUID         `db:"document_id" json:"document_id"`
	ChunkIndex int               `db:"chunk_index" json:"chunk_index"`
	Content    string            `db:"content" json:"content"`
	Metadata   map[string]string `db:"metadata" json:"metadata"`
}
