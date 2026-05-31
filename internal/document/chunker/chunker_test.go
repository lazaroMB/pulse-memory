package chunker

import (
	"testing"

	"github.com/google/uuid"
)

func TestChunker(t *testing.T) {
	c := NewChunker(100, 20)
	docID := uuid.New()
	baseMeta := map[string]string{"source": "test"}

	text := "This is a sentence that will serve as test text. It contains multiple sentences. We want to test how the recursive splitter segments it into smaller bounds."
	chunks := c.SplitToChunks(docID, text, baseMeta)

	if len(chunks) == 0 {
		t.Fatalf("expected chunks, got 0")
	}

	for i, chunk := range chunks {
		if chunk.DocumentID != docID {
			t.Errorf("chunk %d: expected document ID %v, got %v", i, docID, chunk.DocumentID)
		}
		if chunk.ChunkIndex != i {
			t.Errorf("chunk %d: expected index %d, got %d", i, i, chunk.ChunkIndex)
		}
		if len(chunk.Content) > c.ChunkSize {
			t.Errorf("chunk %d: content length %d exceeds max chunk size %d", i, len(chunk.Content), c.ChunkSize)
		}
		if chunk.Metadata["source"] != "test" {
			t.Errorf("chunk %d: missing metadata 'source'", i)
		}
	}
}
