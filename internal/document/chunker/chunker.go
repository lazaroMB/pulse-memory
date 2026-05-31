package chunker

import (
	"math"
	"strings"

	"github.com/google/uuid"
	"pulse/internal/document"
)

type Chunker struct {
	ChunkSize    int
	ChunkOverlap int
}

func NewChunker(size, overlap int) *Chunker {
	if size <= 0 {
		size = 1000
	}
	if overlap < 0 || overlap >= size {
		overlap = 200
	}
	return &Chunker{
		ChunkSize:    size,
		ChunkOverlap: overlap,
	}
}

func (c *Chunker) SplitToChunks(docID uuid.UUID, text string, baseMetadata map[string]string) []document.DocumentChunk {
	if len(text) == 0 {
		return nil
	}

	rawChunks := c.recursiveSplit(text, []string{"\n\n", "\n", ". ", "? ", "! ", " ", ""})
	var chunks []document.DocumentChunk
	chunkIndex := 0

	var currentChunk strings.Builder
	for _, raw := range rawChunks {
		trimmed := strings.TrimSpace(raw)
		if trimmed == "" {
			continue
		}

		if currentChunk.Len()+len(trimmed) > c.ChunkSize {
			if currentChunk.Len() > 0 {
				meta := make(map[string]string)
				for k, v := range baseMetadata {
					meta[k] = v
				}

				chunks = append(chunks, document.DocumentChunk{
					ID:         uuid.New(),
					DocumentID: docID,
					ChunkIndex: chunkIndex,
					Content:    strings.TrimSpace(currentChunk.String()),
					Metadata:   meta,
				})
				chunkIndex++

				// Apply overlap logic safely on a rune slice to avoid cutting multi-byte UTF-8 characters
				runes := []rune(currentChunk.String())
				overlapStart := int(math.Max(0, float64(len(runes)-c.ChunkOverlap)))
				overlapText := string(runes[overlapStart:])
				currentChunk.Reset()
				currentChunk.WriteString(overlapText)
			}
		}

		if currentChunk.Len() > 0 && !strings.HasSuffix(currentChunk.String(), " ") {
			currentChunk.WriteByte(' ')
		}
		currentChunk.WriteString(trimmed)
	}

	if currentChunk.Len() > 0 {
		meta := make(map[string]string)
		for k, v := range baseMetadata {
			meta[k] = v
		}

		chunks = append(chunks, document.DocumentChunk{
			ID:         uuid.New(),
			DocumentID: docID,
			ChunkIndex: chunkIndex,
			Content:    strings.TrimSpace(currentChunk.String()),
			Metadata:   meta,
		})
	}

	return chunks
}

func (c *Chunker) recursiveSplit(text string, separators []string) []string {
	if len(text) <= c.ChunkSize || len(separators) == 0 {
		return []string{text}
	}

	sep := separators[0]
	nextSeps := separators[1:]

	parts := strings.Split(text, sep)
	var result []string

	for _, p := range parts {
		if len(p) <= c.ChunkSize {
			result = append(result, p+sep)
		} else {
			subParts := c.recursiveSplit(p, nextSeps)
			result = append(result, subParts...)
		}
	}

	return result
}
