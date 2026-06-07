package consolidation

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"pulse/internal/document/chunker"
	"pulse/internal/document/parser"
	"pulse/internal/domain/entity"
)

func (wp *WorkerPool) processDocumentJob(ctx context.Context, workerID int, job entity.DocumentJob) {
	log.Printf("[Document Worker %d] Starting processing for document %s (%s)", workerID, job.DocumentID, job.SourceType)

	// 1. Update status to Processing
	if err := wp.Store.UpdateDocumentStatus(ctx, job.DocumentID, entity.StatusProcessing, ""); err != nil {
		log.Printf("[Document Worker %d] Error updating document status to processing: %v", workerID, err)
	}

	// 2. Select parser and open source stream
	var p parser.DocumentParser
	var reader io.Reader
	var closeFn func()

	options := make(map[string]string)
	for k, v := range job.Metadata {
		options[k] = v
	}

	switch job.SourceType {
	case entity.SourcePDF:
		p = &parser.PDFParser{}
		if job.FilePath != "" {
			f, err := os.Open(job.FilePath)
			if err != nil {
				wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("failed to open local pdf: %w", err))
				return
			}
			reader = f
			closeFn = func() { f.Close() }
		} else if job.URL != "" {
			req, err := http.NewRequestWithContext(ctx, "GET", job.URL, nil)
			if err != nil {
				wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("failed to create pdf request: %w", err))
				return
			}
			req.Header.Set("User-Agent", "PulseSwarmMemoryIngester/1.0")

			client := &http.Client{Timeout: 30 * time.Second}
			resp, err := client.Do(req)
			if err != nil {
				wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("failed to fetch pdf page: %w", err))
				return
			}
			reader = resp.Body
			closeFn = func() { resp.Body.Close() }
		} else {
			wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("no source path or URL provided for PDF"))
			return
		}

	case entity.SourceMarkdown:
		p = &parser.MarkdownParser{}
		f, err := os.Open(job.FilePath)
		if err != nil {
			wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("failed to open local markdown: %w", err))
			return
		}
		reader = f
		closeFn = func() { f.Close() }

	case entity.SourceWebPage:
		p = &parser.WebPageParser{}
		req, err := http.NewRequestWithContext(ctx, "GET", job.URL, nil)
		if err != nil {
			wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("failed to create web request: %w", err))
			return
		}
		req.Header.Set("User-Agent", "PulseSwarmMemoryIngester/1.0")

		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("failed to fetch web page: %w", err))
			return
		}
		reader = resp.Body
		closeFn = func() { resp.Body.Close() }

	case entity.SourceGoogleDocs:
		p = &parser.GoogleDocsParser{}
		options["url"] = job.URL
		// GoogleDocs parser manages its own download stream when URL option is provided
		reader = stringsReaderFallback("") 
		closeFn = func() {}

	default:
		wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("unsupported source type: %s", job.SourceType))
		return
	}

	defer func() {
		if closeFn != nil {
			closeFn()
		}
	}()

	// 3. Parse content
	parsed, err := p.Parse(ctx, reader, options)
	if err != nil {
		wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("parsing failed: %w", err))
		return
	}

	// Sanitize text and metadata strings to be valid UTF-8
	parsed.Text = strings.ToValidUTF8(parsed.Text, "")
	for k, v := range parsed.Metadata {
		parsed.Metadata[k] = strings.ToValidUTF8(v, "")
	}

	// 4. Segment content into chunks
	c := chunker.NewChunker(1000, 200)
	chunks := c.SplitToChunks(job.DocumentID, parsed.Text, parsed.Metadata)
	if len(chunks) == 0 {
		wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("document generated 0 chunks"))
		return
	}

	// 5. Generate vector embeddings in batch
	contents := make([]string, len(chunks))
	for i, chk := range chunks {
		contents[i] = chk.Content
	}

	embeddings, err := wp.LLM.GenerateEmbeddings(ctx, contents)
	if err != nil {
		wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("embedding generation failed: %w", err))
		return
	}

	// 6. Write chunks to storage
	if err := wp.Store.InsertDocumentChunks(ctx, chunks, embeddings); err != nil {
		wp.failJob(ctx, workerID, job.DocumentID, fmt.Errorf("failed to store chunks: %w", err))
		return
	}

	// Link document to author or topic if metadata holds them
	if authorVal, ok := parsed.Metadata["author"]; ok && authorVal != "" {
		authorID := uuid.NewMD5(uuid.NameSpaceDNS, []byte("author-"+authorVal))
		_ = wp.Store.LinkDocumentToAuthor(ctx, job.DocumentID, authorID)
	}
	if tagsVal, ok := parsed.Metadata["tags"]; ok && tagsVal != "" {
		for _, tag := range splitTags(tagsVal) {
			_ = wp.Store.LinkDocumentToTopic(ctx, job.DocumentID, tag)
		}
	}

	// 7. Update status to Completed
	if err := wp.Store.UpdateDocumentStatus(ctx, job.DocumentID, entity.StatusCompleted, ""); err != nil {
		log.Printf("[Document Worker %d] Error updating document status to completed: %v", workerID, err)
	}

	log.Printf("[Document Worker %d] Successfully ingested document %s into standard RAG (chunks: %d)", workerID, job.DocumentID, len(chunks))

	// 8. Sleep-Time Cognitive Extraction: Extract Facts & Relations from Chunks asynchronously
	go wp.extractCognitiveGraph(ctx, workerID, job.DocumentID, chunks)
}

func (wp *WorkerPool) failJob(ctx context.Context, workerID int, docID uuid.UUID, err error) {
	log.Printf("[Document Worker %d] Job failed for document %s: %v", workerID, docID, err)
	_ = wp.Store.UpdateDocumentStatus(ctx, docID, entity.StatusFailed, err.Error())
}

func (wp *WorkerPool) extractCognitiveGraph(ctx context.Context, workerID int, docID uuid.UUID, chunks []entity.DocumentChunk) {
	bgCtx := context.Background() // Run in background independent of request lifecycle
	log.Printf("[Cognitive Extraction %d] Extracting facts & relations from %d chunks for document %s...", workerID, len(chunks), docID)

	for _, chunk := range chunks {
		// A. Extract atomic facts
		facts, err := wp.LLM.ExtractFacts(bgCtx, chunk.Content)
		if err != nil {
			log.Printf("[Cognitive Extraction %d] Error extracting facts from chunk %s: %v", workerID, chunk.ID, err)
			continue
		}

		for _, ext := range facts {
			// All document-extracted facts represent shared general knowledge accessible by all entities
			entityID := uuid.NewMD5(uuid.NameSpaceDNS, []byte("shared-general-knowledge"))
			
			// Deactivate older conflicting attributes if singular
			if isSingularAttribute(ext.Attribute) {
				activeFacts, err := wp.Store.SearchHybrid(bgCtx, &entity.MemorySearchQuery{
					TargetEntity: entityID,
					MaxResults:   100,
				})
				if err == nil {
					for _, existing := range activeFacts {
						if existing.Attribute == ext.Attribute && existing.Value != ext.Value {
							_ = wp.Store.DeactivateFact(bgCtx, existing.ID)
						}
					}
				}
			}

			// Save fact
			factID := uuid.New()
			representation := fmt.Sprintf("%s: %s", ext.Attribute, ext.Value)
			embedding, err := wp.LLM.GenerateEmbedding(bgCtx, representation)
			if err != nil {
				continue
			}

			newFact := &entity.Fact{
				ID:              factID,
				EntityID:        entityID,
				Attribute:       ext.Attribute,
				Value:           ext.Value,
				ConfidenceScore: ext.ConfidenceScore,
				ValidFrom:       time.Now(),
				SourceAgent:     "document:" + docID.String(),
			}

			if err := wp.Store.InsertFactWithProvenance(bgCtx, newFact, embedding, docID, chunk.ID); err != nil {
				log.Printf("[Cognitive Extraction %d] Error inserting fact with provenance: %v", workerID, err)
			}
		}

		// B. Extract relations
		rels, err := wp.LLM.ExtractRelations(bgCtx, chunk.Content)
		if err == nil {
			for _, ext := range rels {
				srcID := uuid.NewMD5(uuid.NameSpaceDNS, []byte(ext.SourceEntity))
				tgtID := uuid.NewMD5(uuid.NameSpaceDNS, []byte(ext.TargetEntity))
				relation := &entity.Relation{
					ID:        uuid.New(),
					SourceID:  srcID,
					TargetID:  tgtID,
					Type:      ext.RelationType,
					ValidFrom: time.Now(),
				}
				_ = wp.Store.InsertRelation(bgCtx, relation)
			}
		}
	}

	log.Printf("[Cognitive Extraction %d] Extraction completed for document %s", workerID, docID)
}

// Fallback helper for string reader
type stringReader struct {
	*io.SectionReader
}

func stringsReaderFallback(s string) io.Reader {
	return io.NewSectionReader(stringsReaderAt(s), 0, int64(len(s)))
}

type stringsReaderAt string

func (s stringsReaderAt) ReadAt(p []byte, off int64) (n int, err error) {
	if off >= int64(len(s)) {
		return 0, io.EOF
	}
	n = copy(p, s[off:])
	if n < len(p) {
		err = io.EOF
	}
	return
}

func splitTags(s string) []string {
	parts := []string{}
	for _, p := range timeFormatSplit(s, ",") {
		trimmed := timeFormatTrim(p)
		if trimmed != "" {
			parts = append(parts, trimmed)
		}
	}
	return parts
}

func timeFormatSplit(s, sep string) []string {
	var parts []string
	for _, p := range strings.Split(s, sep) {
		parts = append(parts, p)
	}
	return parts
}

func timeFormatTrim(s string) string {
	return strings.TrimSpace(s)
}
