package parser

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

type GoogleDocsParser struct{}

// ExtractDocID parses a Google Docs URL to find the document identifier.
func ExtractDocID(url string) (string, error) {
	// Pattern to match /document/d/{doc_id}
	re := regexp.MustCompile(`/document/d/([a-zA-Z0-9-_]+)`)
	matches := re.FindStringSubmatch(url)
	if len(matches) < 2 {
		return "", fmt.Errorf("could not parse document ID from URL: %s", url)
	}
	return matches[1], nil
}

func (g *GoogleDocsParser) Parse(ctx context.Context, src io.Reader, options map[string]string) (*ParsedContent, error) {
	// If src is provided, parse it as raw text directly (fallback)
	buf, err := io.ReadAll(src)
	if err == nil && len(buf) > 0 {
		return &ParsedContent{
			Text:     strings.TrimSpace(string(buf)),
			Metadata: make(map[string]string),
		}, nil
	}

	// Fetch doc directly if URL option is provided
	docURL, ok := options["url"]
	if !ok || docURL == "" {
		return nil, fmt.Errorf("missing required 'url' option for google docs parser")
	}

	docID, err := ExtractDocID(docURL)
	if err != nil {
		return nil, fmt.Errorf("failed to extract Google Doc ID: %w", err)
	}

	// Construct export URL
	exportURL := fmt.Sprintf("https://docs.google.com/feeds/download/documents/export/Export?id=%s&exportFormat=txt", docID)

	req, err := http.NewRequestWithContext(ctx, "GET", exportURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create http request for google doc export: %w", err)
	}

	// Add Authorization token if passed in options
	if token, ok := options["auth_token"]; ok && token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{
		Timeout: 15 * time.Second,
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to download google doc: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("google doc download failed with status code: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read google doc download body: %w", err)
	}

	metadata := make(map[string]string)
	metadata["document_id"] = docID
	metadata["source_url"] = docURL

	return &ParsedContent{
		Text:     strings.TrimSpace(string(body)),
		Metadata: metadata,
	}, nil
}
