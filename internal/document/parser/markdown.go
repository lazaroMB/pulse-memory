package parser

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"

	"gopkg.in/yaml.v3"
)

type MarkdownParser struct{}

func (p *MarkdownParser) Parse(ctx context.Context, src io.Reader, options map[string]string) (*ParsedContent, error) {
	buf, err := io.ReadAll(src)
	if err != nil {
		return nil, fmt.Errorf("failed to read markdown stream: %w", err)
	}

	metadata := make(map[string]string)
	text := string(buf)

	// Check for frontmatter
	if strings.HasPrefix(text, "---\n") || strings.HasPrefix(text, "---\r\n") {
		scanner := bufio.NewScanner(bytes.NewReader(buf))
		
		// Skip first line "---"
		scanner.Scan()
		
		var yamlBuilder strings.Builder
		var bodyBuilder strings.Builder
		inFrontmatter := true

		for scanner.Scan() {
			line := scanner.Text()
			if inFrontmatter {
				if line == "---" {
					inFrontmatter = false
					continue
				}
				yamlBuilder.WriteString(line)
				yamlBuilder.WriteByte('\n')
			} else {
				bodyBuilder.WriteString(line)
				bodyBuilder.WriteByte('\n')
			}
		}

		if !inFrontmatter {
			// Successfully extracted frontmatter
			var yamlMap map[string]interface{}
			err := yaml.Unmarshal([]byte(yamlBuilder.String()), &yamlMap)
			if err == nil {
				for k, v := range yamlMap {
					metadata[k] = fmt.Sprintf("%v", v)
				}
			}
			text = bodyBuilder.String()
		}
	}

	return &ParsedContent{
		Text:     strings.TrimSpace(text),
		Metadata: metadata,
	}, nil
}
