package parser

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/dslipak/pdf"
)

type PDFParser struct{}

func (p *PDFParser) Parse(ctx context.Context, src io.Reader, options map[string]string) (*ParsedContent, error) {
	buf, err := io.ReadAll(src)
	if err != nil {
		return nil, fmt.Errorf("failed to read pdf stream: %w", err)
	}

	readerAt := bytes.NewReader(buf)
	size := int64(len(buf))

	r, err := pdf.NewReader(readerAt, size)
	if err != nil {
		return nil, fmt.Errorf("failed to create pdf reader: %w", err)
	}

	var sb strings.Builder
	numPages := r.NumPage()
	for i := 1; i <= numPages; i++ {
		p := r.Page(i)
		if p.V.IsNull() {
			continue
		}

		text, err := p.GetPlainText(nil)
		if err == nil {
			sb.WriteString(text)
			sb.WriteByte('\n')
		}
	}

	metadata := make(map[string]string)
	metadata["page_count"] = fmt.Sprintf("%d", numPages)

	return &ParsedContent{
		Text:     sb.String(),
		Metadata: metadata,
	}, nil
}
