package parser

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"strings"

	"github.com/dslipak/pdf"
)

type PDFParser struct{}

func (p *PDFParser) Parse(ctx context.Context, src io.Reader, options map[string]string) (*ParsedContent, error) {
	// First check if pdftotext is available in the system path
	if pdftotextPath, err := exec.LookPath("pdftotext"); err == nil {
		cmd := exec.CommandContext(ctx, pdftotextPath, "-", "-")
		cmd.Stdin = src

		var stdoutBuf, stderrBuf bytes.Buffer
		cmd.Stdout = &stdoutBuf
		cmd.Stderr = &stderrBuf

		if err := cmd.Run(); err == nil {
			text := stdoutBuf.String()
			pageCount := strings.Count(text, "\x0c")
			if pageCount == 0 && len(text) > 0 {
				pageCount = 1
			}

			metadata := make(map[string]string)
			metadata["page_count"] = fmt.Sprintf("%d", pageCount)
			metadata["parser"] = "pdftotext"

			return &ParsedContent{
				Text:     text,
				Metadata: metadata,
			}, nil
		}
		// If pdftotext fails, fall through and use native reader
	}

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
	metadata["parser"] = "native"

	return &ParsedContent{
		Text:     sb.String(),
		Metadata: metadata,
	}, nil
}
