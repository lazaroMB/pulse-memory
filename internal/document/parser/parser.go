package parser

import (
	"context"
	"io"
)

type ParsedContent struct {
	Text     string            `json:"text"`
	Metadata map[string]string `json:"metadata"`
}

type DocumentParser interface {
	Parse(ctx context.Context, src io.Reader, options map[string]string) (*ParsedContent, error)
}
