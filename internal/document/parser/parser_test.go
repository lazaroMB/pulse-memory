package parser

import (
	"context"
	"strings"
	"testing"
)

func TestMarkdownParser(t *testing.T) {
	p := &MarkdownParser{}
	md := `---
title: Test Title
author: Jane Doe
tags: go, test, parser
---
# Main Header
This is the body content of the test document.`

	res, err := p.Parse(context.Background(), strings.NewReader(md), nil)
	if err != nil {
		t.Fatalf("failed to parse markdown: %v", err)
	}

	if res.Metadata["title"] != "Test Title" {
		t.Errorf("expected title 'Test Title', got '%s'", res.Metadata["title"])
	}
	if res.Metadata["author"] != "Jane Doe" {
		t.Errorf("expected author 'Jane Doe', got '%s'", res.Metadata["author"])
	}
	if !strings.Contains(res.Text, "This is the body content") {
		t.Errorf("body text missing target sentence")
	}
	if strings.Contains(res.Text, "title: Test Title") {
		t.Errorf("body text contains yaml frontmatter block")
	}
}

func TestWebPageParser(t *testing.T) {
	p := &WebPageParser{}
	html := `<html>
<head><title>Web Title</title></head>
<body>
<header><nav><a href="/">Home</a></nav></header>
<main>
<h1>Main Article Heading</h1>
<p>First paragraph of the webpage article.</p>
</main>
<footer>Standard Footer Copyright</footer>
</body>
</html>`

	res, err := p.Parse(context.Background(), strings.NewReader(html), nil)
	if err != nil {
		t.Fatalf("failed to parse html page: %v", err)
	}

	if res.Metadata["title"] != "Web Title" {
		t.Errorf("expected title 'Web Title', got '%s'", res.Metadata["title"])
	}
	if strings.Contains(res.Text, "Standard Footer") {
		t.Errorf("boilerplate footer text was not stripped from text content")
	}
	if !strings.Contains(res.Text, "Main Article Heading") {
		t.Errorf("article content missing header")
	}
}
