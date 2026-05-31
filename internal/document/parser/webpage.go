package parser

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

type WebPageParser struct{}

func (w *WebPageParser) Parse(ctx context.Context, src io.Reader, options map[string]string) (*ParsedContent, error) {
	doc, err := goquery.NewDocumentFromReader(src)
	if err != nil {
		return nil, fmt.Errorf("failed to parse html document: %w", err)
	}

	// 1. Remove non-content elements
	doc.Find("script, style, noscript, iframe, nav, footer, header, aside, .footer, .header, #footer, #header, .sidebar, #sidebar").Each(func(i int, s *goquery.Selection) {
		s.Remove()
	})

	metadata := make(map[string]string)
	
	// Try to get title
	title := strings.TrimSpace(doc.Find("title").Text())
	if title != "" {
		metadata["title"] = title
	}

	// Meta description
	desc, exists := doc.Find("meta[name='description']").Attr("content")
	if exists {
		metadata["description"] = strings.TrimSpace(desc)
	}

	// Extract tags/keywords for graph topic matching
	var tags []string
	keywords, exists := doc.Find("meta[name='keywords']").Attr("content")
	if exists {
		for _, kw := range strings.Split(keywords, ",") {
			kwTrim := strings.TrimSpace(kw)
			if kwTrim != "" {
				tags = append(tags, kwTrim)
			}
		}
	}
	doc.Find("meta[property='article:tag']").Each(func(i int, s *goquery.Selection) {
		tag, exists := s.Attr("content")
		if exists {
			tagTrim := strings.TrimSpace(tag)
			if tagTrim != "" {
				tags = append(tags, tagTrim)
			}
		}
	})
	if len(tags) > 0 {
		metadata["tags"] = strings.Join(tags, ",")
	}

	// Extract author for graph authorship matching
	author, exists := doc.Find("meta[name='author']").Attr("content")
	if !exists {
		author, exists = doc.Find("meta[property='article:author']").Attr("content")
	}
	if exists && strings.TrimSpace(author) != "" {
		metadata["author"] = strings.TrimSpace(author)
	}

	// 2. Extract structured content block by block
	var sb strings.Builder
	
	// We scan the main content area (e.g. article, main, or fallback body)
	mainContent := doc.Find("article, main, [role='main']")
	if mainContent.Length() == 0 {
		mainContent = doc.Find("body")
	}

	// Traverse children of mainContent recursively and format headers/paragraphs
	formatSelection(mainContent, &sb)

	return &ParsedContent{
		Text:     strings.TrimSpace(sb.String()),
		Metadata: metadata,
	}, nil
}

func formatSelection(s *goquery.Selection, sb *strings.Builder) {
	s.Children().Each(func(i int, child *goquery.Selection) {
		tagName := goquery.NodeName(child)
		
		switch tagName {
		case "h1", "h2", "h3", "h4", "h5", "h6":
			level := int(tagName[1] - '0')
			sb.WriteString("\n\n")
			sb.WriteString(strings.Repeat("#", level))
			sb.WriteByte(' ')
			sb.WriteString(strings.TrimSpace(child.Text()))
			sb.WriteString("\n\n")
			
		case "p":
			text := strings.TrimSpace(child.Text())
			if text != "" {
				sb.WriteString(text)
				sb.WriteString("\n\n")
			}
			
		case "ul":
			child.ChildrenFiltered("li").Each(func(j int, li *goquery.Selection) {
				text := strings.TrimSpace(li.Text())
				if text != "" {
					sb.WriteString("* ")
					sb.WriteString(text)
					sb.WriteByte('\n')
				}
			})
			sb.WriteByte('\n')
			
		case "ol":
			child.ChildrenFiltered("li").Each(func(j int, li *goquery.Selection) {
				text := strings.TrimSpace(li.Text())
				if text != "" {
					sb.WriteString(fmt.Sprintf("%d. ", j+1))
					sb.WriteString(text)
					sb.WriteByte('\n')
				}
			})
			sb.WriteByte('\n')
			
		case "div", "section", "article":
			// If it contains only text, append it. Otherwise, recurse.
			if child.Children().Length() == 0 {
				text := strings.TrimSpace(child.Text())
				if text != "" {
					sb.WriteString(text)
					sb.WriteString("\n\n")
				}
			} else {
				formatSelection(child, sb)
			}
			
		default:
			// Recurse down for unknown inline wrappers
			if child.Children().Length() > 0 {
				formatSelection(child, sb)
			} else {
				text := strings.TrimSpace(child.Text())
				if text != "" {
					sb.WriteString(text)
					sb.WriteByte(' ')
				}
			}
		}
	})
}
