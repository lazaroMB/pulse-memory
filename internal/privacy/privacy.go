package privacy

import (
	"context"
	"regexp"
	"strings"
	"pulse/internal/memory"
)

type LocalPrivacyFilter struct {
	emailRegex *regexp.Regexp
	phoneRegex *regexp.Regexp
	ipRegex    *regexp.Regexp
}

func NewLocalPrivacyFilter() *LocalPrivacyFilter {
	return &LocalPrivacyFilter{
		emailRegex: regexp.MustCompile(`(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}`),
		// Simple phone scrubber (captures common formats)
		phoneRegex: regexp.MustCompile(`\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}`),
		ipRegex:    regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`),
	}
}

// ScrubText removes PII patterns locally before sending text to the LLM or storage
func (f *LocalPrivacyFilter) ScrubText(ctx context.Context, text string) (string, error) {
	scrubbed := f.emailRegex.ReplaceAllString(text, "[EMAIL_REDACTED]")
	scrubbed = f.ipRegex.ReplaceAllString(scrubbed, "[IP_REDACTED]")
	
	// We only scrub phone numbers if they are 7+ digits to avoid false-positives on version numbers
	scrubbed = f.phoneRegex.ReplaceAllStringFunc(scrubbed, func(match string) string {
		digitsOnly := regexp.MustCompile(`\d`).FindAllString(match, -1)
		if len(digitsOnly) >= 7 {
			return "[PHONE_REDACTED]"
		}
		return match
	})

	return scrubbed, nil
}

// ValidateAccess implements Role-Based Access Control (RBAC) at the memory level
func (f *LocalPrivacyFilter) ValidateAccess(ctx context.Context, agentRole string, fact *memory.Fact) bool {
	// If a fact is written by an agent with higher privilege (e.g. "admin_agent"),
	// a lower privilege agent (e.g. "guest_agent") should not be able to retrieve it.
	agentRole = strings.ToLower(agentRole)
	sourceAgent := strings.ToLower(fact.SourceAgent)

	// Admin role has access to everything
	if agentRole == "admin" || agentRole == "admin_agent" {
		return true
	}

	// Facts created by admin are restricted from other roles
	if strings.Contains(sourceAgent, "admin") && !strings.Contains(agentRole, "admin") {
		return false
	}

	// Default: allow access
	return true
}
