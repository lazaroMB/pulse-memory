package consolidation

import (
	"testing"
)

func TestIsSingularAttribute(t *testing.T) {
	tests := []struct {
		attr     string
		expected bool
	}{
		// Singular / Mutually Exclusive Attributes
		{"user_name", true},
		{"company", true},
		{"company_city", true},
		{"company_country", true},
		{"email", true},
		{"preferred_programming_language", true},
		{"current_city", true},

		// Cumulative / List-like Attributes (with former_/past_/visited_ prefixes)
		{"former_company", false},
		{"former_company_city", false},
		{"former_company_country", false},
		{"past_injury", false},
		{"past_hospitalization", false},
		{"visited_city", false},
		{"visited_country", false},

		// Cumulative / List-like Attributes (with suffix patterns)
		{"travel_history", false},
		{"programming_list", false},
		{"reading_hobbies", false},
		{"scientific_interests", false},

		// Specific cumulative keywords and composite attributes
		{"hospitalization", false},
		{"injury_history", false},
		{"hobby", false},
		{"interest", false},
		{"user_preference_hobby", false},
		{"user_preference_interest", false},
		{"allergy", false},
		{"medication", false},
	}

	for _, tt := range tests {
		t.Run(tt.attr, func(t *testing.T) {
			result := isSingularAttribute(tt.attr)
			if result != tt.expected {
				t.Errorf("expected isSingularAttribute(%q) to be %v, got %v", tt.attr, tt.expected, result)
			}
		})
	}
}
