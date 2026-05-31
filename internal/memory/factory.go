package memory

import (
	"fmt"
)

// NewMemoryStore instantiates a concrete database store matching the configured provider.
func NewMemoryStore(cfg Config) (MemoryStore, error) {
	switch cfg.Provider {
	case "postgres":
		return NewPGStore(cfg.PostgresURL)
	case "neo4j":
		return NewNeo4jStore(cfg.Neo4jURI, cfg.Neo4jUsername, cfg.Neo4jPassword)
	case "falkordb":
		return NewFalkorDBStore(cfg.FalkorDBURL, cfg.FalkorDBGraphName)
	default:
		return nil, fmt.Errorf("unsupported database provider: %s", cfg.Provider)
	}
}
