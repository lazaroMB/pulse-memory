package memory

import "time"

// Config contains the environment configuration variables for all supported database providers.
type Config struct {
	Provider    string        // "postgres", "neo4j", "falkordb"
	Timeout     time.Duration // General request timeout

	// PostgreSQL
	PostgresURL string

	// Neo4j
	Neo4jURI      string
	Neo4jUsername string
	Neo4jPassword string

	// FalkorDB
	FalkorDBURL       string
	FalkorDBGraphName string
}
