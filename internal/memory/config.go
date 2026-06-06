package memory

import "time"

// Config contains the environment configuration variables for the ArcadeDB provider.
type Config struct {
	URL      string        // e.g. "http://localhost:2480"
	Database string        // e.g. "pulse"
	Username string
	Password string
	Timeout  time.Duration // General request timeout
}
