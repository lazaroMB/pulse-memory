package entity

// SwarmAgent represents a member of the multi-agent cognitive swarm.
type SwarmAgent struct {
	ID         string  // e.g. "agent:chemistry", "agent:logistics", "user"
	Role       string  // e.g. "admin", "writer", "guest"
	TrustScore float64 // Epistemic trust score between 0.0 and 1.0
}
