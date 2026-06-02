package agent

import (
	"context"
	"log"

	"pulse/internal/memory"
)

type SwarmAgent struct {
	ID         string  // e.g. "agent:chemistry", "agent:logistics", "user"
	Role       string  // e.g. "admin", "writer", "guest"
	TrustScore float64 // Epistemic trust score between 0.0 and 1.0
}

// SwarmConsensusManager handles multi-agent epistemic validation, trust scores, and write authorization.
type SwarmConsensusManager struct {
	Agents map[string]SwarmAgent
	Store  memory.MemoryStore
}

// NewSwarmConsensusManager instantiates a concrete SwarmConsensusManager
func NewSwarmConsensusManager(store memory.MemoryStore) *SwarmConsensusManager {
	mgr := &SwarmConsensusManager{
		Agents: make(map[string]SwarmAgent),
		Store:  store,
	}
	// Register default agents in the swarm with their trust profiles
	mgr.RegisterAgent(SwarmAgent{ID: "agent:admin", Role: "admin", TrustScore: 1.0})
	mgr.RegisterAgent(SwarmAgent{ID: "agent:logistics", Role: "writer", TrustScore: 0.90})
	mgr.RegisterAgent(SwarmAgent{ID: "agent:chemistry", Role: "writer", TrustScore: 0.95})
	mgr.RegisterAgent(SwarmAgent{ID: "reflection_engine", Role: "writer", TrustScore: 0.80})
	mgr.RegisterAgent(SwarmAgent{ID: "document", Role: "writer", TrustScore: 0.85})
	return mgr
}

// RegisterAgent adds or updates an agent profile in the consensus swarm
func (scm *SwarmConsensusManager) RegisterAgent(agent SwarmAgent) {
	scm.Agents[agent.ID] = agent
}

// EvaluateFactConsensus adjusts the confidence score of a fact based on the source agent's trust score.
func (scm *SwarmConsensusManager) EvaluateFactConsensus(ctx context.Context, sourceAgent string, candidateAttribute string, candidateValue string) float64 {
	ag, ok := scm.Agents[sourceAgent]
	if !ok {
		return 0.5 // Default fallback trust score for unknown agents
	}
	
	log.Printf("[Swarm Consensus] Evaluated trust score for '%s': %.2f (Role: %s)", ag.ID, ag.TrustScore, ag.Role)
	return ag.TrustScore
}

// IsAuthorized checks if an agent has the required privileges to write to a specific fact attribute.
func (scm *SwarmConsensusManager) IsAuthorized(sourceAgent string, attribute string) bool {
	ag, ok := scm.Agents[sourceAgent]
	if !ok {
		return true // Allow by default for backwards compatibility
	}
	
	// Guests are prohibited from writing critical system parameters or proprietary formulas
	if ag.Role == "guest" && (attribute == "system_formula" || attribute == "proprietary_value") {
		return false
	}
	return true
}
