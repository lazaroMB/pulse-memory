package memory

// NewMemoryStore instantiates a concrete ArcadeDB database store.
func NewMemoryStore(cfg Config) (MemoryStore, error) {
	return NewArcadeDBStore(cfg.URL, cfg.Database, cfg.Username, cfg.Password)
}
