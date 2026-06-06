package redis

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/gomodule/redigo/redis"
	"github.com/google/uuid"
	"pulse/internal/domain/entity"
	"pulse/internal/usecase/ports"
)

// CacheEntry represents an element stored in the semantic cache.
type CacheEntry struct {
	QueryText    string    `json:"query_text"`
	QueryVector  []float32 `json:"query_vector"`
	ResponseText string    `json:"response_text"`
	CreatedAt    time.Time `json:"created_at"`
	AgentOwner   uuid.UUID `json:"agent_owner,omitempty"`
}

// InMemorySemanticCache implements SemanticCache purely in local RAM.
type InMemorySemanticCache struct {
	entries   map[uuid.UUID][]CacheEntry
	mu        sync.RWMutex
	threshold float64
}

// NewInMemorySemanticCache initializes a memory-only semantic cache.
func NewInMemorySemanticCache(threshold float64) *InMemorySemanticCache {
	if threshold <= 0 {
		threshold = 0.95
	}
	return &InMemorySemanticCache{
		entries:   make(map[uuid.UUID][]CacheEntry),
		threshold: threshold,
	}
}

func (c *InMemorySemanticCache) Get(ctx context.Context, queryVector []float32) (string, bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	agentOwner, _ := entity.GetAgentOwner(ctx)
	entries := c.entries[agentOwner]

	var bestMatch string
	var maxSimilarity float64 = -1.0

	for _, entry := range entries {
		select {
		case <-ctx.Done():
			return "", false, ctx.Err()
		default:
			sim, err := entity.CosineSimilarity(queryVector, entry.QueryVector)
			if err != nil {
				continue
			}
			if sim > maxSimilarity {
				maxSimilarity = sim
				bestMatch = entry.ResponseText
			}
		}
	}

	if maxSimilarity >= c.threshold {
		return bestMatch, true, nil
	}

	return "", false, nil
}

func (c *InMemorySemanticCache) Set(ctx context.Context, queryText string, queryVector []float32, responseText string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	agentOwner, _ := entity.GetAgentOwner(ctx)

	c.entries[agentOwner] = append(c.entries[agentOwner], CacheEntry{
		QueryText:    queryText,
		QueryVector:  queryVector,
		ResponseText: responseText,
		CreatedAt:    time.Now(),
		AgentOwner:   agentOwner,
	})
	return nil
}

func (c *InMemorySemanticCache) Invalidate(ctx context.Context, agentOwner uuid.UUID) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.entries, agentOwner)
	return nil
}

func (c *InMemorySemanticCache) Close() error {
	return nil
}

// HybridRedisSemanticCache stores cache data in Redis and maintains a local in-memory index.
type HybridRedisSemanticCache struct {
	pool      *redis.Pool
	entries   map[uuid.UUID][]CacheEntry
	loaded    map[uuid.UUID]bool
	mu        sync.RWMutex
	threshold float64
	keyPrefix string
}

// NewHybridRedisSemanticCache initializes a hybrid semantic cache.
func NewHybridRedisSemanticCache(redisAddress string, threshold float64) (*HybridRedisSemanticCache, error) {
	if threshold <= 0 {
		threshold = 0.95
	}
	pool := &redis.Pool{
		MaxIdle:     3,
		MaxActive:   10,
		IdleTimeout: 240 * time.Second,
		Dial: func() (redis.Conn, error) {
			return redis.Dial("tcp", redisAddress, redis.DialConnectTimeout(5*time.Second))
		},
	}

	// Test connection
	conn := pool.Get()
	defer conn.Close()
	if _, err := conn.Do("PING"); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to connect to Redis for semantic cache: %w", err)
	}

	return &HybridRedisSemanticCache{
		pool:      pool,
		entries:   make(map[uuid.UUID][]CacheEntry),
		loaded:    make(map[uuid.UUID]bool),
		threshold: threshold,
		keyPrefix: "cache:semantic:entries",
	}, nil
}

func (c *HybridRedisSemanticCache) loadEntriesForOwner(agentOwner uuid.UUID) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.loaded[agentOwner] {
		return nil
	}

	conn := c.pool.Get()
	defer conn.Close()

	key := fmt.Sprintf("%s:%s", c.keyPrefix, agentOwner.String())
	values, err := redis.ByteSlices(conn.Do("LRANGE", key, 0, -1))
	if err != nil && !errors.Is(err, redis.ErrNil) {
		return err
	}

	for _, val := range values {
		var entry CacheEntry
		if err := json.Unmarshal(val, &entry); err == nil {
			c.entries[agentOwner] = append(c.entries[agentOwner], entry)
		}
	}

	c.loaded[agentOwner] = true
	return nil
}

func (c *HybridRedisSemanticCache) Get(ctx context.Context, queryVector []float32) (string, bool, error) {
	agentOwner, _ := entity.GetAgentOwner(ctx)
	if err := c.loadEntriesForOwner(agentOwner); err != nil {
		fmt.Printf("Warning: failed to load semantic cache from Redis for owner %s: %v\n", agentOwner, err)
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	entries := c.entries[agentOwner]
	var bestMatch string
	var maxSimilarity float64 = -1.0

	for _, entry := range entries {
		select {
		case <-ctx.Done():
			return "", false, ctx.Err()
		default:
			sim, err := entity.CosineSimilarity(queryVector, entry.QueryVector)
			if err != nil {
				continue
			}
			if sim > maxSimilarity {
				maxSimilarity = sim
				bestMatch = entry.ResponseText
			}
		}
	}

	if maxSimilarity >= c.threshold {
		return bestMatch, true, nil
	}

	return "", false, nil
}

func (c *HybridRedisSemanticCache) Set(ctx context.Context, queryText string, queryVector []float32, responseText string) error {
	agentOwner, _ := entity.GetAgentOwner(ctx)
	if err := c.loadEntriesForOwner(agentOwner); err != nil {
		fmt.Printf("Warning: failed to load semantic cache from Redis for owner %s: %v\n", agentOwner, err)
	}

	entry := CacheEntry{
		QueryText:    queryText,
		QueryVector:  queryVector,
		ResponseText: responseText,
		CreatedAt:    time.Now(),
		AgentOwner:   agentOwner,
	}

	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	conn := c.pool.Get()
	defer conn.Close()

	key := fmt.Sprintf("%s:%s", c.keyPrefix, agentOwner.String())
	if err := conn.Send("RPUSH", key, data); err != nil {
		return err
	}
	if err := conn.Send("LTRIM", key, -1000, -1); err != nil {
		return err
	}
	if err := conn.Flush(); err != nil {
		return err
	}

	for i := 0; i < 2; i++ {
		if _, err := conn.Receive(); err != nil {
			return err
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries[agentOwner] = append(c.entries[agentOwner], entry)
	if len(c.entries[agentOwner]) > 1000 {
		c.entries[agentOwner] = c.entries[agentOwner][len(c.entries[agentOwner])-1000:]
	}

	return nil
}

func (c *HybridRedisSemanticCache) Invalidate(ctx context.Context, agentOwner uuid.UUID) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.entries, agentOwner)
	delete(c.loaded, agentOwner)

	conn := c.pool.Get()
	defer conn.Close()

	key := fmt.Sprintf("%s:%s", c.keyPrefix, agentOwner.String())
	_, err := conn.Do("DEL", key)
	return err
}

func (c *HybridRedisSemanticCache) Close() error {
	return c.pool.Close()
}

// NewSemanticCacheFactory is a factory function to instantiate the matching SemanticCache implementation.
func NewSemanticCacheFactory(provider, redisAddress string, threshold float64) (ports.SemanticCache, error) {
	if strings.ToLower(provider) == "redis" || strings.ToLower(provider) == "valkey" {
		return NewHybridRedisSemanticCache(redisAddress, threshold)
	}
	return NewInMemorySemanticCache(threshold), nil
}
