package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/gomodule/redigo/redis"
)

// CacheEntry representa un elemento guardado en la caché semántica.
type CacheEntry struct {
	QueryText    string    `json:"query_text"`
	QueryVector  []float32 `json:"query_vector"`
	ResponseText string    `json:"response_text"`
	CreatedAt    time.Time `json:"created_at"`
}

// SemanticCache define la interfaz para interactuar con la caché semántica.
type SemanticCache interface {
	Get(ctx context.Context, queryVector []float32) (string, bool, error)
	Set(ctx context.Context, queryText string, queryVector []float32, responseText string) error
	Close() error
}

// InMemorySemanticCache implementa la caché semántica enteramente en memoria RAM caliente.
type InMemorySemanticCache struct {
	entries   []CacheEntry
	mu        sync.RWMutex
	threshold float64
}

// NewInMemorySemanticCache inicializa una instancia de caché semántica pura en memoria.
func NewInMemorySemanticCache(threshold float64) *InMemorySemanticCache {
	if threshold <= 0 {
		threshold = 0.95
	}
	return &InMemorySemanticCache{
		entries:   make([]CacheEntry, 0),
		threshold: threshold,
	}
}

// Get calcula la similitud del coseno de todas las entradas cacheadas frente al vector de consulta.
func (c *InMemorySemanticCache) Get(ctx context.Context, queryVector []float32) (string, bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var bestMatch string
	var maxSimilarity float64 = -1.0

	for _, entry := range c.entries {
		select {
		case <-ctx.Done():
			return "", false, ctx.Err()
		default:
			sim, err := CosineSimilarity(queryVector, entry.QueryVector)
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

// Set añade un par pregunta-respuesta al caché en memoria.
func (c *InMemorySemanticCache) Set(ctx context.Context, queryText string, queryVector []float32, responseText string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = append(c.entries, CacheEntry{
		QueryText:    queryText,
		QueryVector:  queryVector,
		ResponseText: responseText,
		CreatedAt:    time.Now(),
	})
	return nil
}

// Close es un método dummy para cumplir con la interfaz SemanticCache.
func (c *InMemorySemanticCache) Close() error {
	return nil
}

// HybridRedisSemanticCache almacena los datos de caché en Redis para persistencia y mantiene un índice local en memoria para búsquedas semánticas de milisegundos.
type HybridRedisSemanticCache struct {
	pool      *redis.Pool
	entries   []CacheEntry
	mu        sync.RWMutex
	threshold float64
	keyPrefix string
}

// NewHybridRedisSemanticCache inicializa una caché semántica híbrida que sincroniza memoria local con persistencia en Redis.
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

	// Probar conexión a Redis
	conn := pool.Get()
	defer conn.Close()
	if _, err := conn.Do("PING"); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to connect to Redis for semantic cache: %w", err)
	}

	cache := &HybridRedisSemanticCache{
		pool:      pool,
		entries:   make([]CacheEntry, 0),
		threshold: threshold,
		keyPrefix: "cache:semantic:entries",
	}

	// Cargar entradas existentes en Redis al índice de memoria RAM caliente
	if err := cache.loadEntriesFromRedis(); err != nil {
		// Log error but do not block startup
		fmt.Printf("Warning: failed to warm up semantic cache from Redis: %v\n", err)
	}

	return cache, nil
}

func (c *HybridRedisSemanticCache) loadEntriesFromRedis() error {
	conn := c.pool.Get()
	defer conn.Close()

	// Obtener todas las entradas almacenadas en la lista de caché de Redis
	values, err := redis.ByteSlices(conn.Do("LRANGE", c.keyPrefix, 0, -1))
	if err != nil && !errors.Is(err, redis.ErrNil) {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	for _, val := range values {
		var entry CacheEntry
		if err := json.Unmarshal(val, &entry); err == nil {
			c.entries = append(c.entries, entry)
		}
	}

	fmt.Printf("[Semantic Cache] Warmed up with %d entries from Redis\n", len(c.entries))
	return nil
}

// Get busca la similitud de coseno en la caché caliente local.
func (c *HybridRedisSemanticCache) Get(ctx context.Context, queryVector []float32) (string, bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var bestMatch string
	var maxSimilarity float64 = -1.0

	for _, entry := range c.entries {
		select {
		case <-ctx.Done():
			return "", false, ctx.Err()
		default:
			sim, err := CosineSimilarity(queryVector, entry.QueryVector)
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

// Set guarda la nueva respuesta en la base de datos de Redis y actualiza el índice en caliente.
func (c *HybridRedisSemanticCache) Set(ctx context.Context, queryText string, queryVector []float32, responseText string) error {
	entry := CacheEntry{
		QueryText:    queryText,
		QueryVector:  queryVector,
		ResponseText: responseText,
		CreatedAt:    time.Now(),
	}

	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	// Escribir en Redis
	conn := c.pool.Get()
	defer conn.Close()

	if err := conn.Send("RPUSH", c.keyPrefix, data); err != nil {
		return err
	}
	// Mantener el tamaño máximo de la caché de Redis en 1000 elementos
	if err := conn.Send("LTRIM", c.keyPrefix, -1000, -1); err != nil {
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

	// Actualizar índice local en memoria RAM caliente
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = append(c.entries, entry)
	if len(c.entries) > 1000 {
		c.entries = c.entries[len(c.entries)-1000:]
	}

	return nil
}

// Close cierra el pool de conexiones de Redis.
func (c *HybridRedisSemanticCache) Close() error {
	return c.pool.Close()
}

// CosineSimilarity calcula la similitud del coseno entre dos vectores densos de punto flotante de 32 bits.
func CosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0, errors.New("vector dimensions mismatch or empty")
	}
	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		valA := float64(a[i])
		valB := float64(b[i])
		dotProduct += valA * valB
		normA += valA * valA
		normB += valB * valB
	}
	if normA == 0.0 || normB == 0.0 {
		return 0.0, nil
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}

// NewSemanticCacheFactory es un helper factory para instanciar el proveedor de caché semántica adecuado.
func NewSemanticCacheFactory(provider, redisAddress string, threshold float64) (SemanticCache, error) {
	if strings.ToLower(provider) == "redis" || strings.ToLower(provider) == "valkey" {
		return NewHybridRedisSemanticCache(redisAddress, threshold)
	}
	return NewInMemorySemanticCache(threshold), nil
}
