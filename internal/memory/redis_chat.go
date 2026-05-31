package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/gomodule/redigo/redis"
)

// RedisChatMemory implements ChatMemory using a Redis connection pool.
type RedisChatMemory struct {
	pool *redis.Pool
}

// NewRedisChatMemory instantiates a connection pool to a Redis or Valkey server.
func NewRedisChatMemory(address string) (*RedisChatMemory, error) {
	pool := &redis.Pool{
		MaxIdle:     5,
		MaxActive:   20,
		IdleTimeout: 240 * time.Second,
		Dial: func() (redis.Conn, error) {
			return redis.Dial("tcp", address, redis.DialConnectTimeout(5*time.Second))
		},
	}

	// Verify connectivity
	conn := pool.Get()
	defer conn.Close()
	if _, err := conn.Do("PING"); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to ping Redis/Valkey instance: %w", err)
	}

	return &RedisChatMemory{pool: pool}, nil
}

// AppendMessage serializes and pushes a message to a Redis list with TTL and automatic capping.
func (r *RedisChatMemory) AppendMessage(ctx context.Context, sessionID string, msg ChatMessage) error {
	conn := r.pool.Get()
	defer conn.Close()

	key := fmt.Sprintf("chat:session:%s", sessionID)
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	// Pipeline commands to execute in a single RTT
	if err := conn.Send("RPUSH", key, data); err != nil {
		return err
	}
	if err := conn.Send("EXPIRE", key, 86400); err != nil { // 24 hours expiry
		return err
	}
	if err := conn.Send("LTRIM", key, -50, -1); err != nil { // Cap list size at 50 messages
		return err
	}
	if err := conn.Flush(); err != nil {
		return err
	}

	// Retrieve responses
	for i := 0; i < 3; i++ {
		if _, err := conn.Receive(); err != nil {
			return err
		}
	}
	return nil
}

// GetSessionHistory retrieves up to the last 'limit' messages in chronological order.
func (r *RedisChatMemory) GetSessionHistory(ctx context.Context, sessionID string, limit int) ([]ChatMessage, error) {
	conn := r.pool.Get()
	defer conn.Close()

	key := fmt.Sprintf("chat:session:%s", sessionID)
	values, err := redis.ByteSlices(conn.Do("LRANGE", key, -limit, -1))
	if err != nil {
		return nil, err
	}

	history := make([]ChatMessage, len(values))
	for i, val := range values {
		var msg ChatMessage
		if err := json.Unmarshal(val, &msg); err != nil {
			return nil, err
		}
		history[i] = msg
	}
	return history, nil
}

// ClearSession purges all history for a session ID.
func (r *RedisChatMemory) ClearSession(ctx context.Context, sessionID string) error {
	conn := r.pool.Get()
	defer conn.Close()

	key := fmt.Sprintf("chat:session:%s", sessionID)
	_, err := conn.Do("DEL", key)
	return err
}

// Close closes the Redis connection pool.
func (r *RedisChatMemory) Close() error {
	return r.pool.Close()
}
