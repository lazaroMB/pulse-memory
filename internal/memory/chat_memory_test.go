package memory

import (
	"context"
	"testing"
	"time"
)

func TestInMemoryChatMemory(t *testing.T) {
	ctx := context.Background()
	mem := NewInMemoryChatMemory()
	defer mem.Close()

	sessionID := "test-session-123"

	// 1. Get history of empty session
	history, err := mem.GetSessionHistory(ctx, sessionID, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(history) != 0 {
		t.Errorf("expected 0 messages, got %d", len(history))
	}

	// 2. Append messages
	msg1 := ChatMessage{Role: "user", Content: "Hello", Timestamp: time.Now()}
	msg2 := ChatMessage{Role: "assistant", Content: "Hi there!", Timestamp: time.Now()}

	if err := mem.AppendMessage(ctx, sessionID, msg1); err != nil {
		t.Fatalf("failed to append message 1: %v", err)
	}
	if err := mem.AppendMessage(ctx, sessionID, msg2); err != nil {
		t.Fatalf("failed to append message 2: %v", err)
	}

	// 3. Get history with limits
	history, err = mem.GetSessionHistory(ctx, sessionID, 1)
	if err != nil {
		t.Fatalf("failed to get history: %v", err)
	}
	if len(history) != 1 {
		t.Fatalf("expected 1 message, got %d", len(history))
	}
	if history[0].Content != "Hi there!" {
		t.Errorf("expected 'Hi there!', got '%s'", history[0].Content)
	}

	history, err = mem.GetSessionHistory(ctx, sessionID, 10)
	if err != nil {
		t.Fatalf("failed to get history: %v", err)
	}
	if len(history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(history))
	}
	if history[0].Content != "Hello" || history[1].Content != "Hi there!" {
		t.Errorf("unexpected message order or content: %v", history)
	}

	// 4. Test capping logic (limit to last 50)
	for i := 0; i < 60; i++ {
		msg := ChatMessage{Role: "user", Content: "spam", Timestamp: time.Now()}
		_ = mem.AppendMessage(ctx, sessionID, msg)
	}

	history, err = mem.GetSessionHistory(ctx, sessionID, 100)
	if err != nil {
		t.Fatalf("failed to get history: %v", err)
	}
	if len(history) != 50 {
		t.Errorf("expected capped history at 50, got %d", len(history))
	}

	// 5. Clear session
	if err := mem.ClearSession(ctx, sessionID); err != nil {
		t.Fatalf("failed to clear session: %v", err)
	}

	history, err = mem.GetSessionHistory(ctx, sessionID, 10)
	if err != nil {
		t.Fatalf("failed to get history: %v", err)
	}
	if len(history) != 0 {
		t.Errorf("expected history to be empty after clear, got %d", len(history))
	}
}

func TestChatMemoryFactory(t *testing.T) {
	cfg := ChatMemoryConfig{
		Provider: "in-memory",
	}

	mem, err := NewChatMemory(cfg)
	if err != nil {
		t.Fatalf("failed to create chat memory: %v", err)
	}
	defer mem.Close()

	if _, ok := mem.(*InMemoryChatMemory); !ok {
		t.Errorf("expected InMemoryChatMemory, got %T", mem)
	}
}
