package app

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatHandler_MissingMessage(t *testing.T) {
	server := NewServer(nil)
	req := httptest.NewRequest("POST", "/api/chat", strings.NewReader(""))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := server.Test(req, -1)
	if err != nil {
		t.Fatalf("server.Test error: %v", err)
	}
	if resp.StatusCode != 400 {
		t.Fatalf("expected 400 for missing message, got %d", resp.StatusCode)
	}
}

// Note: full integration tests that exercise parsing require injecting a fake Agent; those
// are covered at the agent package level (unit tests). This test file keeps lightweight
// server validation tests.
