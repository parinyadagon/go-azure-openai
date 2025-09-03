package agent

import (
	"context"
	"testing"

	openai "github.com/sashabaranov/go-openai"
)

// fakeClient implements oaiClient for testing
type fakeClient struct {
	resp openai.ChatCompletionResponse
	err  error
}

func (f *fakeClient) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	return f.resp, f.err
}

func (f *fakeClient) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error) {
	return nil, nil
}

func TestChatStructured_HappyPath(t *testing.T) {
	f := &fakeClient{
		resp: openai.ChatCompletionResponse{
			ID:    "r1",
			Model: "gpt-test",
			Choices: []openai.ChatCompletionChoice{{
				Message:      openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "hello world"},
				FinishReason: openai.FinishReason("stop"),
			}},
			Usage: openai.Usage{TotalTokens: 5},
		},
		err: nil,
	}
	a := &Agent{cfg: Config{Model: "gpt-test"}, client: f}
	ctx := context.Background()
	res, err := a.ChatStructured(ctx, "hi")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if res.Text != "hello world" {
		t.Fatalf("unexpected text: %s", res.Text)
	}
	if res.Model != "gpt-test" {
		t.Fatalf("unexpected model: %s", res.Model)
	}
	if res.Tokens != 5 {
		t.Fatalf("unexpected tokens: %d", res.Tokens)
	}
}

func TestChatStructuredJSON_Parse(t *testing.T) {
	f := &fakeClient{
		resp: openai.ChatCompletionResponse{
			ID:    "r2",
			Model: "gpt-test",
			Choices: []openai.ChatCompletionChoice{{
				Message:      openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: `{"greeting":"hello","value":123}`},
				FinishReason: openai.FinishReason("stop"),
			}},
			Usage: openai.Usage{TotalTokens: 7},
		},
		err: nil,
	}
	a := &Agent{cfg: Config{Model: "gpt-test"}, client: f}
	ctx := context.Background()
	res, parsed, err := a.ChatStructuredJSON(ctx, "hi", WithOutputSchema(`{"greeting":"string","value":"number"}`))
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if res.Text == "" {
		t.Fatalf("expected text, got empty")
	}
	m, ok := parsed.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map result, got %T", parsed)
	}
	if m["greeting"] != "hello" {
		t.Fatalf("unexpected greeting: %v", m["greeting"])
	}
}
