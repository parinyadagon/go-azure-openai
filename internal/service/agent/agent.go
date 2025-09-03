package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// Config holds minimal Azure OpenAI configuration.
// Values can be left empty to fall back to environment variables:
//
//	AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL, AZURE_OPENAI_DEPLOYMENT
type Config struct {
	Key        string
	Endpoint   string
	Model      string
	Deployment string // optional; if empty uses Model
	Timeout    time.Duration
}

// LoadEnv fills empty fields from environment variables.
func (c *Config) LoadEnv() {
	if c.Key == "" {
		c.Key = os.Getenv("AZURE_OPENAI_KEY")
	}
	if c.Endpoint == "" {
		c.Endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	if c.Model == "" {
		c.Model = os.Getenv("AZURE_OPENAI_MODEL")
	}
	if c.Deployment == "" {
		if d := os.Getenv("AZURE_OPENAI_DEPLOYMENT"); d != "" {
			c.Deployment = d
		}
	}
}

// Validate basic required fields.
func (c *Config) Validate() error {
	if c.Key == "" || c.Endpoint == "" || c.Model == "" {
		return errors.New("missing required azure openai configuration (need key, endpoint, model)")
	}
	return nil
}

// Agent is a lightweight wrapper around the OpenAI client to simplify common chat use cases.
type Agent struct {
	cfg    Config
	client *openai.Client
}

// GetConfig returns a copy of the agent configuration (read-only for caller).
func (a *Agent) GetConfig() Config { return a.cfg }

// New creates a new Agent using the provided config (with env fallbacks).
func New(cfg Config) (*Agent, error) {
	cfg.LoadEnv()
	if cfg.Timeout == 0 {
		cfg.Timeout = 60 * time.Second
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.Deployment == "" {
		cfg.Deployment = cfg.Model
	}
	oaiCfg := openai.DefaultAzureConfig(cfg.Key, cfg.Endpoint)
	// Map logical model -> deployment
	oaiCfg.AzureModelMapperFunc = func(model string) string {
		// Always return the explicit deployment for our configured model.
		if model == cfg.Model {
			return cfg.Deployment
		}
		// fallback: echo original (allows direct deployment usage)
		return model
	}
	client := openai.NewClientWithConfig(oaiCfg)
	return &Agent{cfg: cfg, client: client}, nil
}

// Option is a functional option to modify agent configuration before initialization.
type Option func(*Config)

// WithKey overrides API key.
func WithKey(v string) Option { return func(c *Config) { c.Key = v } }

// WithEndpoint overrides API endpoint.
func WithEndpoint(v string) Option { return func(c *Config) { c.Endpoint = v } }

// WithModel sets logical model name.
func WithModel(v string) Option { return func(c *Config) { c.Model = v } }

// WithDeployment sets deployment mapping explicitly.
func WithDeployment(v string) Option { return func(c *Config) { c.Deployment = v } }

// WithTimeout sets request timeout.
func WithTimeout(d time.Duration) Option { return func(c *Config) { c.Timeout = d } }

// NewAuto creates an Agent pulling defaults from environment variables first, then applying options.
// This allows super simple usage: a, _ := agent.NewAuto(agent.WithModel("gpt-4o-mini"))
func NewAuto(opts ...Option) (*Agent, error) {
	cfg := Config{}
	// load env first
	cfg.LoadEnv()
	for _, o := range opts {
		o(&cfg)
	}
	return New(cfg)
}

// ChatOption allows customizing a single Chat call.
type ChatOption func(*chatParams)

type chatParams struct {
	system      string
	temperature float32
	maxTokens   int
	// future: tools, response format, etc.
}

// WithSystem sets a system prompt.
func WithSystem(system string) ChatOption { return func(p *chatParams) { p.system = system } }

// WithTemperature sets sampling temperature (0-2, typical 0-1).
func WithTemperature(t float32) ChatOption { return func(p *chatParams) { p.temperature = t } }

// WithMaxTokens limits output tokens (0 lets API decide / defaults).
func WithMaxTokens(n int) ChatOption { return func(p *chatParams) { p.maxTokens = n } }

// Chat sends a single-turn user prompt and returns the assistant's reply text.
func (a *Agent) Chat(ctx context.Context, userPrompt string, opts ...ChatOption) (string, error) {
	if a == nil || a.client == nil {
		return "", errors.New("agent not initialized")
	}
	p := chatParams{temperature: 0.7}
	for _, o := range opts {
		o(&p)
	}
	msgs := make([]openai.ChatCompletionMessage, 0, 2)
	if p.system != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: p.system})
	}
	msgs = append(msgs, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: userPrompt})

	req := openai.ChatCompletionRequest{
		Model:       a.cfg.Model,
		Messages:    msgs,
		Temperature: p.temperature,
	}
	if p.maxTokens > 0 {
		req.MaxTokens = p.maxTokens
	}
	ctx, cancel := context.WithTimeout(ctx, a.cfg.Timeout)
	defer cancel()
	resp, err := a.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("empty response choices")
	}
	return resp.Choices[0].Message.Content, nil
}

// StreamHandler receives incremental tokens. Return false to stop early.
type StreamHandler func(delta string) bool

// ChatStream streams a response token-by-token (or chunk) invoking handler.
// It returns the full aggregated text (unless stopped early) and any error.
func (a *Agent) ChatStream(ctx context.Context, userPrompt string, handler StreamHandler, opts ...ChatOption) (string, error) {
	if a == nil || a.client == nil {
		return "", errors.New("agent not initialized")
	}
	if handler == nil {
		return "", errors.New("nil stream handler")
	}
	p := chatParams{temperature: 0.7}
	for _, o := range opts {
		o(&p)
	}
	msgs := make([]openai.ChatCompletionMessage, 0, 2)
	if p.system != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: p.system})
	}
	msgs = append(msgs, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: userPrompt})

	req := openai.ChatCompletionRequest{Model: a.cfg.Model, Messages: msgs, Temperature: p.temperature, Stream: true}
	if p.maxTokens > 0 {
		req.MaxTokens = p.maxTokens
	}
	ctx, cancel := context.WithTimeout(ctx, a.cfg.Timeout)
	defer cancel()
	stream, err := a.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return "", err
	}
	defer stream.Close()
	var full string
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, context.Canceled) {
			return full, nil
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return full, err
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta.Content
		if delta == "" { // may carry role/done markers
			continue
		}
		full += delta
		if cont := handler(delta); !cont {
			// stop early by canceling context
			cancel()
			break
		}
	}
	return full, nil
}

// Simple helper to demonstrate usage from another package.
func Example() {
	ctx := context.Background()
	a, err := NewAuto(WithModel("gpt-4o-mini"))
	if err != nil {
		fmt.Println("init error:", err)
		return
	}
	reply, err := a.Chat(ctx, "สวัสดี ช่วยเขียนกลอนความรัก 4 บทให้หน่อย")
	if err != nil {
		fmt.Println("chat error:", err)
		return
	}
	fmt.Println(reply)
}
