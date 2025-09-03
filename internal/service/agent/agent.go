package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
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
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

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
	client oaiClient
}

// oaiClient is a minimal interface of the go-openai client used by Agent.
// This allows injecting a fake client in unit tests.
type oaiClient interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
	CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
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

// NewWithClient creates an Agent using a provided client implementation.
// This is intended for tests or advanced usage where you want to inject
// a fake or custom client. The provided client must implement the two
// methods used by Agent (CreateChatCompletion, CreateChatCompletionStream).
func NewWithClient(cfg Config, client interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
	CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
}) (*Agent, error) {
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
	// assert to internal oaiClient
	oc, ok := client.(oaiClient)
	if !ok {
		return nil, errors.New("provided client does not implement required methods")
	}
	return &Agent{cfg: cfg, client: oc}, nil
}

// ChatOption allows customizing a single Chat call.
type ChatOption func(*chatParams)

type chatParams struct {
	system       string
	temperature  float32
	maxTokens    int
	outputSchema string
	// future: tools, response format, etc.
}

// WithSystem sets a system prompt.
func WithSystem(system string) ChatOption { return func(p *chatParams) { p.system = system } }

// WithTemperature sets sampling temperature (0-2, typical 0-1).
func WithTemperature(t float32) ChatOption { return func(p *chatParams) { p.temperature = t } }

// WithMaxTokens limits output tokens (0 lets API decide / defaults).
func WithMaxTokens(n int) ChatOption { return func(p *chatParams) { p.maxTokens = n } }

// WithOutputSchema instructs the agent to return a JSON object matching the provided schema.
// The schema is a free-form string (for example a JSON Schema or a simple description) that
// will be injected as a system instruction to the model.
func WithOutputSchema(schema string) ChatOption {
	return func(p *chatParams) { p.outputSchema = schema }
}

// StreamHandler receives incremental tokens. Return false to stop early.
type StreamHandler func(delta string) bool

// Simple helper to demonstrate usage from another package.
func Example() {
	ctx := context.Background()
	a, err := NewAuto(WithModel("gpt-4o-mini"))
	if err != nil {
		fmt.Println("init error:", err)
		return
	}
	// use structured output
	res, err := a.ChatStructured(ctx, "สวัสดี ช่วยเขียนกลอนความรัก 4 บทให้หน่อย")
	if err != nil {
		fmt.Println("chat error:", err)
		return
	}
	fmt.Println(res.Text)
}

// ChatResult is a structured representation of a chat response.
type ChatResult struct {
	Text         string                         `json:"text"`
	Model        string                         `json:"model,omitempty"`
	FinishReason string                         `json:"finish_reason,omitempty"`
	Tokens       int                            `json:"tokens,omitempty"`
	Raw          *openai.ChatCompletionResponse `json:"-"`
}

// ChatStructured sends a single-turn user prompt and returns a structured result.
func (a *Agent) ChatStructured(ctx context.Context, userPrompt string, opts ...ChatOption) (ChatResult, error) {
	var empty ChatResult
	if a == nil || a.client == nil {
		return empty, errors.New("agent not initialized")
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
		return empty, err
	}
	if len(resp.Choices) == 0 {
		return empty, errors.New("empty response choices")
	}
	r := ChatResult{
		Text:  resp.Choices[0].Message.Content,
		Model: resp.Model,
		Raw:   &resp,
	}
	if len(resp.Choices) > 0 {
		r.FinishReason = string(resp.Choices[0].FinishReason)
	}
	// Usage is a struct with TotalTokens in the go-openai client
	r.Tokens = resp.Usage.TotalTokens
	return r, nil
}

// ChatStructuredJSON calls ChatStructured but also attempts to parse the returned text
// as JSON into an interface{}. It respects the WithOutputSchema option which injects
// a system instruction asking the model to respond in the requested structured format.
func (a *Agent) ChatStructuredJSON(ctx context.Context, userPrompt string, opts ...ChatOption) (ChatResult, interface{}, error) {
	// detect schema option
	var p chatParams
	for _, o := range opts {
		o(&p)
	}
	if p.outputSchema != "" {
		// validate schema is valid JSON
		var tmp interface{}
		if err := json.Unmarshal([]byte(p.outputSchema), &tmp); err != nil {
			return ChatResult{}, nil, errors.New("invalid output_schema JSON: " + err.Error())
		}
		// generate a stricter system prompt that includes the schema and an example
		sys := generateSystemPromptFromJSONSchema(p.outputSchema)
		// prepend system instruction
		newOpts := make([]ChatOption, 0, len(opts)+1)
		newOpts = append(newOpts, WithSystem(sys))
		newOpts = append(newOpts, opts...)
		opts = newOpts
	}

	res, err := a.ChatStructured(ctx, userPrompt, opts...)
	if err != nil {
		return ChatResult{}, nil, err
	}
	var parsed interface{}
	if err := json.Unmarshal([]byte(res.Text), &parsed); err != nil {
		return res, nil, err
	}
	return res, parsed, nil
}

// generateSystemPromptFromJSONSchema builds a strict system instruction asking the model
// to respond with JSON conforming to the provided JSON Schema. It also embeds a small
// example generated from the schema to guide the model.
func generateSystemPromptFromJSONSchema(schema string) string {
	example := buildExampleFromSchema(schema)
	var b strings.Builder
	b.WriteString("You are a strict JSON generator.\n")
	b.WriteString("Respond with a single JSON object that conforms exactly to the given JSON Schema.\n")
	b.WriteString("Do NOT include any surrounding text, explanations, or markdown. Output MUST be valid JSON.\n")
	b.WriteString("If you cannot produce a valid object, respond with an empty JSON object {}.\n")
	b.WriteString("JSON Schema:\n")
	b.WriteString(schema)
	if example != "" {
		b.WriteString("\nExample output:\n")
		b.WriteString(example)
	}
	return b.String()
}

// buildExampleFromSchema attempts a tiny, best-effort example JSON from a minimal schema.
func buildExampleFromSchema(schema string) string {
	var s map[string]interface{}
	if err := json.Unmarshal([]byte(schema), &s); err != nil {
		return ""
	}
	// walk top-level properties if available
	props, ok := s["properties"].(map[string]interface{})
	if !ok {
		// fallback: return the raw schema as example
		return ""
	}
	example := map[string]interface{}{}
	for k, v := range props {
		if mv, ok := v.(map[string]interface{}); ok {
			if t, ok := mv["type"].(string); ok {
				switch t {
				case "string":
					example[k] = "example"
				case "number", "integer":
					example[k] = 0
				case "boolean":
					example[k] = true
				case "object":
					example[k] = map[string]interface{}{}
				case "array":
					// try to get items type
					if items, ok := mv["items"].(map[string]interface{}); ok {
						if it, ok := items["type"].(string); ok {
							switch it {
							case "string":
								example[k] = []interface{}{"example"}
							case "number", "integer":
								example[k] = []interface{}{0}
							case "boolean":
								example[k] = []interface{}{true}
							default:
								example[k] = []interface{}{}
							}
						} else {
							example[k] = []interface{}{}
						}
					} else {
						example[k] = []interface{}{}
					}
				default:
					example[k] = nil
				}
			} else {
				example[k] = nil
			}
		} else {
			example[k] = nil
		}
	}
	if out, err := json.Marshal(example); err == nil {
		return string(out)
	}
	return ""
}

// SchemaFromFields builds a minimal JSON Schema (as a string) from a map of property names -> JSON types.
// Example:
//
//	props := map[string]string{"name":"string", "age":"integer"}
//	schema, _ := SchemaFromFields(props, []string{"name"})
func SchemaFromFields(props map[string]string, required []string) (string, error) {
	// Build nested schema supporting dot-notation keys
	schema := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{},
	}
	// helper set for required full paths
	reqSet := map[string]struct{}{}
	for _, r := range required {
		reqSet[r] = struct{}{}
	}

	for key, spec := range props {
		parts := strings.Split(key, ".")
		// parse spec for shorthand and default: e.g., "text=hello"
		typ, def := parseTypeAndDefault(spec)
		typ = normalizeType(typ)

		// navigate/create nested objects
		cur := schema
		for i, part := range parts {
			propsMap := cur["properties"].(map[string]interface{})
			if i == len(parts)-1 {
				// set property at this level
				prop := map[string]interface{}{"type": typ}
				if def != nil {
					prop["default"] = def
				}
				propsMap[part] = prop
				// if this full key is required, add to parent.required
				fullKey := strings.Join(parts[:i+1], ".")
				if _, ok := reqSet[fullKey]; ok {
					addRequiredToObject(cur, part)
				}
			} else {
				// ensure object exists
				if existing, ok := propsMap[part]; ok {
					// existing may be map[string]interface{}
					if em, ok := existing.(map[string]interface{}); ok {
						// ensure it has properties
						if _, has := em["properties"]; !has {
							em["type"] = "object"
							em["properties"] = map[string]interface{}{}
						}

						cur = em
						continue
					}
				}
				// create new object
				newObj := map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
				propsMap[part] = newObj

				cur = newObj
			}
		}
	}

	// attach top-level required if any top-level keys were requested
	if len(required) > 0 {
		// gather top-level required (those without dot)
		topReq := []string{}
		for _, r := range required {
			if !strings.Contains(r, ".") {
				topReq = append(topReq, r)
			}
		}
		if len(topReq) > 0 {
			schema["required"] = topReq
		}
	}

	b, err := json.Marshal(schema)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// parseTypeAndDefault parses a spec like "text=hello" into type and default value (as interface{}).
func parseTypeAndDefault(spec string) (string, interface{}) {
	parts := strings.SplitN(spec, "=", 2)
	typ := strings.TrimSpace(parts[0])
	if typ == "" {
		typ = "string"
	}
	if len(parts) == 1 {
		return typ, nil
	}
	defStr := strings.TrimSpace(parts[1])
	// try to convert default according to type
	switch strings.ToLower(typ) {
	case "integer", "int":
		if v, err := strconv.Atoi(defStr); err == nil {
			return typ, v
		}
	case "number", "float":
		if v, err := strconv.ParseFloat(defStr, 64); err == nil {
			return typ, v
		}
	case "boolean", "bool":
		if v, err := strconv.ParseBool(defStr); err == nil {
			return typ, v
		}
	}
	// fallback keep as string
	return typ, defStr
}

// normalizeType maps shorthand types to JSON Schema types
func normalizeType(t string) string {
	switch strings.ToLower(t) {
	case "text", "string":
		return "string"
	case "int", "integer":
		return "integer"
	case "float", "number":
		return "number"
	case "bool", "boolean":
		return "boolean"
	case "object":
		return "object"
	case "array":
		return "array"
	default:
		return t
	}
}

// addRequiredToObject ensures the object map has a required array and appends the field
func addRequiredToObject(obj map[string]interface{}, field string) {
	if obj == nil {
		return
	}
	if _, ok := obj["required"]; !ok {
		obj["required"] = []string{field}
		return
	}
	if arr, ok := obj["required"].([]string); ok {
		obj["required"] = append(arr, field)
		obj["required"] = arr
		return
	}
	// if it's []interface{}, convert
	if arr2, ok := obj["required"].([]interface{}); ok {
		strs := make([]string, 0, len(arr2)+1)
		for _, v := range arr2 {
			if s, ok := v.(string); ok {
				strs = append(strs, s)
			}
		}
		strs = append(strs, field)
		obj["required"] = strs
		return
	}
}

// SchemaFromMap is a convenience wrapper when you don't need required fields.
func SchemaFromMap(props map[string]string) (string, error) {
	return SchemaFromFields(props, nil)
}

// ChatStreamStructured streams token chunks and returns a structured result (aggregated text + minimal metadata).
func (a *Agent) ChatStreamStructured(ctx context.Context, userPrompt string, handler StreamHandler, opts ...ChatOption) (ChatResult, error) {
	var empty ChatResult
	if a == nil || a.client == nil {
		return empty, errors.New("agent not initialized")
	}
	if handler == nil {
		return empty, errors.New("nil stream handler")
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
		return empty, err
	}
	defer stream.Close()
	var full string
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, context.Canceled) {
			return ChatResult{Text: full, Model: a.cfg.Model}, nil
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return ChatResult{Text: full, Model: a.cfg.Model}, err
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
	return ChatResult{Text: full, Model: a.cfg.Model}, nil
}

// keep compatibility: original Chat returns text only using ChatStructured
func (a *Agent) Chat(ctx context.Context, userPrompt string, opts ...ChatOption) (string, error) {
	r, err := a.ChatStructured(ctx, userPrompt, opts...)
	if err != nil {
		return "", err
	}
	return r.Text, nil
}

// keep compatibility for ChatStream
func (a *Agent) ChatStream(ctx context.Context, userPrompt string, handler StreamHandler, opts ...ChatOption) (string, error) {
	r, err := a.ChatStreamStructured(ctx, userPrompt, handler, opts...)
	if err != nil {
		return r.Text, err
	}
	return r.Text, nil
}
