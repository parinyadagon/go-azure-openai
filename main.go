package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

// OpenAIConfig เก็บค่า config สำหรับ Azure OpenAI
type OpenAIConfig struct {
	APIKey         string
	Endpoint       string
	ModelName      string
	DeploymentName string
	Timeout        time.Duration
}

// FieldSchema รองรับ nested JSON
type FieldSchema struct {
	Name        string
	Type        string // string, number, array, object
	Description string
	Fields      []FieldSchema // สำหรับ type == "object" หรือ array of objects
}

// NewAzureClient สร้าง OpenAI client สำหรับ Azure
func NewAzureClient(cfg OpenAIConfig) *openai.Client {
	clientCfg := openai.DefaultAzureConfig(cfg.APIKey, cfg.Endpoint)
	clientCfg.AzureModelMapperFunc = func(model string) string {
		return cfg.DeploymentName
	}
	return openai.NewClientWithConfig(clientCfg)
}

// BuildSchemaPrompt สร้าง prompt dynamic สำหรับ JSON complex schema
func BuildSchemaPrompt(topic string, fields []FieldSchema, instructions []string) string {
	var buildFields func(fields []FieldSchema, indent string) string
	buildFields = func(fields []FieldSchema, indent string) string {
		prompt := ""
		for _, f := range fields {
			switch f.Type {
			case "object":
				prompt += fmt.Sprintf("%s\"%s\": {\n", indent, f.Name)
				prompt += buildFields(f.Fields, indent+"  ")
				prompt += fmt.Sprintf("%s},\n", indent)
			case "array":
				if len(f.Fields) > 0 && f.Fields[0].Type == "object" {
					prompt += fmt.Sprintf("%s\"%s\": [\n%s  {\n", indent, f.Name, indent)
					prompt += buildFields(f.Fields, indent+"    ")
					prompt += fmt.Sprintf("%s  }\n%s],\n", indent, indent)
				} else {
					prompt += fmt.Sprintf("%s\"%s\": [\"%s\"],\n", indent, f.Name, f.Description)
				}
			default:
				prompt += fmt.Sprintf("%s\"%s\": \"%s (%s)\",\n", indent, f.Name, f.Description, f.Type)
			}
		}
		return prompt
	}

	prompt := fmt.Sprintf("Please explain \"%s\" in JSON format with the following structure:\n{\n", topic)
	prompt += buildFields(fields, "  ")
	prompt += "}\n"

	if len(instructions) > 0 {
		prompt += "Additional instructions:\n"
		for _, ins := range instructions {
			prompt += "- " + ins + "\n"
		}
	}

	prompt += "Ensure valid JSON only, no extra text."
	return prompt
}

// ValidateJSON recursive validation
func ValidateJSON(data interface{}, schema []FieldSchema) error {
	m, ok := data.(map[string]interface{})
	if !ok {
		return errors.New("top-level JSON is not an object")
	}

	for _, f := range schema {
		val, exists := m[f.Name]
		if !exists {
			return fmt.Errorf("missing field: %s", f.Name)
		}
		switch f.Type {
		case "string":
			if _, ok := val.(string); !ok {
				return fmt.Errorf("field '%s' should be string", f.Name)
			}
		case "number":
			if _, ok := val.(float64); !ok {
				return fmt.Errorf("field '%s' should be number", f.Name)
			}
		case "array":
			arr, ok := val.([]interface{})
			if !ok {
				return fmt.Errorf("field '%s' should be array", f.Name)
			}
			if len(f.Fields) > 0 && f.Fields[0].Type == "object" {
				for _, item := range arr {
					if err := ValidateJSON(item, f.Fields); err != nil {
						return fmt.Errorf("array item in '%s' invalid: %v", f.Name, err)
					}
				}
			}
		case "object":
			if err := ValidateJSON(val, f.Fields); err != nil {
				return fmt.Errorf("object field '%s' invalid: %v", f.Name, err)
			}
		}
	}
	return nil
}

// FetchJSONFromAI ส่ง prompt และ validate complex schema
func FetchJSONFromAI(cfg OpenAIConfig, prompt string, schema []FieldSchema) (map[string]interface{}, error) {
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	client := NewAzureClient(cfg)
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: cfg.ModelName,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("chat completion error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from AI")
	}

	output := resp.Choices[0].Message.Content
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(output), &data); err != nil {
		log.Printf("Raw AI output: %s", output)
		return nil, fmt.Errorf("json unmarshal error: %w", err)
	}

	if err := ValidateJSON(data, schema); err != nil {
		return nil, fmt.Errorf("JSON validation failed: %w", err)
	}

	return data, nil
}

func main() {
	godotenv.Load()
	cfg := OpenAIConfig{
		APIKey:         os.Getenv("AZURE_OPENAI_KEY"),
		Endpoint:       os.Getenv("AZURE_OPENAI_ENDPOINT"),
		ModelName:      "gpt-4o-mini",
		DeploymentName: "gpt-4o-mini",
		Timeout:        15 * time.Second,
	}

	// Nested/complex schema example
	schema := []FieldSchema{
		{
			Name:        "clubs",
			Type:        "string",
			Description: "name of the football in the premier league",
		},
		{
			Name: "players",
			Type: "array",
			Fields: []FieldSchema{
				{
					Name:        "name",
					Type:        "string",
					Description: "player name in the club",
				},
				{
					Name:        "number",
					Type:        "string",
					Description: "player's number",
				},
			},
		},
	}

	instructions := []string{
		"Use concise language",
		"Return arrays as JSON arrays",
		"Avoid extra explanation outside JSON",
	}

	prompt := BuildSchemaPrompt("premier league", schema, instructions)
	jsonData, err := FetchJSONFromAI(cfg, prompt, schema)
	if err != nil {
		log.Fatalf("Error fetching JSON from AI: %v", err)
	}

	// Print nicely
	fmt.Println("=== AI JSON Output ===")
	b, _ := json.MarshalIndent(jsonData, "", "  ")
	fmt.Println(string(b))
}
