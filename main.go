package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"go-azure-openai/internal/service/agent"
)

func main() {
	// Minimal usage: rely on env vars for key/endpoint; specify model via option if needed.
	a, err := agent.NewAuto(agent.WithModel("gpt-4o-mini"))
	if err != nil {
		log.Fatalf("init agent: %v", err)
	}
	ctx := context.Background()
	// Example: request a structured JSON output
	schema := `{"poems":[{"stanza":"string"}]}`
	res, parsed, err := a.ChatStructuredJSON(ctx, "แต่งกลอนเกี่ยวกับความรัก 4 บท และส่งออกเป็น JSON ตาม schema", agent.WithTemperature(0.8), agent.WithOutputSchema(schema))
	if err != nil {
		// If parsing failed, err may be json.Unmarshal error; still print raw structured result
		log.Fatalf("chat: %v", err)
	}

	out, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println("--- Reply (structured) ---")
	fmt.Println(string(out))
	if parsed != nil {
		p, _ := json.MarshalIndent(parsed, "", "  ")
		fmt.Println("--- Parsed JSON ---")
		fmt.Println(string(p))
	}
}
