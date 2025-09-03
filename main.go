package main

import (
	"context"
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
	reply, err := a.Chat(ctx, "แต่งกลอนเกี่ยวกับความรัก 4 บท", agent.WithTemperature(0.8))
	if err != nil {
		log.Fatalf("chat: %v", err)
	}
	fmt.Println("--- Reply ---")
	fmt.Println(reply)
}
