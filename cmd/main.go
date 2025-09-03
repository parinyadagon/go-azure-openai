package main

import (
	"go-azure-openai/internal/app"
	"go-azure-openai/internal/service/agent"
	"log"
	"os"
)

func main() {
	a, err := agent.NewAuto()
	if err != nil {
		log.Fatalf("init agent: %v", err)
	}
	// controller-provided JSON Schema (server-side)
	schema, _ := agent.SchemaFromMap(map[string]string{"author.name": "text", "author.age": "int"})
	appServer := app.NewServer(a, schema)
	port := os.Getenv("PORT")
	if port == "" {
		port = "8888"
	}
	if err := appServer.Listen(":" + port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
