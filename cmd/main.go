package main

import (
	"go-azure-openai/internal/app"
	"log"
	"os"
)

func main() {
	appServer := app.NewServer()
	port := os.Getenv("PORT")
	if port == "" {
		port = "8888"
	}
	if err := appServer.Listen(":" + port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
