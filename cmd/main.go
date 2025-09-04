package main

import (
	"go-azure-openai/internal/server"
	"log"
	"os"

	"github.com/gofiber/fiber/v2"
)

func main() {
	app := server.NewServer("AI-Service")

	app.Get("/ai/chat", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"message": "Chat",
		})
	})

	// ใช้ PORT จาก env ถ้าไม่มีให้ default เป็น 3000
	port := os.Getenv("PORT")
	if port == "" {
		port = "3000"
	}

	log.Fatal(app.Listen(":" + port))
}
