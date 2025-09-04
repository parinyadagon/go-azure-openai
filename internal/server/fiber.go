package server

import (
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

func NewServer(AppName string) *fiber.App {
	// สร้าง instance ของ Fiber app
	app := fiber.New(fiber.Config{
		AppName: AppName,
	})

	// Middleware พื้นฐาน
	app.Use(recover.New()) // ป้องกัน server crash
	app.Use(logger.New())  // log request/response

	// Health check route
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"app":    AppName,
			"status": "ok",
		})
	})

	return app
}
