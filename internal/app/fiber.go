package app

import (
	p "go-azure-openai/internal/service/prompt"

	"github.com/gofiber/fiber/v2"
)

// NewServer creates and configures a Fiber server instance.
func NewServer() *fiber.App {
	app := fiber.New()

	// POST /api/prompt
	// Accepts form-data or raw body field 'raw' containing a full combined template.
	// Returns JSON: {"templates": ["...", "...", ...]}
	app.Post("/api/prompt", func(c *fiber.Ctx) error {
		raw := c.FormValue("message")
		if raw == "" {
			body := c.Body()
			if len(body) > 0 {
				raw = string(body)
			}
		}
		if raw == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "missing raw template content"})
		}
		objs, err := p.BuildCriterionTemplatesFromRaw(raw)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(fiber.Map{"data": objs, "count": len(objs)})
	})

	return app
}
