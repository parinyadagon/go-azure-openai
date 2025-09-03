package app

import (
	"context"
	"go-azure-openai/internal/service/agent"
	p "go-azure-openai/internal/service/prompt"
	"strconv"

	"github.com/gofiber/fiber/v2"
)

// NewServer creates and configures a Fiber server instance using the provided agent.
// chatSchema is a server-side JSON Schema (string) which the controller can set.
// If a is nil, the handler will attempt to construct an agent via NewAuto.
func NewServer(a *agent.Agent, chatSchema string) *fiber.App {
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

	// POST /api/chat
	// body/form: message (string), optional: system, temperature (float), max_tokens (int)
	app.Post("/api/chat", func(c *fiber.Ctx) error {
		msg := c.FormValue("message")
		if msg == "" {
			body := c.Body()
			if len(body) > 0 {
				msg = string(body)
			}
		}
		if msg == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "missing message"})
		}
		system := c.FormValue("system")
		tempStr := c.FormValue("temperature")
		maxTokStr := c.FormValue("max_tokens")
		// ignore client-provided output_schema: controller sets schema via chatSchema

		var opts []agent.ChatOption
		if system != "" {
			opts = append(opts, agent.WithSystem(system))
		}
		if tempStr != "" {
			if t, err := strconv.ParseFloat(tempStr, 32); err == nil {
				opts = append(opts, agent.WithTemperature(float32(t)))
			}
		}
		if maxTokStr != "" {
			if n, err := strconv.Atoi(maxTokStr); err == nil {
				opts = append(opts, agent.WithMaxTokens(n))
			}
		}
		if chatSchema != "" {
			opts = append(opts, agent.WithOutputSchema(chatSchema))
		}

		// ensure we have an agent
		if a == nil {
			aa, err := agent.NewAuto()
			if err != nil {
				return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
			}
			a = aa
		}

		ctx := context.Background()
		// If a server-side output schema is configured, try to parse JSON result
		if chatSchema != "" {
			res, parsed, err := a.ChatStructuredJSON(ctx, msg, opts...)
			if err != nil {
				// parsing failed -> 422 Unprocessable Entity
				return c.Status(fiber.StatusUnprocessableEntity).JSON(fiber.Map{"data": res, "parsed": nil, "parse_error": err.Error()})
			}
			return c.JSON(fiber.Map{"data": res, "parsed": parsed})
		}
		res, err := a.ChatStructured(ctx, msg, opts...)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(fiber.Map{"data": res})
	})

	return app
}
