package prompt

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// CriterionItem represents a single bullet point under a criterion with its maximum marks.
type CriterionItem struct {
	Description string
	MaxScore    float64
}

// Criterion represents a scoring criterion (e.g., Content) with a max score and its items.
type Criterion struct {
	// Index is the display order (1-based). If 0, order will be derived from slice position.
	Index    int
	Title    string
	MaxScore float64
	Items    []CriterionItem
}

// CriterionTemplate represents a generated template for a single criterion.
type CriterionTemplate struct {
	Title    string  `json:"title"`
	Prompt   string  `json:"prompt"`
	MaxScore float64 `json:"maxScore"`
}

// (All older template-building helpers removed; only parser + builder below retained.)

// formatScore formats a float score preserving .5 or .25 etc., but dropping trailing .0.
func formatScore(f float64) string {
	if f == float64(int64(f)) { // integer
		return fmt.Sprintf("%d", int64(f))
	}
	// Use up to 2 decimal places without trailing zeros beyond necessary.
	s := fmt.Sprintf("%.2f", f)
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	return s
}

// DefaultB1Criteria returns a pre-filled slice of Criterion for the provided B1 speaking template.
func DefaultB1Criteria() []Criterion {
	return []Criterion{
		{
			Index:    1,
			Title:    "Content",
			MaxScore: 5,
			Items: []CriterionItem{
				{Description: "Is it about the topic stated in the task?", MaxScore: 1},
				{Description: "Does it address all the notes mentioned in the task? Or does it answer the question(s) in the task?", MaxScore: 2},
				{Description: "Are the answers of an appropriate length for the task?", MaxScore: 2},
			},
		},
		{
			Index:    2,
			Title:    "Communicative Achievement & Organization",
			MaxScore: 5,
			Items: []CriterionItem{
				{Description: "Does the text use appropriate language and phrases to respond to all the notes?", MaxScore: 2},
				{Description: "Are the ideas presented in a logical order?", MaxScore: 1},
				{Description: "Does the text use a variety of linking words or cohesive devices (such as although, and, but, because, so that, whether etc., and referencing language)?", MaxScore: 1},
				{Description: "Is the purpose of the answer clear (e.g., agreeing, disagreeing, giving opinion, explaining)?", MaxScore: 1},
			},
		},
		{
			Index:    3,
			Title:    "Language Grammar and Vocabulary",
			MaxScore: 5,
			Items: []CriterionItem{
				{Description: "Does the text use a range of vocabulary?", MaxScore: 1.5},
				{Description: "Does the text use simple grammar accurately (e.g., basic tenses and simple clauses)?", MaxScore: 1},
				{Description: "Does it use some complex grammatical structures (such as relative clauses, passives, modal forms and tense contrasts)?", MaxScore: 1.5},
				{Description: "Is the spelling accurate enough for the meaning to be clear?", MaxScore: 1},
			},
		},
	}
}

// BuildDefaultB1Template convenience wrapper for common use case.
// (Removed helper wrappers; using only BuildCriterionTemplatesFromRaw now.)

// ParseRawEvaluationTemplate parses a full combined template text into its components.
// Expected markers: ===INSTRUCTIONS===, ===CRITERIA===, ===QUESTION===, ===ANSWER===
// Criteria lines like: CRITERIA 1: Content (0-5 marks)
// Item lines like: - Description  (0-2 marks)
func ParseRawEvaluationTemplate(raw string) (instructions string, criteria []Criterion, questionTitle string, questionBullets []string, answerPlaceholder string) {
	section := ""
	lines := strings.Split(raw, "\n")
	var instrBuilder, answerBuilder strings.Builder
	var currentCriterion *Criterion
	critPattern := regexp.MustCompile(`^CRITERIA\s+(\d+):\s+(.*?)\s*\((?:0-)?([0-9]+(?:\.[0-9]+)?)\s+marks?\)\s*$`)
	itemPattern := regexp.MustCompile(`^-\s+(.*?)\s*\((?:0-)?([0-9]+(?:\.[0-9]+)?)\s+marks?\)\s*$`)

	for _, rawLine := range lines {
		line := strings.TrimRight(rawLine, "\r")
		trimmed := strings.TrimSpace(line)
		switch trimmed {
		case "===INSTRUCTIONS===", "===CRITERIA===", "===QUESTION===", "===ANSWER===":
			section = trimmed
			continue
		}

		if trimmed == "" { // keep blank lines only in instructions and answer for fidelity
			if section == "===INSTRUCTIONS===" {
				instrBuilder.WriteString("\n")
			} else if section == "===ANSWER===" {
				answerBuilder.WriteString("\n")
			}
			continue
		}

		switch section {
		case "===INSTRUCTIONS===":
			if instrBuilder.Len() > 0 {
				instrBuilder.WriteString("\n")
			}
			instrBuilder.WriteString(trimmed)
		case "===CRITERIA===":
			if m := critPattern.FindStringSubmatch(trimmed); m != nil {
				// Start new criterion
				if currentCriterion != nil {
					criteria = append(criteria, *currentCriterion)
				}
				currentCriterion = &Criterion{}
				currentCriterion.Index = atoiSafe(m[1])
				currentCriterion.Title = m[2]
				currentCriterion.MaxScore = atofSafe(m[3])
			} else if m := itemPattern.FindStringSubmatch(trimmed); m != nil {
				if currentCriterion == nil {
					continue
				}
				currentCriterion.Items = append(currentCriterion.Items, CriterionItem{Description: m[1], MaxScore: atofSafe(m[2])})
			}
		case "===QUESTION===":
			// First non-bullet line -> questionTitle; bullet lines -> bullets
			isBullet := strings.HasPrefix(trimmed, "•") || strings.HasPrefix(trimmed, "-") || strings.HasPrefix(trimmed, "*")
			if !isBullet && questionTitle == "" {
				questionTitle = trimmed
			} else {
				// Normalize bullet
				if strings.HasPrefix(trimmed, "•") {
					questionBullets = append(questionBullets, strings.TrimSpace(strings.TrimPrefix(trimmed, "•")))
				} else if strings.HasPrefix(trimmed, "-") || strings.HasPrefix(trimmed, "*") {
					questionBullets = append(questionBullets, strings.TrimSpace(strings.TrimLeft(trimmed, "-* ")))
				} else {
					// treat as extra bullet
					questionBullets = append(questionBullets, trimmed)
				}
			}
		case "===ANSWER===":
			if answerBuilder.Len() > 0 {
				answerBuilder.WriteString("\n")
			}
			answerBuilder.WriteString(line)
		}
	}
	if currentCriterion != nil {
		criteria = append(criteria, *currentCriterion)
	}
	instructions = strings.TrimSpace(instrBuilder.String())
	answerPlaceholder = strings.TrimSpace(answerBuilder.String())
	// Re-add bullet prefix to bullets (we stored without leading symbol) when building output.
	// Clean bullet trims
	cleaned := make([]string, 0, len(questionBullets))
	for _, b := range questionBullets {
		cleaned = append(cleaned, strings.TrimSpace(b))
	}
	questionBullets = cleaned
	return
}

// BuildTemplatesPerCriterionFromRaw takes a combined raw template and returns one template per criterion.
// (Deprecated convenience wrapper removed.)

// BuildCriterionTemplatesFromRaw like BuildTemplatesPerCriterionFromRaw but returns structured objects with title.
func BuildCriterionTemplatesFromRaw(raw string) ([]CriterionTemplate, error) {
	instructions, criteria, questionTitle, questionBullets, answer := ParseRawEvaluationTemplate(raw)
	if len(criteria) == 0 {
		return nil, fmt.Errorf("no criteria parsed")
	}
	// Rebuild bullets (strip any existing bullet prefix, builder will add)
	stripped := make([]string, 0, len(questionBullets))
	for _, b := range questionBullets {
		b2 := strings.TrimSpace(b)
		b2 = strings.TrimLeft(b2, "•-* \t")
		if b2 != "" {
			stripped = append(stripped, b2)
		}
	}
	// Order criteria
	type ordCrit struct {
		order int
		c     Criterion
	}
	ordered := make([]ordCrit, 0, len(criteria))
	for i, c := range criteria {
		o := c.Index
		if o == 0 {
			o = i + 1
		}
		ordered = append(ordered, ordCrit{order: o, c: c})
	}
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].order < ordered[j].order })
	results := make([]CriterionTemplate, 0, len(ordered))
	for _, oc := range ordered {
		var b strings.Builder
		b.WriteString("===INSTRUCTIONS===\n")
		b.WriteString(strings.TrimSpace(instructions))
		b.WriteString("\n===CRITERIA===\n")
		b.WriteString(fmt.Sprintf("CRITERIA %d: %s (0-%s marks)\n", oc.order, oc.c.Title, formatScore(oc.c.MaxScore)))
		for _, item := range oc.c.Items {
			b.WriteString(fmt.Sprintf("- %s  (0-%s marks)\n", item.Description, formatScore(item.MaxScore)))
		}
		b.WriteString("===QUESTION===\n")
		if questionTitle != "" {
			b.WriteString(questionTitle + "\n")
		}
		for _, qb := range stripped {
			if qb != "" {
				b.WriteString("• " + qb + "\n")
			}
		}
		b.WriteString("===ANSWER===\n")
		if answer != "" {
			b.WriteString(answer)
		}
		results = append(results, CriterionTemplate{Title: oc.c.Title, Prompt: b.String(), MaxScore: oc.c.MaxScore})
	}
	return results, nil
}

// Helper convert string to int with fallback 0.
func atoiSafe(s string) int { var v int; fmt.Sscanf(s, "%d", &v); return v }

// Helper convert string to float with fallback 0.
func atofSafe(s string) float64 { var f float64; fmt.Sscanf(s, "%f", &f); return f }
