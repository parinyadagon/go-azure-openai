package prompt

import (
	"strings"
	"testing"
)

func TestDefaultB1CriteriaStructure(t *testing.T) {
	cs := DefaultB1Criteria()
	if len(cs) != 3 {
		t.Fatalf("expected 3 criteria, got %d", len(cs))
	}
	if cs[0].Title != "Content" {
		t.Fatalf("unexpected first title: %s", cs[0].Title)
	}
	if cs[1].Index != 2 || cs[2].Index != 3 {
		t.Fatalf("unexpected indices: %+v", cs)
	}
	// total max should be 15 (5+5+5)
	var total float64
	for _, c := range cs {
		total += c.MaxScore
	}
	if total != 15 {
		t.Fatalf("expected total 15, got %v", total)
	}
}

func TestParseRawEvaluationTemplate(t *testing.T) {
	raw := "===INSTRUCTIONS===\nInstr line 1\n===CRITERIA===\nCRITERIA 1: Content (0-5 marks)\n- Test item (0-2 marks)\n===QUESTION===\nSample Question\n• Bullet one\n• Bullet two\n===ANSWER===\nquery\n"
	instr, criteria, qTitle, bullets, ans := ParseRawEvaluationTemplate(raw)
	if instr != "Instr line 1" {
		t.Fatalf("unexpected instructions: %q", instr)
	}
	if len(criteria) != 1 || criteria[0].Title != "Content" {
		t.Fatalf("criteria parse failed: %+v", criteria)
	}
	if qTitle != "Sample Question" {
		t.Fatalf("question title parse failed: %q", qTitle)
	}
	if len(bullets) != 2 {
		t.Fatalf("expected 2 bullets, got %d", len(bullets))
	}
	if ans != "query" {
		t.Fatalf("answer parse failed: %q", ans)
	}
}

func TestBuildCriterionTemplatesFromRaw(t *testing.T) {
	raw := "===INSTRUCTIONS===\nPlease eva\n===CRITERIA===\nCRITERIA 1: A (0-2 marks)\n- A1 (0-1 marks)\nCRITERIA 2: B (0-3 marks)\n- B1 (0-2 marks)\n===QUESTION===\nTitle\n• Q1?\n• Q2?\n===ANSWER===\nquery\n"
	objs, err := BuildCriterionTemplatesFromRaw(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(objs) != 2 {
		t.Fatalf("expected 2 objects, got %d", len(objs))
	}
	if objs[0].Title != "A" || objs[0].MaxScore != 2 || !strings.Contains(objs[0].Prompt, "CRITERIA 1: A (0-2 marks)") {
		t.Fatalf("first prompt malformed: %v", objs[0])
	}
	if !strings.Contains(objs[1].Prompt, "CRITERIA 2: B (0-3 marks)") {
		t.Fatalf("second prompt missing header")
	}
	if !strings.Contains(objs[0].Prompt, "===QUESTION===") {
		t.Fatalf("question section missing")
	}
	if !strings.Contains(objs[0].Prompt, "• Q1?") {
		t.Fatalf("bullet not rendered")
	}
}
