"""Smoke test the full end-to-end pipeline with no external services."""

from __future__ import annotations

from src import (
    DEFAULT_TECHNIQUE_GRADES,
    EXAMPLE_SYLLABI,
    LearningStyleSystem,
    extract_syllabus,
    generate_quiz,
    generate_study_plan,
)


def test_full_pipeline_runs_offline(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    raw = EXAMPLE_SYLLABI["Calculus III"].encode("utf-8")
    parsed = extract_syllabus(raw, "math.txt")
    assert parsed.text

    system = LearningStyleSystem(DEFAULT_TECHNIQUE_GRADES)
    results = system.analyze_syllabus(parsed.text)
    assert results["course_type_scores"]
    assert results["top_techniques"]

    course_type = results["course_type_scores"][0][0]
    technique = results["top_techniques"][0]["technique"]

    plan = generate_study_plan(
        course_title="Test Course",
        course_type=course_type,
        primary_technique=technique,
        syllabus_text=parsed.text,
        weeks=2,
        hours_per_week=3,
    )
    assert plan.total_sessions > 0

    first = plan.sessions[0]
    quiz = generate_quiz(
        topic=first.topic,
        technique=first.technique,
        syllabus_text=parsed.text,
        n_questions=3,
    )
    assert quiz.questions
    assert quiz.source in {"fallback", "llm"}
