"""Unit tests for src/quiz.py — exercises the fallback path (no API key needed)."""

from __future__ import annotations

import os

from src.quiz import Quiz, QuizQuestion, TECHNIQUE_INSTRUCTIONS, generate_quiz


def _ensure_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_fallback_quiz_returned_without_key(monkeypatch) -> None:
    _ensure_no_api_key(monkeypatch)
    quiz = generate_quiz(
        topic="Stoichiometry",
        technique="Active Recall",
        n_questions=4,
    )
    assert isinstance(quiz, Quiz)
    assert quiz.source == "fallback"
    assert len(quiz.questions) == 4
    assert all(isinstance(q, QuizQuestion) for q in quiz.questions)
    assert any("GEMINI_API_KEY" in n for n in quiz.notes)


def test_question_count_is_clamped(monkeypatch) -> None:
    _ensure_no_api_key(monkeypatch)
    quiz = generate_quiz("Topic", "Active Recall", n_questions=99)
    assert len(quiz.questions) == 10  # upper bound

    quiz = generate_quiz("Topic", "Active Recall", n_questions=0)
    assert len(quiz.questions) == 1  # lower bound


def test_quiz_to_dict_round_trip(monkeypatch) -> None:
    _ensure_no_api_key(monkeypatch)
    quiz = generate_quiz("Limits", "Worked Example Analysis", n_questions=2)
    data = quiz.to_dict()
    assert data["topic"] == "Limits"
    assert data["technique"] == "Worked Example Analysis"
    assert len(data["questions"]) == 2
    assert "question" in data["questions"][0]
    assert "answer" in data["questions"][0]


def test_technique_instructions_cover_known_techniques() -> None:
    expected = {
        "Active Recall",
        "Worked Example Analysis",
        "Feynman Technique",
        "Conceptual Mapping",
        "Case Study Analysis",
        "Immersive Practice",
        "Incremental Skill Building",
        "Project-Based Learning",
        "Spaced Repetition",
    }
    assert expected.issubset(TECHNIQUE_INSTRUCTIONS.keys())


def test_unknown_technique_still_produces_quiz(monkeypatch) -> None:
    _ensure_no_api_key(monkeypatch)
    quiz = generate_quiz("Mystery topic", "Some Made-Up Technique", n_questions=3)
    assert quiz.source == "fallback"
    assert len(quiz.questions) == 3
