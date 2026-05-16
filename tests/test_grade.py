"""Unit tests for src/grade.py — exercises the offline fallback path."""

from __future__ import annotations

import pytest

from src.grade import Grade, grade_answer


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_blank_answer_scores_zero() -> None:
    g = grade_answer("Q?", "Some expected answer here", "")
    assert isinstance(g, Grade)
    assert g.score == 0
    assert g.is_correct is False
    assert g.source == "fallback"
    assert "No answer" in g.feedback


def test_high_overlap_is_correct() -> None:
    g = grade_answer(
        question="Define photosynthesis.",
        expected_answer="Photosynthesis is the process plants use to convert sunlight into chemical energy.",
        user_answer="Photosynthesis is the process plants use to convert sunlight into chemical energy.",
    )
    assert g.score == 5
    assert g.is_correct is True
    assert g.missing_points == []


def test_partial_overlap_scores_in_middle() -> None:
    g = grade_answer(
        question="Define photosynthesis.",
        expected_answer="Photosynthesis converts sunlight into chemical energy in plants.",
        user_answer="Plants use sunlight.",
    )
    assert 1 <= g.score <= 3
    assert g.is_correct is False
    assert g.missing_points  # at least one missing concept


def test_zero_overlap_scores_zero() -> None:
    g = grade_answer(
        question="Define photosynthesis.",
        expected_answer="Photosynthesis converts sunlight into chemical energy.",
        user_answer="Banana republic legislation.",
    )
    assert g.score == 0
    assert g.is_correct is False


def test_grade_to_dict_round_trip() -> None:
    g = grade_answer("Q?", "expected", "expected")
    data = g.to_dict()
    assert set(data.keys()) >= {"score", "is_correct", "feedback", "missing_points", "source"}
    assert data["score"] == g.score


def test_missing_points_are_capped_at_three() -> None:
    g = grade_answer(
        question="List concepts.",
        expected_answer=(
            "Mitochondria endoplasmic reticulum lysosome ribosome nucleus "
            "cytoplasm chloroplast vacuole cytoskeleton"
        ),
        user_answer="Nothing relevant here.",
    )
    assert len(g.missing_points) <= 3


def test_empty_expected_answer_is_handled() -> None:
    g = grade_answer("Q?", "", "Some answer")
    assert isinstance(g, Grade)
    # The fallback should be conservative and not crash on empty model answer.
    assert g.source == "fallback"
