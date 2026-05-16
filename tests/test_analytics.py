"""Unit tests for src/analytics.py — pure offline coverage of the practice log."""

from __future__ import annotations

import pytest

from src.analytics import PracticeLog, PracticeRecord
from src.grade import Grade


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _grade(score: int, source: str = "fallback") -> Grade:
    """Build a Grade directly without invoking grade_answer()."""
    return Grade(
        score=score,
        is_correct=score >= 4,
        feedback=f"score {score}",
        missing_points=[],
        source=source,
    )


def _log_with_records(specs: list[tuple[str, str, int]]) -> PracticeLog:
    """specs: list of (topic, technique, score). Timestamps are deterministic."""
    log = PracticeLog()
    for i, (topic, technique, score) in enumerate(specs):
        log.record_grade(
            session_id=f"sess-{i}",
            topic=topic,
            technique=technique,
            question=f"Question about {topic} #{i}",
            grade=_grade(score),
            timestamp=f"2026-05-16T00:00:{i:02d}",
        )
    return log


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_empty_log_returns_zero_value_summary_stats() -> None:
    log = PracticeLog()
    stats = log.summary_stats()
    assert stats == {
        "total_attempted": 0,
        "total_correct": 0,
        "accuracy_pct": 0.0,
        "avg_score": 0.0,
        "topics_seen": 0,
        "techniques_seen": 0,
    }
    assert log.score_trend() == []
    assert log.weak_topics() == []
    assert log.recommended_review() == []
    assert len(log) == 0


def test_record_grade_appends_record_and_updates_totals() -> None:
    log = PracticeLog()

    record = log.record_grade(
        session_id="w1-s1-Active Recall",
        topic="Limits",
        technique="Active Recall",
        question="Define a limit.",
        grade=_grade(5, source="llm"),
    )

    assert isinstance(record, PracticeRecord)
    assert record.score == 5
    assert record.is_correct is True
    assert record.grade_source == "llm"
    assert record.timestamp  # auto-populated

    log.record_grade(
        session_id="w1-s1-Active Recall",
        topic="Limits",
        technique="Active Recall",
        question="State the squeeze theorem.",
        grade=_grade(2),
    )
    log.record_grade(
        session_id="w1-s2-Feynman Technique",
        topic="Derivatives",
        technique="Feynman Technique",
        question="Explain the chain rule simply.",
        grade=_grade(4),
    )

    stats = log.summary_stats()
    assert stats["total_attempted"] == 3
    assert stats["total_correct"] == 2  # scores 5 and 4
    assert stats["accuracy_pct"] == pytest.approx(66.7, abs=0.1)
    assert stats["avg_score"] == pytest.approx(11 / 3, abs=0.01)
    assert stats["topics_seen"] == 2
    assert stats["techniques_seen"] == 2


def test_weak_topics_excludes_topics_with_too_few_attempts() -> None:
    log = _log_with_records([
        ("Stoichiometry", "Worked Example Analysis", 1),  # only 1 attempt → ignored
        ("Kinematics", "Worked Example Analysis", 1),
        ("Kinematics", "Worked Example Analysis", 2),
        ("Kinematics", "Worked Example Analysis", 2),
    ])
    weak = log.weak_topics(min_attempts=2, max_avg=3.0)

    topics = {w["topic"] for w in weak}
    assert "Kinematics" in topics
    assert "Stoichiometry" not in topics
    kine = next(w for w in weak if w["topic"] == "Kinematics")
    assert kine["attempts"] == 3
    assert kine["avg_score"] == pytest.approx(5 / 3, abs=0.01)


def test_weak_topics_excludes_topics_with_high_avg_score() -> None:
    log = _log_with_records([
        ("Mastered", "Active Recall", 5),
        ("Mastered", "Active Recall", 5),
        ("Mastered", "Active Recall", 4),
        ("Struggling", "Active Recall", 1),
        ("Struggling", "Active Recall", 2),
    ])
    weak = log.weak_topics(min_attempts=2, max_avg=3.0)
    topics = {w["topic"] for w in weak}

    assert "Mastered" not in topics  # avg = 4.67, above the cutoff
    assert "Struggling" in topics
    # Tighten the bar — Struggling now needs avg <= 1.0, which it doesn't have.
    assert log.weak_topics(min_attempts=2, max_avg=1.0) == []


def test_score_trend_returns_expected_shape() -> None:
    log = _log_with_records([
        ("T", "Active Recall", 2),
        ("T", "Active Recall", 4),
        ("T", "Active Recall", 5),
    ])
    trend = log.score_trend(window=2)

    assert len(trend) == 3
    # x-axis is 1-indexed attempt counter
    assert [pt[0] for pt in trend] == [1, 2, 3]
    # Running average with window=2:
    #   attempt 1: avg(2)        = 2.0
    #   attempt 2: avg(2,4)      = 3.0
    #   attempt 3: avg(4,5)      = 4.5
    assert trend[0][1] == pytest.approx(2.0)
    assert trend[1][1] == pytest.approx(3.0)
    assert trend[2][1] == pytest.approx(4.5)


def test_to_dict_from_dict_round_trips_a_populated_log() -> None:
    log = _log_with_records([
        ("Limits", "Active Recall", 3),
        ("Limits", "Active Recall", 4),
        ("Derivatives", "Feynman Technique", 5),
    ])
    data = log.to_dict()

    assert data["version"] == 1
    assert len(data["records"]) == 3
    assert data["records"][0]["topic"] == "Limits"

    rebuilt = PracticeLog.from_dict(data)
    assert len(rebuilt) == 3
    assert rebuilt.summary_stats() == log.summary_stats()
    assert rebuilt.records[0].technique == "Active Recall"
    assert rebuilt.records[-1].grade_source == "fallback"


def test_recommended_review_prefers_weakest_topics() -> None:
    log = _log_with_records([
        ("Strong", "Active Recall", 5),
        ("Strong", "Active Recall", 5),
        ("Medium", "Feynman Technique", 3),
        ("Medium", "Feynman Technique", 3),
        ("Weak", "Conceptual Mapping", 1),
        ("Weak", "Conceptual Mapping", 2),
    ])
    recs = log.recommended_review(top_k=2)

    assert len(recs) == 2
    assert recs[0]["topic"] == "Weak"
    assert recs[0]["technique"] == "Conceptual Mapping"
    assert recs[0]["avg_score"] < recs[1]["avg_score"]
    for r in recs:
        assert r["reason"]
        assert "last_score" in r
        assert "avg_score" in r


def test_clear_empties_the_log() -> None:
    log = _log_with_records([("T", "Active Recall", 3)])
    assert len(log) == 1
    log.clear()
    assert len(log) == 0
    assert log.summary_stats()["total_attempted"] == 0
