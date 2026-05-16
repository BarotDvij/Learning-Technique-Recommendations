"""Unit tests for src/research.py — evidence records and alignment scores."""

from __future__ import annotations

import pytest

from src.classifier import SyllabusClassifier
from src.research import (
    COURSE_TYPE_ALIGNMENT,
    EVIDENCE_WEIGHTS,
    RESEARCH,
    TechniqueEvidence,
    all_techniques,
    compute_evidence_scores,
    get_evidence,
)


def test_get_evidence_returns_record_for_known_technique() -> None:
    ev = get_evidence("Active Recall")
    assert isinstance(ev, TechniqueEvidence)
    assert ev.name == "Active Recall"
    assert ev.evidence_rating in EVIDENCE_WEIGHTS


def test_get_evidence_returns_none_for_unknown_technique() -> None:
    assert get_evidence("Not A Real Technique") is None


def test_compute_evidence_scores_matches_formula_and_bounds() -> None:
    scores = compute_evidence_scores()
    ct = "Applied Calculation-Driven Learning"
    tech = "Worked Example Analysis"
    alignment = COURSE_TYPE_ALIGNMENT[ct][tech]
    evidence = RESEARCH[tech]
    expected = round(EVIDENCE_WEIGHTS[evidence.evidence_rating] * alignment, 2)
    assert scores[ct][tech] == expected
    assert 0 <= scores[ct][tech] <= 100
    flat = [v for inner in scores.values() for v in inner.values()]
    assert max(flat) <= 100
    assert min(flat) >= 0


def test_compute_evidence_scores_covers_all_alignment_entries() -> None:
    scores = compute_evidence_scores()
    assert set(scores.keys()) == set(COURSE_TYPE_ALIGNMENT.keys())
    for course_type, tech_map in COURSE_TYPE_ALIGNMENT.items():
        assert set(scores[course_type].keys()) == set(tech_map.keys())


def test_all_techniques_is_sorted_and_matches_research_keys() -> None:
    names = all_techniques()
    assert names == sorted(names)
    assert set(names) == set(RESEARCH.keys())


def test_alignment_matrix_every_technique_exists_in_research() -> None:
    for course_type, tech_align in COURSE_TYPE_ALIGNMENT.items():
        for technique in tech_align:
            assert technique in RESEARCH, (
                f"Alignment references '{technique}' under '{course_type}' "
                "but it is missing from RESEARCH."
            )


def test_alignment_matrix_course_types_match_classifier_labels() -> None:
    clf = SyllabusClassifier()
    assert set(COURSE_TYPE_ALIGNMENT.keys()) == set(clf.course_types)


def test_every_technique_in_research_has_valid_evidence_rating() -> None:
    for name, ev in RESEARCH.items():
        assert ev.evidence_rating in EVIDENCE_WEIGHTS, (
            f"{name} has unknown evidence_rating {ev.evidence_rating!r}"
        )


@pytest.mark.parametrize(
    "technique",
    [
        "Worked Example Analysis",
        "Case Study Analysis",
        "Iterative Writing & Editing",
    ],
)
def test_get_evidence_parametrized_known_techniques(technique: str) -> None:
    assert get_evidence(technique) is not None
