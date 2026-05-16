"""Unit tests for src/classifier.py — TF-IDF syllabus classification."""

from __future__ import annotations

from src.classifier import SyllabusClassifier


def test_course_types_match_defined_labels_and_order() -> None:
    clf = SyllabusClassifier()
    expected = list(SyllabusClassifier.COURSE_TYPE_DESCRIPTIONS.keys())
    assert clf.course_types == expected
    assert len(clf.course_types) == len(SyllabusClassifier.COURSE_TYPE_DESCRIPTIONS)


def test_classify_returns_all_types_sorted_descending_by_score() -> None:
    clf = SyllabusClassifier()
    ranked = clf.classify("mixed syllabus calculus essays cases projects language")
    assert len(ranked) == len(clf.course_types)
    names = [item[0] for item in ranked]
    assert set(names) == set(clf.course_types)
    scores = [item[1] for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_classify_math_heavy_text_ranks_applied_calculation_first() -> None:
    clf = SyllabusClassifier()
    text = (
        "Students solve quantitative problems using equations, formulas, calculus, "
        "derivatives, integrals, computational algorithms, and numerical methods."
    )
    ranked = clf.classify(text)
    assert ranked[0][0] == "Applied Calculation-Driven Learning"


def test_classify_conceptual_heavy_text_ranks_deep_conceptual_first() -> None:
    clf = SyllabusClassifier()
    text = (
        "Critical examination of theoretical frameworks, abstract principles, "
        "philosophical arguments, conceptual analysis, and essays."
    )
    ranked = clf.classify(text)
    assert ranked[0][0] == "Deep Conceptual Learning"


def test_classify_case_strategy_heavy_text_ranks_case_based_first() -> None:
    clf = SyllabusClassifier()
    text = (
        "Real-world case studies, strategic decision-making, management scenarios, "
        "business strategy analyses, and policy cases."
    )
    ranked = clf.classify(text)
    assert ranked[0][0] == "Case-Based & Strategic Learning"


def test_classify_language_heavy_text_ranks_language_communication_first() -> None:
    clf = SyllabusClassifier()
    text = (
        "Developing rhetoric, literature analysis, presentation skills, journalism "
        "writing exercises, and communication projects."
    )
    ranked = clf.classify(text)
    assert ranked[0][0] == "Language & Communication-Based Learning"


def test_classify_project_heavy_text_ranks_hands_on_project_first() -> None:
    clf = SyllabusClassifier()
    text = (
        "Hands-on laboratories, engineering design projects, building prototypes, "
        "programming portfolios, and practical demonstrations."
    )
    ranked = clf.classify(text)
    assert ranked[0][0] == "Hands-On, Project-Based Learning"


def test_classify_handles_empty_input_without_crashing() -> None:
    clf = SyllabusClassifier()
    ranked = clf.classify("")
    assert len(ranked) == len(clf.course_types)
