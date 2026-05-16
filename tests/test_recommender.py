"""Unit tests for src/recommender.py — grade-based technique ranking."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.data import REQUIRED_COLUMNS
from src.recommender import LearningTechniqueRecommender


@pytest.fixture()
def sample_grades() -> dict[str, dict[str, float]]:
    return {
        "Type A": {"Tech One": 70.0, "Tech Two": 91.5, "Tech Three": 88.0},
        "Type B": {"Alpha": 60.0, "Beta": 62.5},
    }


def test_predict_returns_highest_grade_technique(sample_grades: dict) -> None:
    rec = LearningTechniqueRecommender(sample_grades)
    assert rec.predict("Type A") == "Tech Two"
    assert rec.predict("Type B") == "Beta"


def test_predict_with_grade_matches_predict_and_lookup(sample_grades: dict) -> None:
    rec = LearningTechniqueRecommender(sample_grades)
    out = rec.predict_with_grade("Type A")
    assert out == {"technique": "Tech Two", "expected_grade": 91.5}


def test_get_all_techniques_ranked_orders_descending_by_grade(sample_grades: dict) -> None:
    rec = LearningTechniqueRecommender(sample_grades)
    ranked = rec.get_all_techniques_ranked("Type A")
    assert ranked == [
        {"technique": "Tech Two", "expected_grade": 91.5},
        {"technique": "Tech Three", "expected_grade": 88.0},
        {"technique": "Tech One", "expected_grade": 70.0},
    ]


def test_unknown_course_type_raises_value_error(sample_grades: dict) -> None:
    rec = LearningTechniqueRecommender(sample_grades)
    with pytest.raises(ValueError, match="Unknown course type"):
        rec.predict("Missing Type")
    with pytest.raises(ValueError, match="Available"):
        rec.get_all_techniques_ranked("Missing Type")


def test_train_from_csv_string_updates_predictions() -> None:
    csv = io.StringIO(
        "Course Type,Learning Technique,Grade of Module (%)\n"
        "STEM-X,Recall,80\n"
        "STEM-X,Recall,90\n"
        "STEM-X,Maps,85\n"
        "Arts-Y,Speak,72\n"
        "Arts-Y,Speak,74\n"
    )
    df = pd.read_csv(csv)
    assert list(df.columns) == list(REQUIRED_COLUMNS)

    rec = LearningTechniqueRecommender({"Old": {"X": 0.0}})
    rec.train(df)

    assert set(rec.available_course_types) == {"STEM-X", "Arts-Y"}
    # Recall mean (80+90)/2 = 85; Maps = 85 — ties resolve via dict iteration order from pandas means.
    assert rec.predict("STEM-X") == "Maps"
    assert rec.predict_with_grade("STEM-X") == {"technique": "Maps", "expected_grade": 85.0}
    assert rec.predict("Arts-Y") == "Speak"
    assert rec.predict_with_grade("Arts-Y")["expected_grade"] == 73.0


def test_train_from_dataframe_matches_manual_compute(sample_grades: dict) -> None:
    rows = []
    for course_type, techs in sample_grades.items():
        for tech, grade in techs.items():
            rows.append(
                {
                    "Course Type": course_type,
                    "Learning Technique": tech,
                    "Grade of Module (%)": grade,
                }
            )
    df = pd.DataFrame(rows)
    rec = LearningTechniqueRecommender({"Placeholder": {"Z": 1.0}})
    rec.train(df)
    assert rec.technique_grades == sample_grades
