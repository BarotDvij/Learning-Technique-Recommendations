"""Learning technique recommender — ranks techniques by historical grades."""

from typing import Dict, List, Union

import pandas as pd

from src.data import compute_technique_grades


class LearningTechniqueRecommender:
    """
    Recommends learning techniques for a course type based on historical
    grade data. Initialized with a technique_grades dict; can be retrained
    from a new DataFrame.
    """

    def __init__(self, technique_grades: Dict[str, Dict[str, float]]):
        self.technique_grades = technique_grades
        self._refresh()

    def predict(self, course_type: str) -> str:
        """Return the single best technique for a course type."""
        self._check(course_type)
        return self._best[course_type]

    def predict_with_grade(self, course_type: str) -> Dict[str, Union[str, float]]:
        """Return the best technique and its expected average grade."""
        technique = self.predict(course_type)
        return {
            "technique": technique,
            "expected_grade": self.technique_grades[course_type][technique],
        }

    def get_all_techniques_ranked(self, course_type: str) -> List[Dict[str, Union[str, float]]]:
        """Return all techniques for a course type sorted best-first."""
        self._check(course_type)
        return [
            {"technique": t, "expected_grade": g}
            for t, g in sorted(
                self.technique_grades[course_type].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

    def train(self, data: pd.DataFrame) -> None:
        """Retrain from a DataFrame with the required columns."""
        self.technique_grades = compute_technique_grades(data)
        self._refresh()

    @property
    def available_course_types(self) -> List[str]:
        return list(self.technique_grades.keys())

    @property
    def all_learning_techniques(self) -> List[str]:
        return sorted({t for techs in self.technique_grades.values() for t in techs})

    def _refresh(self) -> None:
        self._best: Dict[str, str] = {
            ct: max(techs.items(), key=lambda x: x[1])[0]
            for ct, techs in self.technique_grades.items()
        }

    def _check(self, course_type: str) -> None:
        if course_type not in self.technique_grades:
            raise ValueError(
                f"Unknown course type '{course_type}'. Available: {self.available_course_types}"
            )
