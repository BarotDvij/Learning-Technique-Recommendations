"""Integrated pipeline — syllabus text in, ranked recommendations out."""

from typing import Dict, List

from src.classifier import SyllabusClassifier
from src.recommender import LearningTechniqueRecommender


class LearningStyleSystem:
    """End-to-end: raw syllabus text → ranked learning technique recommendations."""

    def __init__(
        self,
        technique_grades: Dict[str, Dict[str, float]],
        course_weight: float = 0.5,
    ):
        """
        Args:
            technique_grades: per-course-type technique → grade mapping
            course_weight: how much weight to give course-type match in the combined
                score, in [0, 1]. The complementary weight goes to technique grade.
        """
        if not 0 <= course_weight <= 1:
            raise ValueError("course_weight must be in [0, 1]")
        self.classifier = SyllabusClassifier()
        self.recommender = LearningTechniqueRecommender(technique_grades)
        self.course_weight = course_weight

    def analyze_syllabus(self, syllabus_text: str, top_n: int = 5) -> Dict:
        """
        Classify the syllabus, then rank techniques by a combined score:
            combined = course_weight * course_match
                     + (1 - course_weight) * (expected_grade / 100)
        Returns the top_n techniques across all course types plus per-type scores.
        """
        course_type_scores = self.classifier.classify(syllabus_text)
        tech_weight = 1.0 - self.course_weight

        results: List[Dict] = []
        for course_type, match_score in course_type_scores:
            for item in self.recommender.get_all_techniques_ranked(course_type):
                combined = (
                    self.course_weight * match_score
                    + tech_weight * (item["expected_grade"] / 100)
                )
                results.append({
                    "technique": item["technique"],
                    "expected_grade": item["expected_grade"],
                    "course_type": course_type,
                    "course_match": round(match_score * 100, 1),
                    "combined_score": round(combined * 100, 1),
                })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return {
            "course_type_scores": course_type_scores,
            "top_techniques": results[:top_n],
            "all_techniques": results,
        }
