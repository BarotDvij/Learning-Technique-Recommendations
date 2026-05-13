"""Integrated pipeline — syllabus text in, ranked recommendations out."""

from typing import Dict, List

from src.classifier import SyllabusClassifier
from src.recommender import LearningTechniqueRecommender
from src.research import get_evidence


class LearningStyleSystem:
    """End-to-end: raw syllabus text → ranked learning technique recommendations."""

    def __init__(
        self,
        technique_grades: Dict[str, Dict[str, float]],
        course_weight: float = 0.5,
    ):
        """
        Args:
            technique_grades: per-course-type technique → evidence-score mapping
            course_weight: weight on course-type match in the combined score,
                in [0, 1]. The complementary weight goes to the technique's
                evidence score.
        """
        if not 0 <= course_weight <= 1:
            raise ValueError("course_weight must be in [0, 1]")
        self.classifier = SyllabusClassifier()
        self.recommender = LearningTechniqueRecommender(technique_grades)
        self.course_weight = course_weight

    def analyze_syllabus(self, syllabus_text: str, top_n: int = 5) -> Dict:
        """
        Classify the syllabus, then rank techniques by a combined score:
            combined = course_weight       * course_match
                     + (1 - course_weight) * (evidence_score / 100)

        Each result is enriched with the underlying research record from
        :mod:`src.research` (citation, effect size, summary, evidence rating).
        """
        course_type_scores = self.classifier.classify(syllabus_text)
        tech_weight = 1.0 - self.course_weight

        results: List[Dict] = []
        for course_type, match_score in course_type_scores:
            for item in self.recommender.get_all_techniques_ranked(course_type):
                technique = item["technique"]
                evidence_score = item["expected_grade"]
                combined = (
                    self.course_weight * match_score
                    + tech_weight * (evidence_score / 100)
                )

                evidence = get_evidence(technique)
                research = (
                    {
                        "short_citation": evidence.short_citation,
                        "full_citation": evidence.full_citation,
                        "effect_size": evidence.effect_size,
                        "effect_metric": evidence.effect_metric,
                        "study_count": evidence.study_count,
                        "evidence_rating": evidence.evidence_rating,
                        "summary": evidence.summary,
                        "dunlosky_utility": evidence.dunlosky_utility,
                    }
                    if evidence
                    else None
                )

                results.append({
                    "technique": technique,
                    "expected_grade": evidence_score,
                    "evidence_score": evidence_score,
                    "course_type": course_type,
                    "course_match": round(match_score * 100, 1),
                    "combined_score": round(combined * 100, 1),
                    "research": research,
                })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return {
            "course_type_scores": course_type_scores,
            "top_techniques": results[:top_n],
            "all_techniques": results,
        }
