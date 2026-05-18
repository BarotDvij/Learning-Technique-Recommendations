"""Syllabus classifier — maps raw syllabus text to a course type."""

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SyllabusClassifier:
    """
    Classifies raw syllabus text into one of the known course types using
    TF-IDF + cosine similarity against hand-crafted type descriptions.
    """

    COURSE_TYPE_DESCRIPTIONS = {
        "Applied Calculation-Driven Learning": (
            "mathematical formulas equations calculations quantitative numerical "
            "computational statistics models algorithms mathematics physics engineering "
            "economics finance problem sets computational exercises solving equations"
        ),
        "Deep Conceptual Learning": (
            "theoretical frameworks abstract concepts principles critical thinking "
            "theories conceptual analysis philosophy theoretical physics pure mathematics "
            "theoretical computer science theories concepts principles models theoretical "
            "essays"
        ),
        "Case-Based & Strategic Learning": (
            "case studies real-world scenarios strategic situations decision-making "
            "frameworks business strategy management law medicine policy cases strategic "
            "analysis frameworks decision scenarios strategic plans case analyses"
        ),
        "Language & Communication-Based Learning": (
            "language skills communication techniques writing speaking rhetoric expression "
            "literature languages journalism communication studies essays presentations "
            "writing exercises communication projects"
        ),
        "Hands-On, Project-Based Learning": (
            "hands-on projects building creating practical application learning by doing "
            "engineering design programming art music laboratory sciences projects "
            "practical exercises build systems project deliverables portfolios "
            "demonstrations"
        ),
        "Seminar / Discussion-Based Learning": (
            "seminar discussion debate Socratic dialogue roundtable reading group analysis "
            "close reading humanities social sciences political science philosophy "
            "weekly readings response papers participation discussion board collaborative "
            "conversation argument critique peer review"
        ),
    }

    def __init__(self):
        self._course_types = list(self.COURSE_TYPE_DESCRIPTIONS.keys())
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = list(self.COURSE_TYPE_DESCRIPTIONS.values())
        self._type_vectors = self._vectorizer.fit_transform(corpus)

    def classify(self, syllabus_text: str) -> List[Tuple[str, float]]:
        """Return (course_type, confidence) pairs sorted descending."""
        syllabus_vector = self._vectorizer.transform([syllabus_text])
        scores = {
            ct: float(cosine_similarity(self._type_vectors[i], syllabus_vector)[0][0])
            for i, ct in enumerate(self._course_types)
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @property
    def course_types(self) -> List[str]:
        return list(self._course_types)
