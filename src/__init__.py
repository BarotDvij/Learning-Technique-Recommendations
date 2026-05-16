"""Learning Technique Recommender — core package."""

from src.analytics import PracticeLog, PracticeRecord
from src.classifier import SyllabusClassifier
from src.data import (
    DEFAULT_TECHNIQUE_GRADES,
    EXAMPLE_SYLLABI,
    REQUIRED_COLUMNS,
    compute_technique_grades,
)
from src.grade import Grade, grade_answer
from src.parsing import ExtractedSyllabus, extract_syllabus
from src.pipeline import LearningStyleSystem
from src.plan import StudyPlan, StudySession, generate_study_plan
from src.quiz import Quiz, QuizQuestion, generate_quiz
from src.recommender import LearningTechniqueRecommender
from src.research import (
    RESEARCH,
    TechniqueEvidence,
    compute_evidence_scores,
    get_evidence,
)

__all__ = [
    "SyllabusClassifier",
    "LearningTechniqueRecommender",
    "LearningStyleSystem",
    "DEFAULT_TECHNIQUE_GRADES",
    "EXAMPLE_SYLLABI",
    "REQUIRED_COLUMNS",
    "compute_technique_grades",
    "RESEARCH",
    "TechniqueEvidence",
    "compute_evidence_scores",
    "get_evidence",
    "StudyPlan",
    "StudySession",
    "generate_study_plan",
    "ExtractedSyllabus",
    "extract_syllabus",
    "Quiz",
    "QuizQuestion",
    "generate_quiz",
    "Grade",
    "grade_answer",
    "PracticeLog",
    "PracticeRecord",
]
