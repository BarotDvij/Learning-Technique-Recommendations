"""Learning Technique Recommender — core package."""

from src.classifier import SyllabusClassifier
from src.data import (
    DEFAULT_TECHNIQUE_GRADES,
    EXAMPLE_SYLLABI,
    REQUIRED_COLUMNS,
    compute_technique_grades,
)
from src.pipeline import LearningStyleSystem
from src.recommender import LearningTechniqueRecommender

__all__ = [
    "SyllabusClassifier",
    "LearningTechniqueRecommender",
    "LearningStyleSystem",
    "DEFAULT_TECHNIQUE_GRADES",
    "EXAMPLE_SYLLABI",
    "REQUIRED_COLUMNS",
    "compute_technique_grades",
]
