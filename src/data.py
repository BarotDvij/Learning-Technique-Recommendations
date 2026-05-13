"""Default data, example syllabi, and shared data utilities."""

from typing import Dict

import pandas as pd

REQUIRED_COLUMNS = ["Course Type", "Learning Technique", "Grade of Module (%)"]


DEFAULT_TECHNIQUE_GRADES: Dict[str, Dict[str, float]] = {
    "Applied Calculation-Driven Learning": {
        "Worked Example Analysis": 93.45,
        "Simulation & Visualization": 93.06,
        "Deliberate Practice": 91.63,
        "Reverse Engineering": 90.80,
        "Iterative Writing & Editing": 63.45,
        "Case Study Analysis": 61.15,
    },
    "Deep Conceptual Learning": {
        "Conceptual Mapping": 93.25,
        "Spaced Repetition": 92.46,
        "Feynman Technique": 92.42,
        "Active Recall": 92.33,
        "Project-Based Learning": 62.93,
        "Open-Ended Exploration": 60.41,
    },
    "Case-Based & Strategic Learning": {
        "Case Study Analysis": 93.97,
        "Experiential Learning": 92.98,
        "Comparative Analysis": 92.60,
        "First-Principles Thinking": 92.50,
        "Simulation & Visualization": 63.42,
        "Spaced Repetition": 60.42,
    },
    "Language & Communication-Based Learning": {
        "Immersive Practice": 93.14,
        "Iterative Writing & Editing": 92.70,
        "Storytelling Frameworks": 92.27,
        "Active Recall & Shadowing": 92.03,
        "Worked Example Analysis": 63.76,
        "Deliberate Practice": 61.82,
    },
    "Hands-On, Project-Based Learning": {
        "Incremental Skill Building": 93.67,
        "Project-Based Learning": 92.23,
        "Learn-By-Building": 91.65,
        "Work-Along & Solving": 91.50,
        "Spaced Repetition": 64.94,
        "Conceptual Mapping": 64.78,
    },
}


EXAMPLE_SYLLABI: Dict[str, str] = {
    "Calculus III": (
        "MATH 301: Calculus III\n"
        "Multivariate calculus: partial derivatives, multiple integrals, vector calculus. "
        "Students solve complex mathematical problems using equations, formulas, and "
        "computational techniques.\n"
        "Assessment: Problem sets (40%), Two midterms (30%), Final exam (30%)"
    ),
    "Introduction to Philosophy": (
        "PHIL 201: Introduction to Philosophy\n"
        "Major philosophical theories and frameworks explored through critical reading. "
        "Topics: epistemology, metaphysics, ethics, consciousness, abstract reasoning.\n"
        "Assessment: Reading responses (30%), Participation (20%), Essays (30%), Paper (20%)"
    ),
    "Strategic Management": (
        "BUS 405: Strategic Management\n"
        "Case study analysis, competitive strategy, and organizational decision-making. "
        "Students develop strategic plans and business recommendations from real-world "
        "scenarios.\n"
        "Assessment: Case analyses (40%), Strategic plan (30%), Final exam (20%), "
        "Participation (10%)"
    ),
    "Creative Writing Workshop": (
        "ENGL 220: Creative Writing Workshop\n"
        "Develop voice and craft through short stories, essays, and poetry. Focus on "
        "expression, rhetoric, narrative, and iterative revision through peer feedback "
        "and writing exercises.\n"
        "Assessment: Workshop submissions (50%), Final portfolio (30%), Peer reviews (20%)"
    ),
    "Full-Stack Web Development": (
        "CS 310: Full-Stack Web Development\n"
        "Hands-on project-based course: HTML, CSS, JavaScript, and backend frameworks. "
        "Students build and ship real applications through incremental project "
        "deliverables.\n"
        "Assessment: Projects (60%), Labs (20%), Participation (20%)"
    ),
}


def compute_technique_grades(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute average grade per (course type, technique) from a DataFrame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return {
        course_type: (
            group.groupby("Learning Technique")["Grade of Module (%)"]
            .mean()
            .round(2)
            .to_dict()
        )
        for course_type, group in data.groupby("Course Type")
    }
