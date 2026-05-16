"""Default data, example syllabi, and shared data utilities.

The default per-course-type technique scores are derived from the peer-reviewed
evidence base in :mod:`src.research` rather than synthetic grade data. See that
module's docstring for the methodology and citations.
"""

from typing import Dict

import pandas as pd

from src.research import compute_evidence_scores

REQUIRED_COLUMNS = ["Course Type", "Learning Technique", "Grade of Module (%)"]


DEFAULT_TECHNIQUE_GRADES: Dict[str, Dict[str, float]] = compute_evidence_scores()


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
    "Political Theory Seminar": (
        "POLS 450: Political Theory Seminar\n"
        "Discussion-intensive seminar on classical and contemporary political thought. "
        "Students read primary texts (Plato, Hobbes, Rawls, Arendt), present position "
        "papers, and engage in structured Socratic debate. Emphasis on argumentation, "
        "comparative textual analysis, and defending first-principles reasoning.\n"
        "Assessment: Weekly response papers (30%), Seminar leadership (20%), "
        "Comparative essay (30%), Final oral defence (20%)"
    ),
}


def compute_technique_grades(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute average grade per (course type, technique) from a DataFrame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return {
        course_type: (
            group.groupby("Learning Technique")["Grade of Module (%)"].mean().round(2).to_dict()
        )
        for course_type, group in data.groupby("Course Type")
    }
