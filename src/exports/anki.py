"""Export quizzes and study plans to Anki .apkg format."""

from __future__ import annotations

import io
import os
import tempfile
from typing import TYPE_CHECKING

import genanki

if TYPE_CHECKING:
    from src.plan import StudyPlan
    from src.quiz import Quiz, QuizQuestion


def _get_or_create_model() -> genanki.Model:
    """Create a Basic Anki model (Front/Back)."""
    return genanki.Model(
        model_id=1607392319,  # Stable ID for the Basic model
        name="Basic",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}<hr id=answer>{{Back}}",
            }
        ],
    )


def quiz_to_anki(quiz: Quiz, deck_name: str | None = None) -> bytes:
    """Convert a Quiz to an Anki .apkg file (bytes).

    Each quiz question becomes a Basic card with:
    - Front: the question text
    - Back: the answer with optional explanation

    Args:
        quiz: A Quiz object with questions to convert
        deck_name: Optional custom deck name (defaults to quiz topic)

    Returns:
        Bytes of the .apkg file
    """
    if not quiz.questions:
        raise ValueError("Quiz has no questions to export")

    # Use quiz topic as deck name if not provided
    if deck_name is None:
        deck_name = quiz.topic or "Study Deck"

    # Create deck with stable ID derived from deck name
    deck_id = hash(deck_name) & 0x7FFFFFFF or 1
    deck = genanki.Deck(deck_id, deck_name)

    model = _get_or_create_model()

    for i, question in enumerate(quiz.questions, start=1):
        back = question.answer
        if question.explanation:
            back = f"{back}\n\n<em>Explanation: {question.explanation}</em>"

        note = genanki.Note(
            model=model,
            fields=[question.question, back],
            tags=["quiz", quiz.technique.lower().replace(" ", "-")],
        )
        deck.add_note(note)

    # Write to temporary file, read bytes, return
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        with open(tmp.name, "rb") as f:
            apkg_bytes = f.read()
        os.unlink(tmp.name)

    return apkg_bytes


def study_plan_to_anki(plan: StudyPlan, deck_name: str | None = None) -> bytes:
    """Convert a StudyPlan's topics to an Anki .apkg file (bytes).

    Creates a single deck with each topic as a note card:
    - Front: the topic name
    - Back: primary technique + review technique (if any)

    Args:
        plan: A StudyPlan object
        deck_name: Optional custom deck name (defaults to course title)

    Returns:
        Bytes of the .apkg file
    """
    if not plan.topics:
        raise ValueError("Study plan has no topics to export")

    # Use course title as deck name if not provided
    if deck_name is None:
        deck_name = plan.course_title or "Study Plan Deck"

    # Create deck with stable ID derived from deck name
    deck_id = hash(deck_name) & 0x7FFFFFFF or 1
    deck = genanki.Deck(deck_id, deck_name)

    model = _get_or_create_model()

    for topic in plan.topics:
        back_parts = [f"**Primary:** {plan.primary_technique}"]
        if plan.review_technique:
            back_parts.append(f"**Review:** {plan.review_technique}")
        back = "<br>".join(back_parts)

        note = genanki.Note(
            model=model,
            fields=[topic, back],
            tags=["study-plan", plan.course_type.lower().replace(" ", "-")],
        )
        deck.add_note(note)

    # Write to temporary file, read bytes, return
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp:
        genanki.Package(deck).write_to_file(tmp.name)
        with open(tmp.name, "rb") as f:
            apkg_bytes = f.read()
        os.unlink(tmp.name)

    return apkg_bytes
