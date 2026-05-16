"""
Per-technique practice quiz generation.

The quiz format is tuned to the learning technique driving the study session:

    Active Recall              → factual / definitional recall
    Worked Example Analysis    → solve-and-explain with multi-step reasoning
    Feynman Technique          → explain-it-simply prompts
    Conceptual Mapping         → identify-the-relationships
    Case Study Analysis        → scenario reasoning
    Immersive Practice         → fill-in / translate / respond-naturally
    Incremental Skill Building → graduated-difficulty drills
    Project-Based Learning     → design / debugging prompts
    Spaced Repetition          → short-form recall reviews
    (default)                  → mixed factual + applied

Generation is via Gemini 2.5 Flash with a JSON schema. If no API key is
present, returns a minimal placeholder quiz with prompts the student can
fill in themselves — the app stays useful offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class QuizQuestion:
    """A single quiz question."""

    question: str
    answer: str
    explanation: str = ""
    difficulty: str = "medium"  # "easy" | "medium" | "hard"


@dataclass
class Quiz:
    """A generated quiz for one (topic, technique) pair."""

    topic: str
    technique: str
    questions: List[QuizQuestion] = field(default_factory=list)
    source: str = "fallback"  # 'llm' | 'fallback'
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render the quiz as study-friendly Markdown with questions then answers."""
        lines: List[str] = [
            f"# Practice quiz: {self.topic}",
            "",
            f"**Technique:** {self.technique}",
            f"**Source:** {self.source}",
            "",
            "## Questions",
            "",
        ]
        for i, q in enumerate(self.questions, start=1):
            lines.append(f"**Q{i}** _(difficulty: {q.difficulty})_")
            lines.append("")
            lines.append(q.question)
            lines.append("")

        lines.extend(["---", "", "## Answer key", ""])
        for i, q in enumerate(self.questions, start=1):
            lines.append(f"**A{i}.** {q.answer}")
            if q.explanation:
                lines.append("")
                lines.append(f"_{q.explanation}_")
            lines.append("")

        if self.notes:
            lines.extend(["---", "", "## Notes", ""])
            for n in self.notes:
                lines.append(f"- {n}")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Technique-specific prompt builders
# ──────────────────────────────────────────────────────────────────────────────


TECHNIQUE_INSTRUCTIONS: Dict[str, str] = {
    "Active Recall": (
        "Generate factual recall questions that force the student to retrieve "
        "core definitions, formulas, dates, names, or principles from memory. "
        "Each answer should be 1-3 sentences and verifiable."
    ),
    "Worked Example Analysis": (
        "Generate problem-solving questions that require multi-step calculation "
        "or derivation. Each answer should walk through the full solution step-by-step "
        "in the 'explanation' field, with the final result in 'answer'."
    ),
    "Feynman Technique": (
        "Generate 'explain it like I'm 12' prompts. The student must rephrase a "
        "concept in plain language with no jargon. The 'answer' is a model plain-language "
        "explanation; the 'explanation' calls out what makes the explanation good."
    ),
    "Conceptual Mapping": (
        "Generate questions about how concepts relate — causes, hierarchies, "
        "differences, dependencies. Each answer names the relationship explicitly."
    ),
    "Case Study Analysis": (
        "Generate short scenario-based questions. Present a brief situation and ask "
        "the student to identify the relevant framework, recommended action, and "
        "primary risk. The answer should articulate all three."
    ),
    "Immersive Practice": (
        "Generate language-practice prompts: complete the sentence, translate, "
        "respond naturally to a scenario, or correct an error. Answer is the target form."
    ),
    "Incremental Skill Building": (
        "Generate drills that gradually increase in difficulty. Each question targets "
        "one micro-skill needed to master the topic, with a concrete deliverable."
    ),
    "Project-Based Learning": (
        "Generate design / debugging / extension prompts grounded in real building. "
        "Each question asks the student to plan, modify, or troubleshoot something concrete."
    ),
    "Spaced Repetition": (
        "Generate short recall checks — single-fact or single-concept questions "
        "that can be answered in under 30 seconds, suitable for review sessions."
    ),
}


DEFAULT_INSTRUCTION = (
    "Generate a balanced mix of factual recall and applied / scenario questions "
    "appropriate for testing mastery of the topic."
)


# ──────────────────────────────────────────────────────────────────────────────
# Gemini client
# ──────────────────────────────────────────────────────────────────────────────


def _get_genai_client(api_key: Optional[str]):
    if not GENAI_AVAILABLE:
        return None
    key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "question": {"type": "STRING"},
            "answer": {"type": "STRING"},
            "explanation": {"type": "STRING"},
            "difficulty": {
                "type": "STRING",
                "enum": ["easy", "medium", "hard"],
            },
        },
        "required": ["question", "answer"],
    },
}


def _fallback_quiz(topic: str, technique: str, n_questions: int) -> Quiz:
    """Generate a minimal placeholder quiz when no LLM is available."""
    instruction = TECHNIQUE_INSTRUCTIONS.get(technique, DEFAULT_INSTRUCTION)
    questions = [
        QuizQuestion(
            question=(
                f"[{technique}] Write your own question #{i + 1} for {topic}. "
                f"Style: {instruction.split('.')[0].lower()}."
            ),
            answer="(Answer this yourself, then check against your source material.)",
            explanation="",
            difficulty="medium",
        )
        for i in range(n_questions)
    ]
    return Quiz(
        topic=topic,
        technique=technique,
        questions=questions,
        source="fallback",
        notes=[
            "Set a GEMINI_API_KEY in .streamlit/secrets.toml for AI-generated quizzes.",
        ],
    )


def generate_quiz(
    topic: str,
    technique: str,
    syllabus_text: str = "",
    n_questions: int = 5,
    api_key: Optional[str] = None,
    difficulty: str = "balanced",
) -> Quiz:
    """
    Generate a practice quiz for one (topic, technique) pair.

    Args:
        topic:         The specific topic, typically pulled from a StudyPlan session.
        technique:     The study technique driving the quiz format.
        syllabus_text: Optional broader context to keep questions on-topic.
        n_questions:   How many questions to generate (3-10 is the sweet spot).
        api_key:       Optional Gemini API key. Falls back to env vars.
        difficulty:    "easy", "medium", "hard", or "balanced" for a mix.

    Returns:
        Quiz with ``source='llm'`` on success, ``source='fallback'`` otherwise.
    """
    n_questions = max(1, min(10, int(n_questions)))
    client = _get_genai_client(api_key)
    if client is None:
        return _fallback_quiz(topic, technique, n_questions)

    instruction = TECHNIQUE_INSTRUCTIONS.get(technique, DEFAULT_INSTRUCTION)

    context_block = ""
    if syllabus_text.strip():
        truncated = syllabus_text.strip()[:4000]
        context_block = f"\n\nCourse context (for grounding only):\n{truncated}\n"

    if difficulty == "balanced":
        difficulty_clause = (
            "Mix difficulties across the questions: roughly 30% easy, 40% medium, 30% hard."
        )
    else:
        difficulty_clause = f"All questions should be of '{difficulty}' difficulty."

    prompt = (
        f"You are an expert tutor designing a practice quiz.\n\n"
        f"Topic: {topic}\n"
        f"Study technique driving the quiz format: {technique}\n\n"
        f"Quiz design instructions:\n{instruction}\n\n"
        f"Generate exactly {n_questions} questions. {difficulty_clause}\n"
        f"For each question include:\n"
        f"  - 'question': the prompt the student sees\n"
        f"  - 'answer': the model answer (1-4 sentences)\n"
        f"  - 'explanation': a brief note on what the question is testing or why\n"
        f"    the answer is what it is (optional, 1-2 sentences)\n"
        f"  - 'difficulty': one of 'easy', 'medium', 'hard'\n"
        f"\nReturn ONLY a JSON array matching the schema. No commentary."
        f"{context_block}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.6,
            ),
        )
        data = json.loads(response.text or "[]")
        if not isinstance(data, list) or not data:
            raise ValueError("Empty or malformed LLM response")

        questions = [
            QuizQuestion(
                question=str(item.get("question", "")).strip(),
                answer=str(item.get("answer", "")).strip(),
                explanation=str(item.get("explanation", "")).strip(),
                difficulty=str(item.get("difficulty", "medium")).strip().lower(),
            )
            for item in data
            if item.get("question") and item.get("answer")
        ][:n_questions]

        if not questions:
            raise ValueError("All questions missing required fields")

        return Quiz(
            topic=topic,
            technique=technique,
            questions=questions,
            source="llm",
        )
    except Exception as exc:
        quiz = _fallback_quiz(topic, technique, n_questions)
        quiz.notes.append(f"LLM call failed: {type(exc).__name__}.")
        return quiz
