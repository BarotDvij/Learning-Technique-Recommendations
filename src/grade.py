"""
Semantic grading of a student's free-text answer against a model answer.

`grade_answer()` returns a structured `Grade` with:
    * a 0-5 numeric score
    * a boolean ``is_correct`` (score >= 4)
    * 2-3 sentences of constructive feedback
    * up to 3 specific concepts the student missed
    * a ``source`` field indicating whether the LLM was used

Without an API key, falls back to a simple keyword-overlap heuristic so the
practice loop stays functional offline (with a clear note in the feedback).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from typing import List, Optional

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
class Grade:
    """Result of grading a single answer."""

    score: int  # 0 (no credit) → 5 (perfect)
    is_correct: bool
    feedback: str
    missing_points: List[str] = field(default_factory=list)
    source: str = "fallback"  # 'llm' | 'fallback'

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# LLM-backed grading
# ──────────────────────────────────────────────────────────────────────────────


_GRADE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "INTEGER"},
        "feedback": {"type": "STRING"},
        "missing_points": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["score", "feedback"],
}


def _get_client(api_key: Optional[str]):
    if not GENAI_AVAILABLE:
        return None
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def _grade_with_llm(
    question: str,
    expected_answer: str,
    user_answer: str,
    technique: str,
    client,
) -> Optional[Grade]:
    """Single Gemini call; returns None on any failure so the caller can fall back."""
    prompt = (
        "You are a precise, fair tutor grading a student's free-text answer.\n\n"
        f"Question: {question}\n"
        f"Model answer: {expected_answer}\n"
        f"Study technique driving the question: {technique}\n"
        f"Student's answer: {user_answer}\n\n"
        "Grade the student on a 0-5 rubric:\n"
        "  0 = blank / off-topic / no credit\n"
        "  1 = wrong but shows minimal engagement\n"
        "  2 = partial — one key idea touched but mostly missing\n"
        "  3 = mostly correct — covers the gist but with notable omissions\n"
        "  4 = correct — captures the key idea, minor gaps OK\n"
        "  5 = excellent — complete, accurate, well-phrased\n\n"
        "Return JSON with:\n"
        "  - 'score': integer 0-5\n"
        "  - 'feedback': 2-3 sentences. Be specific. Acknowledge what's right;\n"
        "    point out what's missing or incorrect. Do not exceed 3 sentences.\n"
        "  - 'missing_points': up to 3 concise bullet points (each <= 12 words)\n"
        "    naming concepts the student missed. Empty list if nothing missing.\n"
        "Return ONLY the JSON object."
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_GRADE_SCHEMA,
                temperature=0.2,
            ),
        )
        data = json.loads(response.text or "{}")
        score = int(data.get("score", 0))
        score = max(0, min(5, score))
        feedback = str(data.get("feedback", "")).strip()
        missing = [str(p).strip() for p in (data.get("missing_points") or []) if str(p).strip()][:3]
        if not feedback:
            return None
        return Grade(
            score=score,
            is_correct=score >= 4,
            feedback=feedback,
            missing_points=missing,
            source="llm",
        )
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Offline fallback — keyword overlap
# ──────────────────────────────────────────────────────────────────────────────


_STOPWORDS = frozenset(
    """
    a an and the of to in on for is are was were be been being it this that those these
    with as by at from or but if then so than which who whom whose where when why how
    do does did has have had can could should would may might will shall not no your you
    i we they he she them us our its their my me his her hers ours yours about into over
    under up down out off again further once also there here only just some any all each
    every more most other another own same such too very most own
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def _keywords(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD_RE.findall(text or "")
        if len(w) >= 3 and w.lower() not in _STOPWORDS
    }


def _grade_with_fallback(
    expected_answer: str,
    user_answer: str,
) -> Grade:
    """Heuristic grading via keyword Jaccard overlap. Conservative and transparent."""
    expected_kws = _keywords(expected_answer)
    user_kws = _keywords(user_answer)

    if not user_answer.strip():
        return Grade(
            score=0,
            is_correct=False,
            feedback="No answer provided. Try drafting a response before checking.",
            missing_points=[],
            source="fallback",
        )

    if not expected_kws:
        return Grade(
            score=3,
            is_correct=False,
            feedback=(
                "Offline grading is keyword-based, and the model answer didn't yield "
                "enough keywords to score reliably. Compare your answer to the model "
                "answer manually."
            ),
            missing_points=[],
            source="fallback",
        )

    overlap = expected_kws & user_kws
    coverage = len(overlap) / len(expected_kws)

    if coverage >= 0.75:
        score = 5
    elif coverage >= 0.55:
        score = 4
    elif coverage >= 0.35:
        score = 3
    elif coverage >= 0.15:
        score = 2
    elif coverage > 0.0:
        score = 1
    else:
        score = 0

    missing = sorted(expected_kws - user_kws)[:3]
    feedback = (
        f"Offline keyword grading: your answer covers {coverage * 100:.0f}% of the "
        f"key concepts in the model answer. "
        + (
            "Set GEMINI_API_KEY to enable AI semantic grading for nuanced feedback."
            if coverage < 1.0
            else "Full keyword coverage."
        )
    )
    return Grade(
        score=score,
        is_correct=score >= 4,
        feedback=feedback,
        missing_points=missing,
        source="fallback",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def grade_answer(
    question: str,
    expected_answer: str,
    user_answer: str,
    technique: str = "",
    api_key: Optional[str] = None,
) -> Grade:
    """
    Grade a student's free-text answer.

    Tries Gemini first when an API key is configured; falls back to deterministic
    keyword-overlap scoring otherwise. The fallback is conservative and transparent
    — the feedback always explains how the score was derived.

    Args:
        question:        The prompt the student answered.
        expected_answer: The model / reference answer.
        user_answer:     The student's free-text response.
        technique:       Optional — the technique driving the question
                         (helps the LLM calibrate strictness).
        api_key:         Optional Gemini API key. Falls back to env vars.

    Returns:
        Grade dataclass with score, is_correct, feedback, missing_points, source.
    """
    client = _get_client(api_key)
    if client is not None:
        llm_grade = _grade_with_llm(
            question=question,
            expected_answer=expected_answer,
            user_answer=user_answer,
            technique=technique,
            client=client,
        )
        if llm_grade is not None:
            return llm_grade

    return _grade_with_fallback(expected_answer, user_answer)
