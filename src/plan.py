"""
Study plan generation — turns a recommended technique + syllabus into a
concrete, session-by-session schedule.

Architecture:
    * Topic extraction      → optional LLM (Gemini Flash) with regex fallback
    * Session scheduling    → deterministic, per-technique templates (pure code)
    * Outputs               → in-memory StudyPlan, plus Markdown and ICS exports

The LLM is used **only** for topic extraction, which is the one step that
must be generated fresh per syllabus. Everything else is store-able logic
that runs without any API calls.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from typing import Callable, Dict, List, Optional

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
class StudySession:
    """One concrete study session in a plan."""

    week: int
    session_in_week: int
    duration_minutes: int
    technique: str
    topic: str
    actions: List[str] = field(default_factory=list)
    is_review: bool = False


@dataclass
class StudyPlan:
    """A complete N-week study plan."""

    course_title: str
    course_type: str
    primary_technique: str
    review_technique: Optional[str]
    weeks: int
    hours_per_week: float
    topics: List[str]
    sessions: List[StudySession]
    notes: List[str] = field(default_factory=list)
    topic_source: str = "fallback"  # "llm" or "fallback"

    # ── derived helpers ───────────────────────────────────────────────────────

    @property
    def total_sessions(self) -> int:
        return len(self.sessions)

    @property
    def total_minutes(self) -> int:
        return sum(s.duration_minutes for s in self.sessions)

    def sessions_in_week(self, week: int) -> List[StudySession]:
        return [s for s in self.sessions if s.week == week]

    # ── exporters ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.course_title} — Study Plan",
            "",
            f"**Course type:** {self.course_type}  ",
            f"**Primary technique:** {self.primary_technique}  ",
        ]
        if self.review_technique:
            lines.append(f"**Review technique:** {self.review_technique}  ")
        lines.extend(
            [
                f"**Duration:** {self.weeks} weeks · {self.hours_per_week:g} hrs/week  ",
                f"**Total sessions:** {self.total_sessions} "
                f"({self.total_minutes // 60}h {self.total_minutes % 60}m total)",
                "",
                "## Topics covered",
            ]
        )
        for t in self.topics:
            lines.append(f"- {t}")
        lines.append("")

        if self.notes:
            lines.append("## Notes")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        for week in range(1, self.weeks + 1):
            week_sessions = self.sessions_in_week(week)
            if not week_sessions:
                continue
            lines.append(f"## Week {week}")
            lines.append("")
            for s in week_sessions:
                tag = " *(review)*" if s.is_review else ""
                lines.append(f"### Session {s.session_in_week} — {s.topic}{tag}")
                lines.append(f"*{s.technique} · {s.duration_minutes} min*")
                lines.append("")
                for a in s.actions:
                    lines.append(f"- {a}")
                lines.append("")

        return "\n".join(lines)

    def to_ics(self, start_date: date, session_hour: int = 18) -> str:
        """Render plan as an iCalendar (.ics) file string.

        Sessions are scheduled one per day starting from ``start_date``, in the
        order they appear in the plan. ``session_hour`` is the local hour each
        session starts (24h clock).
        """

        def fmt(dt: datetime) -> str:
            return dt.strftime("%Y%m%dT%H%M%S")

        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Learning Technique Recommender//Study Plan//EN",
            "CALSCALE:GREGORIAN",
        ]
        cursor = datetime(start_date.year, start_date.month, start_date.day, session_hour, 0)
        for s in self.sessions:
            start = cursor
            end = start + timedelta(minutes=s.duration_minutes)
            description = "\\n".join(s.actions).replace(",", "\\,")
            lines.extend(
                [
                    "BEGIN:VEVENT",
                    f"UID:{uuid.uuid4()}@learning-technique-recommender",
                    f"DTSTAMP:{fmt(datetime.utcnow())}Z",
                    f"DTSTART:{fmt(start)}",
                    f"DTEND:{fmt(end)}",
                    f"SUMMARY:{s.topic} ({s.technique})",
                    f"DESCRIPTION:{description}",
                    "END:VEVENT",
                ]
            )
            cursor = start + timedelta(days=1)
        lines.append("END:VCALENDAR")
        return "\r\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Topic extraction (LLM-optional)
# ──────────────────────────────────────────────────────────────────────────────


def _get_genai_client(api_key: Optional[str]):
    """Return a configured Gemini client, or None if no usable key is present."""
    if not GENAI_AVAILABLE:
        return None
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


_MAX_SYLLABUS_CHARS = 4000  # Match quiz.py; prevents prompt-stuffing and runaway costs


def _extract_topics_llm(client, syllabus_text: str, n_topics: int) -> List[str]:
    """Use Gemini Flash to extract clean topic strings from raw syllabus text."""
    safe_syllabus = syllabus_text.strip()[:_MAX_SYLLABUS_CHARS]
    prompt = (
        f"You are a study planner. Read the course syllabus below and extract "
        f"the {n_topics} most important study topics a student must master.\n\n"
        f"Each topic must be:\n"
        f"  - 3-7 words long\n"
        f"  - specific and actionable\n"
        f"  - a discrete unit of content (not too broad)\n\n"
        f"Return ONLY a JSON array of strings. No numbering, no commentary.\n\n"
        f"<syllabus>\n{safe_syllabus}\n</syllabus>"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    data = json.loads(response.text)
    if not isinstance(data, list):
        raise ValueError("LLM response was not a JSON list")
    return [str(t).strip() for t in data if str(t).strip()][:n_topics]


_TOPIC_PATTERNS = [
    re.compile(r"^\s*\d+[.)]\s+(.+)$", re.MULTILINE),
    re.compile(r"^\s*[-•*]\s+(.+)$", re.MULTILINE),
]


def _extract_topics_fallback(syllabus_text: str, n_topics: int) -> List[str]:
    """Heuristic topic extraction without an LLM."""
    candidates: List[str] = []
    for pattern in _TOPIC_PATTERNS:
        for m in pattern.finditer(syllabus_text):
            candidates.append(m.group(1).strip())

    if not candidates:
        for raw in syllabus_text.split("\n"):
            line = raw.strip()
            if 18 <= len(line) <= 160 and ":" not in line[:14]:
                candidates.append(line)

    seen = set()
    unique: List[str] = []
    for c in candidates:
        cleaned = re.sub(r"\s+", " ", c).strip(" .;,:!?")
        if 6 <= len(cleaned) <= 90:
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                unique.append(cleaned)

    if not unique:
        unique = [f"Course unit {i + 1}" for i in range(n_topics)]
    return unique[:n_topics]


def extract_topics(
    syllabus_text: str,
    n_topics: int = 6,
    api_key: Optional[str] = None,
) -> tuple[List[str], str]:
    """Extract topics. Returns ``(topics, source)`` where source ∈ {'llm', 'fallback'}."""
    client = _get_genai_client(api_key)
    if client is not None:
        try:
            topics = _extract_topics_llm(client, syllabus_text, n_topics)
            if topics:
                return topics, "llm"
        except Exception as exc:
            # Log the failure so callers can surface it; fall through to heuristic.
            import sys
            print(f"[plan] LLM topic extraction failed ({type(exc).__name__}: {exc}); using fallback.", file=sys.stderr)
    return _extract_topics_fallback(syllabus_text, n_topics), "fallback"


# ──────────────────────────────────────────────────────────────────────────────
# Per-technique session templates
# ──────────────────────────────────────────────────────────────────────────────


def _sessions_per_week(hours_per_week: float, target_minutes: int) -> tuple[int, int]:
    """Return (sessions_per_week, minutes_per_session) for a target session length."""
    total_min = max(30, int(hours_per_week * 60))
    sessions = max(2, min(7, round(total_min / target_minutes)))
    duration = max(15, total_min // sessions)
    return sessions, duration


def _template_worked_examples(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 50)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        for s in range(1, n + 1):
            topic = topics[((w - 1) * n + (s - 1)) % len(topics)]
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Worked Example Analysis",
                    topic=topic,
                    actions=[
                        f"Study 2 fully worked solutions involving {topic} (15 min)",
                        "Annotate each step with the underlying principle (10 min)",
                        "Attempt 1 similar problem from scratch without looking (15 min)",
                        "Compare your solution to the model, note any drift (10 min)",
                    ],
                )
            )
    return sessions


def _template_feynman(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 55)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        for s in range(1, n + 1):
            topic = topics[((w - 1) * n + (s - 1)) % len(topics)]
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Feynman Technique",
                    topic=topic,
                    actions=[
                        f"Pick 1 sub-concept from {topic} (5 min)",
                        "Write a plain-language explanation as if teaching a 12-year-old (15 min)",
                        "Identify every place you reached for jargon — those are gaps (5 min)",
                        "Re-study the gaps in your source material (15 min)",
                        "Rewrite the explanation from scratch, simpler than before (15 min)",
                    ],
                )
            )
    return sessions


def _template_concept_mapping(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 45)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        for s in range(1, n + 1):
            topic = topics[((w - 1) * n + (s - 1)) % len(topics)]
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Conceptual Mapping",
                    topic=topic,
                    actions=[
                        f"List every concept tied to {topic} (10 min)",
                        "Draw a concept map: nodes for concepts, labeled arrows for relations (20 min)",
                        "Mark every relation you can't articulate clearly — study those (10 min)",
                        "Connect this week's map to last week's master map (5 min)",
                    ],
                )
            )
    return sessions


def _template_case_study(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 75)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        for s in range(1, n + 1):
            topic = topics[((w - 1) * n + (s - 1)) % len(topics)]
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Case Study Analysis",
                    topic=topic,
                    actions=[
                        f"Find or re-read 1 case relevant to {topic} (15 min)",
                        "Map the situation, stakeholders, and decision points (15 min)",
                        "Identify which framework or principle applies (15 min)",
                        "Write your recommendation with evidence and tradeoffs (20 min)",
                        "Compare your reasoning to the published outcome (10 min)",
                    ],
                )
            )
    return sessions


def _template_immersive(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 40)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        for s in range(1, n + 1):
            topic = topics[((w - 1) * n + (s - 1)) % len(topics)]
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Immersive Practice",
                    topic=topic,
                    actions=[
                        f"Listen to or read native content centered on {topic} (15 min)",
                        "Speak or write a short response without consulting references (10 min)",
                        "Compare to native usage and note 3 corrections (10 min)",
                        "Re-attempt the same response with corrections internalized (5 min)",
                    ],
                )
            )
    return sessions


def _template_incremental_skill(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 60)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        for s in range(1, n + 1):
            topic = topics[((w - 1) * n + (s - 1)) % len(topics)]
            actions = [
                f"Recall what you know about {topic} (5 min)",
                "Complete the next scaffolded exercise at your edge — not too easy, not too hard (30 min)",
                "Reflect: what just clicked, and what's still fuzzy? Write 3 lines (5 min)",
                "Plan the smallest next step you'd take tomorrow (5 min)",
            ]
            if dur >= 60:
                actions.insert(
                    2, "Stretch goal: attempt one variation of the exercise unguided (15 min)"
                )
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Incremental Skill Building",
                    topic=topic,
                    actions=actions,
                )
            )
    return sessions


def _template_project_based(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 75)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        milestone = topics[(w - 1) % len(topics)]
        for s in range(1, n + 1):
            actions = [
                f"Open the project working toward this week's milestone: {milestone} (5 min)",
                "Build the next slice end-to-end — vertical, not horizontal (45 min)",
                "Test it. If it doesn't run, debug now, not later (15 min)",
                "Commit and write a one-line note on what's still incomplete (10 min)",
            ]
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Project-Based Learning",
                    topic=milestone,
                    actions=actions,
                )
            )
    return sessions


def _template_active_recall(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 35)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        for s in range(1, n + 1):
            topic = topics[((w - 1) * n + (s - 1)) % len(topics)]
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Active Recall",
                    topic=topic,
                    actions=[
                        f"Read through your notes on {topic} once (5 min)",
                        "Close notes. Write down everything you remember (15 min)",
                        "Re-open notes. Mark every gap in red (5 min)",
                        "Re-attempt recall on just the gaps (10 min)",
                    ],
                )
            )
    return sessions


def _template_spaced_repetition(topics, weeks, hours_per_week):
    """Reviews at expanding intervals across the plan."""
    sessions: List[StudySession] = []
    intervals = [1, 3, 7, 14]
    base_dur = max(15, int((hours_per_week * 60) / 4))
    for i, topic in enumerate(topics):
        for offset in intervals:
            day = (i + 1) + offset
            week = ((day - 1) // 7) + 1
            if week > weeks:
                break
            sin_week = ((day - 1) % 7) + 1
            sessions.append(
                StudySession(
                    week=week,
                    session_in_week=sin_week,
                    duration_minutes=base_dur,
                    technique="Spaced Repetition",
                    topic=topic,
                    actions=[
                        f"Recall everything you know about {topic} from memory (10 min)",
                        "Check against notes; mark only the missed items for next review (5 min)",
                        f"Estimate next review interval (typically {offset}x current spacing)",
                    ],
                    is_review=True,
                )
            )
    return sessions


def _template_default(topics, weeks, hours_per_week):
    n, dur = _sessions_per_week(hours_per_week, 45)
    sessions: List[StudySession] = []
    for w in range(1, weeks + 1):
        for s in range(1, n + 1):
            topic = topics[((w - 1) * n + (s - 1)) % len(topics)]
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=s,
                    duration_minutes=dur,
                    technique="Focused Study Block",
                    topic=topic,
                    actions=[
                        f"Preview {topic}: skim source material to build a roadmap (5 min)",
                        "Deep work block on the hardest part — phones off, single window (30 min)",
                        "Self-test on 2 questions you couldn't have answered an hour ago (10 min)",
                    ],
                )
            )
    return sessions


TEMPLATES: Dict[str, Callable[[List[str], int, float], List[StudySession]]] = {
    "Worked Example Analysis": _template_worked_examples,
    "Feynman Technique": _template_feynman,
    "Conceptual Mapping": _template_concept_mapping,
    "Case Study Analysis": _template_case_study,
    "Immersive Practice": _template_immersive,
    "Incremental Skill Building": _template_incremental_skill,
    "Project-Based Learning": _template_project_based,
    "Active Recall": _template_active_recall,
    "Spaced Repetition": _template_spaced_repetition,
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


REVIEW_EXEMPT_TECHNIQUES = {"Active Recall", "Spaced Repetition"}


def generate_study_plan(
    course_title: str,
    course_type: str,
    primary_technique: str,
    syllabus_text: str,
    weeks: int = 4,
    hours_per_week: float = 5.0,
    n_topics: int = 6,
    api_key: Optional[str] = None,
    add_spaced_review: bool = True,
) -> StudyPlan:
    """Build a complete study plan for a course.

    The primary technique drives the session structure; if ``add_spaced_review``
    is True and the primary technique is not itself a review technique, short
    Spaced Repetition review sessions are added at the end of each week.
    """
    if weeks < 1 or hours_per_week <= 0:
        raise ValueError("weeks must be >= 1 and hours_per_week must be > 0")

    topics, topic_source = extract_topics(syllabus_text, n_topics=n_topics, api_key=api_key)
    template = TEMPLATES.get(primary_technique, _template_default)
    sessions = template(topics, weeks, hours_per_week)

    review_tech: Optional[str] = None
    if add_spaced_review and primary_technique not in REVIEW_EXEMPT_TECHNIQUES:
        review_tech = "Spaced Repetition"
        next_session_idx = max((s.session_in_week for s in sessions if s.week == 1), default=0) + 1
        for w in range(1, weeks + 1):
            sessions.append(
                StudySession(
                    week=w,
                    session_in_week=next_session_idx,
                    duration_minutes=20,
                    technique="Spaced Repetition",
                    topic=f"Week {w} consolidation",
                    actions=[
                        "Recall the week's topics from memory, one minute each (10 min)",
                        "Re-test only the items you missed last time (10 min)",
                    ],
                    is_review=True,
                )
            )

    sessions.sort(key=lambda s: (s.week, s.is_review, s.session_in_week))

    notes: List[str] = []
    if topic_source == "fallback":
        notes.append(
            "Topics extracted with heuristic parsing — set a GEMINI_API_KEY for "
            "richer LLM-based extraction."
        )

    return StudyPlan(
        course_title=course_title,
        course_type=course_type,
        primary_technique=primary_technique,
        review_technique=review_tech,
        weeks=weeks,
        hours_per_week=hours_per_week,
        topics=topics,
        sessions=sessions,
        notes=notes,
        topic_source=topic_source,
    )
