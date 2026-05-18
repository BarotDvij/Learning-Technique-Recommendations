"""
Practice analytics — track quiz attempts and surface review recommendations.

The Progress tab in ``app.py`` is powered entirely by this module. Every time
``grade_answer()`` returns a :class:`~src.grade.Grade`, the UI calls
:meth:`PracticeLog.record_grade` so we accumulate a per-session history of
(topic × technique × score) attempts. From that history we derive:

  * Summary stats (totals, accuracy, mean score)
  * A running-average score trend for sparkline-style line charts
  * Weak topics — topics with enough attempts to be confident in but a
    consistently low average score
  * Recommended review sessions — weak topics ranked by a combination of
    poor performance and recency, so the suggested next study block targets
    the highest-leverage gap

Everything here is pure-stdlib Python — no LLM calls, no network, no Plotly /
Streamlit imports — so the module is trivially unit-testable and the Progress
tab continues to work fully offline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

from src.grade import Grade


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PracticeRecord:
    """A single graded quiz attempt."""

    session_id: str
    topic: str
    technique: str
    question: str
    score: int
    is_correct: bool
    timestamp: str  # ISO-8601, UTC
    grade_source: str  # 'llm' | 'fallback'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PracticeRecord":
        return cls(
            session_id=str(data.get("session_id", "")),
            topic=str(data.get("topic", "")),
            technique=str(data.get("technique", "")),
            question=str(data.get("question", "")),
            score=int(data.get("score", 0)),
            is_correct=bool(data.get("is_correct", False)),
            timestamp=str(data.get("timestamp", "")),
            grade_source=str(data.get("grade_source", "fallback")),
        )


@dataclass
class PracticeLog:
    """Append-only log of graded practice attempts for the current session."""

    records: List[PracticeRecord] = field(default_factory=list)

    # ── Mutation ──────────────────────────────────────────────────────────

    def record_grade(
        self,
        session_id: str,
        topic: str,
        technique: str,
        question: str,
        grade: Grade,
        timestamp: Optional[str] = None,
    ) -> PracticeRecord:
        """Append a new :class:`PracticeRecord` derived from a ``Grade``.

        Args:
            session_id: Stable identifier for the study-plan session the
                question came from (e.g. ``"w1-s2-Active Recall"``).
            topic: The topic the question targets.
            technique: The study technique driving the question format.
            question: The literal question text (truncated downstream if needed).
            grade: The :class:`~src.grade.Grade` returned by ``grade_answer``.
            timestamp: Override timestamp for deterministic testing. Defaults
                to ``datetime.utcnow().isoformat()``.

        Returns:
            The :class:`PracticeRecord` that was appended.
        """
        record = PracticeRecord(
            session_id=session_id,
            topic=topic,
            technique=technique,
            question=question,
            score=int(grade.score),
            is_correct=bool(grade.is_correct),
            timestamp=timestamp or datetime.utcnow().isoformat(),
            grade_source=str(grade.source),
        )
        self.records.append(record)
        return record

    def clear(self) -> None:
        """Drop every record. Used by 'Clear log' in the UI."""
        self.records.clear()

    # ── Aggregations ──────────────────────────────────────────────────────

    def summary_stats(self) -> dict:
        """Headline metrics for the Progress tab's metric row."""
        total = len(self.records)
        if total == 0:
            return {
                "total_attempted": 0,
                "total_correct": 0,
                "accuracy_pct": 0.0,
                "avg_score": 0.0,
                "topics_seen": 0,
                "techniques_seen": 0,
            }

        total_correct = sum(1 for r in self.records if r.is_correct)
        avg_score = sum(r.score for r in self.records) / total
        return {
            "total_attempted": total,
            "total_correct": total_correct,
            "accuracy_pct": round((total_correct / total) * 100, 1),
            "avg_score": round(avg_score, 2),
            "topics_seen": len({r.topic for r in self.records if r.topic}),
            "techniques_seen": len({r.technique for r in self.records if r.technique}),
        }

    def score_trend(self, window: int = 5) -> List[tuple[int, float]]:
        """Running-average score over the last ``window`` attempts.

        Returns a list of ``(attempt_index, running_avg)`` tuples — 1-indexed
        on the attempt axis, so the result drops straight into Plotly.
        """
        if not self.records:
            return []
        window = max(1, int(window))
        trend: List[tuple[int, float]] = []
        for i in range(len(self.records)):
            start = max(0, i + 1 - window)
            slice_ = self.records[start : i + 1]
            avg = sum(r.score for r in slice_) / len(slice_)
            trend.append((i + 1, round(avg, 2)))
        return trend

    def technique_breakdown(self) -> List[dict]:
        """Per-technique stats — used for the 'avg score by technique' bar chart."""
        buckets: dict[str, List[PracticeRecord]] = defaultdict(list)
        for r in self.records:
            if r.technique:
                buckets[r.technique].append(r)
        out: List[dict] = []
        for technique, items in buckets.items():
            out.append(
                {
                    "technique": technique,
                    "attempts": len(items),
                    "avg_score": round(sum(r.score for r in items) / len(items), 2),
                    "accuracy_pct": round(
                        100 * sum(1 for r in items if r.is_correct) / len(items), 1
                    ),
                }
            )
        out.sort(key=lambda d: d["avg_score"])
        return out

    def topic_breakdown(self) -> List[dict]:
        """Per-topic stats. Used internally by :meth:`weak_topics`."""
        buckets: dict[str, List[PracticeRecord]] = defaultdict(list)
        for r in self.records:
            if r.topic:
                buckets[r.topic].append(r)
        out: List[dict] = []
        for topic, items in buckets.items():
            # Most-recent-first technique used while studying this topic.
            techniques = list({r.technique for r in items if r.technique})
            last = items[-1]
            out.append(
                {
                    "topic": topic,
                    "attempts": len(items),
                    "avg_score": round(sum(r.score for r in items) / len(items), 2),
                    "accuracy_pct": round(
                        100 * sum(1 for r in items if r.is_correct) / len(items), 1
                    ),
                    "last_score": last.score,
                    "last_timestamp": last.timestamp,
                    "last_technique": last.technique,
                    "techniques": sorted(techniques),
                }
            )
        return out

    def weak_topics(
        self,
        min_attempts: int = 2,
        max_avg: float = 3.0,
    ) -> List[dict]:
        """Topics worth re-studying.

        A topic qualifies as "weak" when it has been attempted at least
        ``min_attempts`` times *and* its average score is at most ``max_avg``.

        The result is sorted by lowest average score first, then by attempt
        count (more attempts → higher confidence the gap is real).
        """
        weak = [
            t
            for t in self.topic_breakdown()
            if t["attempts"] >= min_attempts and t["avg_score"] <= max_avg
        ]
        weak.sort(key=lambda t: (t["avg_score"], -t["attempts"]))
        return weak

    def recommended_review(self, top_k: int = 3) -> List[dict]:
        """Top review suggestions, ranked by score gap with a recency tiebreak.

        Combines ``weak_topics()`` with the position of the topic's last
        attempt in the log — more recently practiced weak topics rank higher,
        since the student is most likely to retain the surrounding context.
        Falls back to "just below mastery" topics when no weak topic clears
        the bar, so the panel never sits empty when records exist.
        """
        top_k = max(1, int(top_k))
        order = {id(r): idx for idx, r in enumerate(self.records)}

        def recency(topic: str) -> int:
            return max(
                (order[id(r)] for r in self.records if r.topic == topic),
                default=-1,
            )

        candidates = self.weak_topics(min_attempts=2, max_avg=3.0)
        used_topics = {c["topic"] for c in candidates}
        if len(candidates) < top_k:
            # Pad with "below mastery" (avg_score < 4) topics not already chosen.
            extras = [
                t
                for t in self.topic_breakdown()
                if t["avg_score"] < 4.0 and t["topic"] not in used_topics
            ]
            extras.sort(key=lambda t: t["avg_score"])
            candidates.extend(extras)

        suggestions: List[dict] = []
        for t in candidates[: top_k * 2]:
            avg = t["avg_score"]
            if avg <= 2.0:
                reason = "Low scores across multiple attempts — re-study the fundamentals."
            elif avg <= 3.0:
                reason = "Partial understanding — revisit the key concepts and try again."
            else:
                reason = "Just below mastery — a short refresh should push this over the line."
            suggestions.append(
                {
                    "topic": t["topic"],
                    "technique": t["last_technique"]
                    or (t["techniques"][0] if t["techniques"] else ""),
                    "reason": reason,
                    "last_score": t["last_score"],
                    "avg_score": t["avg_score"],
                    "attempts": t["attempts"],
                    "recency_rank": recency(t["topic"]),
                }
            )

        suggestions.sort(key=lambda s: (s["avg_score"], -s["recency_rank"]))
        return suggestions[:top_k]

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the log."""
        return {
            "version": 1,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PracticeLog":
        """Round-trip with :meth:`to_dict`. Tolerates missing/extra keys."""
        raw = (data or {}).get("records", []) or []
        records = [PracticeRecord.from_dict(r) for r in raw]
        return cls(records=records)

    # ── Convenience ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)
