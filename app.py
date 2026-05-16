"""
Learning Technique Recommender — Streamlit App.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import (
    DEFAULT_TECHNIQUE_GRADES,
    EXAMPLE_SYLLABI,
    Grade,
    LearningStyleSystem,
    PracticeLog,
    REQUIRED_COLUMNS,
    RESEARCH,
    Quiz,
    StudyPlan,
    compute_technique_grades,
    extract_syllabus,
    generate_quiz,
    generate_study_plan,
    grade_answer,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Learning Technique Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    .hero-title { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.25rem; }
    .hero-subtitle { font-size: 1.05rem; color: #555; margin-bottom: 1.5rem; }
    .tech-card {
        background: #FFFFFF;
        border: 1px solid #E6E8EE;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .tech-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
    }
    .tech-card-top {
        background: linear-gradient(135deg, #EEF2FF 0%, #FFFFFF 60%);
        border: 1px solid #C7D2FE;
    }
    .tech-rank { font-size: 0.8rem; font-weight: 600; color: #6366F1; letter-spacing: 0.08em; text-transform: uppercase; }
    .tech-name { font-size: 1.2rem; font-weight: 600; color: #111827; margin: 0.1rem 0 0.35rem 0; }
    .tech-meta { font-size: 0.88rem; color: #6B7280; }
    .tech-score { font-size: 1.6rem; font-weight: 700; color: #4F46E5; }
    .tech-summary {
        font-size: 0.92rem;
        color: #4B5563;
        line-height: 1.5;
        margin-top: 0.7rem;
        padding-top: 0.7rem;
        border-top: 1px solid #F1F5F9;
    }
    .tech-citation {
        font-size: 0.8rem;
        color: #6366F1;
        margin-top: 0.35rem;
        font-style: italic;
    }
    .pill {
        display: inline-block;
        background: #F3F4F6;
        color: #374151;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 0.4rem;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-right: 0.4rem;
    }
    .badge-high          { background: #DCFCE7; color: #166534; }
    .badge-moderate-high { background: #DBEAFE; color: #1E40AF; }
    .badge-moderate      { background: #FEF3C7; color: #92400E; }
    .badge-mixed         { background: #FED7AA; color: #9A3412; }
    .badge-low           { background: #FEE2E2; color: #991B1B; }
    section[data-testid="stSidebar"] { background: #FAFBFD; }
    .stTextArea textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached system
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_resource
def get_default_system() -> LearningStyleSystem:
    """Build the default system once and reuse across reruns."""
    return LearningStyleSystem(DEFAULT_TECHNIQUE_GRADES)


def get_system() -> LearningStyleSystem:
    """Get the active system — custom data if uploaded, otherwise default."""
    grades = st.session_state.get("custom_grades")
    if grades:
        return LearningStyleSystem(grades, course_weight=st.session_state.get("course_weight", 0.5))
    sys = get_default_system()
    sys.course_weight = st.session_state.get("course_weight", 0.5)
    return sys


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────


def render_sidebar() -> str:
    """Render the sidebar and return the currently selected syllabus text."""
    with st.sidebar:
        st.markdown("### Examples")
        st.caption("Load a sample syllabus to try the system.")

        labels = list(EXAMPLE_SYLLABI.keys())
        chosen = st.radio("Example syllabus", labels, label_visibility="collapsed", key="example_pick")

        st.divider()
        st.markdown("### Settings")

        st.slider(
            "Course-type weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="How much the combined score weighs course-type match vs. raw technique grade.",
            key="course_weight",
        )

        st.divider()
        st.markdown("### Custom Data")
        st.caption(
            "Upload a CSV with columns: "
            f"`{'`, `'.join(REQUIRED_COLUMNS)}` to retrain on your own data."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                grades = compute_technique_grades(df)
                st.session_state["custom_grades"] = grades
                st.success(f"Loaded {len(df):,} rows · {len(grades)} course types")
            except Exception as exc:
                st.error(f"Could not load CSV: {exc}")

        if st.session_state.get("custom_grades"):
            if st.button("Reset to default data", use_container_width=True):
                st.session_state.pop("custom_grades", None)
                st.rerun()

        st.divider()
        st.markdown("### About")
        st.caption(
            "Recommends study techniques from raw syllabus text using TF-IDF "
            "classification + historical grade data."
        )
        st.caption("Built with scikit-learn · Streamlit · Plotly")

    return EXAMPLE_SYLLABI[chosen]


# ──────────────────────────────────────────────────────────────────────────────
# Result rendering
# ──────────────────────────────────────────────────────────────────────────────


def render_course_type_chart(course_type_scores: list[tuple[str, float]]) -> None:
    """Horizontal bar chart of course-type confidence scores."""
    df = pd.DataFrame(course_type_scores, columns=["Course Type", "Score"])
    df["Score"] = (df["Score"] * 100).round(1)
    df = df.sort_values("Score")

    fig = px.bar(
        df,
        x="Score",
        y="Course Type",
        orientation="h",
        text=df["Score"].apply(lambda v: f"{v:.1f}%"),
        color="Score",
        color_continuous_scale=["#E0E7FF", "#6366F1", "#4338CA"],
        range_color=[0, max(df["Score"].max(), 1)],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False,
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", range=[0, max(df["Score"].max() * 1.15, 5)])
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


def render_technique_cards(top_techniques: list[dict]) -> None:
    """Card-style display for top techniques, enriched with research evidence."""
    for i, t in enumerate(top_techniques, start=1):
        card_class = "tech-card tech-card-top" if i == 1 else "tech-card"
        rank_label = f"Rank {i}"

        research = t.get("research") or {}
        rating = research.get("evidence_rating", "moderate")
        badge_class = f"badge badge-{rating.replace(' ', '-')}"
        summary = research.get("summary", "")
        short_citation = research.get("short_citation", "")
        effect_size = research.get("effect_size")
        effect_metric = research.get("effect_metric", "")
        study_count = research.get("study_count")

        effect_pieces: list[str] = []
        if effect_size is not None and effect_metric:
            sign = "+" if effect_size >= 0 else ""
            effect_pieces.append(f"{effect_metric} {sign}{effect_size:.2f}")
        if study_count:
            effect_pieces.append(f"{study_count} studies")
        effect_pill = (
            f'<span class="pill">{" · ".join(effect_pieces)}</span>'
            if effect_pieces
            else ""
        )

        citation_html = (
            f'<div class="tech-citation">Source: {short_citation}</div>'
            if short_citation
            else ""
        )
        summary_html = (
            f'<div class="tech-summary">{summary}</div>' if summary else ""
        )

        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <div class="tech-rank">{rank_label}</div>
                        <div class="tech-name">{t['technique']}</div>
                        <div class="tech-meta">
                            <span class="{badge_class}">{rating} evidence</span>
                            <span class="pill">{t['course_type']}</span>
                            <span class="pill">match {t['course_match']}%</span>
                            <span class="pill">evidence {t['evidence_score']}</span>
                            {effect_pill}
                        </div>
                        {summary_html}
                        {citation_html}
                    </div>
                    <div style="text-align: right; padding-left: 1.5rem;">
                        <div class="tech-score">{t['combined_score']}</div>
                        <div class="tech-meta">combined</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detailed_table(all_techniques: list[dict]) -> None:
    """Sortable table of all techniques across all course types."""
    rows = []
    for t in all_techniques:
        r = t.get("research") or {}
        rows.append({
            "Technique": t["technique"],
            "Course Type": t["course_type"],
            "Course Match (%)": t["course_match"],
            "Evidence Score": t["evidence_score"],
            "Evidence Rating": r.get("evidence_rating", ""),
            "Source": r.get("short_citation", ""),
            "Combined Score": t["combined_score"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_data_explorer(system: LearningStyleSystem) -> None:
    """Browse the research-grounded technique scores underlying the recommender."""
    course_types = system.recommender.available_course_types
    selected = st.selectbox("Pick a course type to explore", course_types, key="explore_pick")
    techniques = system.recommender.get_all_techniques_ranked(selected)

    df = pd.DataFrame(techniques).rename(
        columns={"technique": "Technique", "expected_grade": "Evidence Score"}
    )
    df = df.sort_values("Evidence Score", ascending=True)

    fig = px.bar(
        df,
        x="Evidence Score",
        y="Technique",
        orientation="h",
        text=df["Evidence Score"].apply(lambda v: f"{v:.1f}"),
        color="Evidence Score",
        color_continuous_scale=[(0, "#FCA5A5"), (0.5, "#FCD34D"), (1, "#34D399")],
        range_color=[0, 100],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis_title=None,
        yaxis_title=None,
        coloraxis_showscale=False,
        plot_bgcolor="white",
    )
    fig.update_xaxes(range=[0, 110], showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Scores are computed as `evidence_weight × domain_alignment%`, where the "
        "evidence weight reflects the strength of meta-analytic support for the "
        "technique and the alignment captures fit to the course type's subfield."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Study plan tab
# ──────────────────────────────────────────────────────────────────────────────


def _get_gemini_key() -> str | None:
    """Pull a Gemini API key from Streamlit secrets if available."""
    try:
        return st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        return None


def render_plan_overview(plan: StudyPlan) -> None:
    """Top-of-tab summary metrics + topic list."""
    cols = st.columns(4)
    cols[0].metric("Weeks", plan.weeks)
    cols[1].metric("Sessions", plan.total_sessions)
    cols[2].metric("Total time", f"{plan.total_minutes // 60}h {plan.total_minutes % 60}m")
    cols[3].metric("Topics", len(plan.topics))

    with st.expander(f"Topics ({plan.topic_source})", expanded=False):
        for t in plan.topics:
            st.markdown(f"- {t}")
        if plan.notes:
            st.caption(" · ".join(plan.notes))


def _session_key(s) -> str:
    """Stable identifier for a single session within session_state."""
    return f"w{s.week}-s{s.session_in_week}-{s.technique}"


def render_quiz(quiz: Quiz, key_prefix: str = "") -> None:
    """Display a quiz as an expandable list of question → reveal-answer cards."""
    source_pill = (
        '<span class="badge badge-high">AI-generated</span>'
        if quiz.source == "llm"
        else '<span class="badge badge-mixed">offline template</span>'
    )
    st.markdown(
        f"<div style='margin: 0.5rem 0;'>{source_pill} "
        f"<span class='pill'>{len(quiz.questions)} questions</span> "
        f"<span class='pill'>{quiz.technique}</span></div>",
        unsafe_allow_html=True,
    )
    for note in quiz.notes:
        st.caption(note)

    grades: dict = st.session_state.setdefault("grades", {})
    practice_log: PracticeLog = st.session_state.setdefault("practice_log", PracticeLog())
    api_key = _get_gemini_key()

    for i, q in enumerate(quiz.questions, start=1):
        diff_cls = {
            "easy": "badge badge-high",
            "medium": "badge badge-moderate-high",
            "hard": "badge badge-mixed",
        }.get(q.difficulty.lower(), "badge badge-moderate")

        q_key = f"{key_prefix}-q{i}"

        with st.expander(f"Q{i}. {q.question}", expanded=False):
            st.markdown(
                f"<div style='margin-bottom: 0.6rem;'>"
                f"<span class='{diff_cls}'>{q.difficulty}</span></div>",
                unsafe_allow_html=True,
            )

            user_answer = st.text_area(
                "Your answer",
                key=f"{q_key}-input",
                height=90,
                placeholder="Write your answer, then check it against the model answer below.",
            )

            action_cols = st.columns([1, 1, 3])
            with action_cols[0]:
                if st.button(
                    "Check my answer",
                    key=f"{q_key}-check",
                    disabled=not user_answer.strip(),
                    use_container_width=True,
                ):
                    with st.spinner("Grading..."):
                        grade = grade_answer(
                            question=q.question,
                            expected_answer=q.answer,
                            user_answer=user_answer,
                            technique=quiz.technique,
                            api_key=api_key,
                        )
                    grades[q_key] = grade
                    practice_log.record_grade(
                        session_id=key_prefix or q_key,
                        topic=quiz.topic,
                        technique=quiz.technique,
                        question=q.question,
                        grade=grade,
                    )
            with action_cols[1]:
                if q_key in grades and st.button(
                    "Clear grade",
                    key=f"{q_key}-clear",
                    use_container_width=True,
                ):
                    grades.pop(q_key, None)
                    st.rerun()

            grade: Grade | None = grades.get(q_key)
            if grade is not None:
                grade_cls = {
                    5: "badge badge-high",
                    4: "badge badge-high",
                    3: "badge badge-moderate-high",
                    2: "badge badge-moderate",
                    1: "badge badge-mixed",
                    0: "badge badge-mixed",
                }.get(grade.score, "badge badge-moderate")
                source_label = "AI" if grade.source == "llm" else "offline"
                st.markdown(
                    f"<div style='margin: 0.6rem 0;'>"
                    f"<span class='{grade_cls}'>Score {grade.score}/5</span> "
                    f"<span class='pill'>{'correct' if grade.is_correct else 'review'}</span> "
                    f"<span class='pill'>{source_label} grading</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"_{grade.feedback}_")
                if grade.missing_points:
                    st.markdown("**Concepts to review:**")
                    for mp in grade.missing_points:
                        st.markdown(f"- {mp}")

            st.markdown("---")
            st.markdown(f"**Model answer:** {q.answer}")
            if q.explanation:
                st.caption(q.explanation)

    safe_topic = "".join(c if c.isalnum() else "_" for c in quiz.topic.lower())[:40] or "quiz"
    dl_cols = st.columns([1, 1, 4])
    with dl_cols[0]:
        st.download_button(
            "Download Markdown",
            data=quiz.to_markdown(),
            file_name=f"quiz_{safe_topic}.md",
            mime="text/markdown",
            key=f"{key_prefix}-md",
            use_container_width=True,
        )
    with dl_cols[1]:
        st.download_button(
            "Download JSON",
            data=json.dumps(quiz.to_dict(), indent=2),
            file_name=f"quiz_{safe_topic}.json",
            mime="application/json",
            key=f"{key_prefix}-json",
            use_container_width=True,
        )


def render_plan_sessions(plan: StudyPlan) -> None:
    """Render the plan as week-by-week session cards with per-session quiz buttons."""
    api_key = _get_gemini_key()
    quizzes: dict = st.session_state.setdefault("quizzes", {})

    non_review_sessions = [s for s in plan.sessions if not s.is_review]
    syllabus_text = st.session_state.get("last_syllabus", "")

    master_cols = st.columns([1.4, 1, 3])
    with master_cols[0]:
        if st.button(
            f"Generate quizzes for all {len(non_review_sessions)} sessions",
            type="secondary",
            use_container_width=True,
            disabled=not non_review_sessions,
            key="gen-all-quizzes",
        ):
            progress = st.progress(0.0, text="Generating quizzes...")
            total = max(1, len(non_review_sessions))
            for idx, s in enumerate(non_review_sessions, start=1):
                sid = _session_key(s)
                if sid in quizzes:
                    progress.progress(idx / total, text=f"Skipping existing quiz {idx}/{total}")
                    continue
                quizzes[sid] = generate_quiz(
                    topic=s.topic,
                    technique=s.technique,
                    syllabus_text=syllabus_text,
                    n_questions=5,
                    api_key=api_key,
                )
                progress.progress(idx / total, text=f"Generated {idx}/{total}")
            progress.empty()
            st.success(f"Generated quizzes for {len(non_review_sessions)} sessions.")

    with master_cols[1]:
        if st.button(
            "Clear all quizzes",
            use_container_width=True,
            disabled=not quizzes,
            key="clear-all-quizzes",
        ):
            quizzes.clear()
            st.rerun()

    for week in range(1, plan.weeks + 1):
        week_sessions = plan.sessions_in_week(week)
        if not week_sessions:
            continue
        st.markdown(f"#### Week {week}")
        for s in week_sessions:
            badge_cls = "badge badge-mixed" if s.is_review else "badge badge-moderate-high"
            badge_label = "review" if s.is_review else s.technique
            actions_html = "".join(f"<li>{a}</li>" for a in s.actions)
            st.markdown(
                f"""
                <div class="tech-card" style="margin-bottom: 0.4rem;">
                  <div style="flex: 1;">
                    <div class="tech-rank">Session {s.session_in_week}</div>
                    <div class="tech-name">{s.topic}</div>
                    <div class="tech-meta">
                      <span class="{badge_cls}">{badge_label}</span>
                      <span class="pill">{s.technique}</span>
                      <span class="pill">{s.duration_minutes} min</span>
                    </div>
                    <ul style="margin-top: 0.7rem; padding-left: 1.25rem; color: #374151; font-size: 0.92rem;">
                      {actions_html}
                    </ul>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            session_id = _session_key(s)
            if s.is_review:
                continue

            btn_cols = st.columns([1, 1, 4])
            existing = quizzes.get(session_id)
            label = "Regenerate quiz" if existing else "Generate practice quiz"
            with btn_cols[0]:
                if st.button(label, key=f"gen-{session_id}", use_container_width=True):
                    syllabus_text = st.session_state.get("last_syllabus", "")
                    with st.spinner(f"Generating quiz for {s.topic}..."):
                        quiz = generate_quiz(
                            topic=s.topic,
                            technique=s.technique,
                            syllabus_text=syllabus_text,
                            n_questions=5,
                            api_key=api_key,
                        )
                    quizzes[session_id] = quiz
                    existing = quiz
            if existing is not None:
                with btn_cols[1]:
                    if st.button("Clear quiz", key=f"clear-{session_id}", use_container_width=True):
                        quizzes.pop(session_id, None)
                        existing = None
                        st.rerun()

            if existing is not None:
                render_quiz(existing, key_prefix=f"q-{session_id}")

            st.markdown("<div style='margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)


def render_study_plan_tab() -> None:
    last_analysis = st.session_state.get("last_analysis")
    last_syllabus = st.session_state.get("last_syllabus", "")

    if not last_analysis:
        st.info(
            "Analyze a syllabus first on the **Analyze** tab. "
            "Once you have recommendations, come back here to build a schedule."
        )
        return

    top_techniques = [t["technique"] for t in last_analysis["top_techniques"]]
    top_course_type = last_analysis["course_type_scores"][0][0]
    default_technique = top_techniques[0]

    st.markdown("#### Plan parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        course_title = st.text_input(
            "Course title",
            value=top_course_type.replace(" Learning", "").strip(),
            help="Shown at the top of the generated plan and calendar.",
        )
    with col2:
        weeks = st.slider("Weeks until exam / deadline", 1, 16, 4)
    with col3:
        hours_per_week = st.slider("Hours per week", 1.0, 20.0, 5.0, 0.5)

    col4, col5 = st.columns([2, 1])
    with col4:
        primary_technique = st.selectbox(
            "Primary technique",
            options=top_techniques,
            index=0,
            help="Top recommendation from the Analyze tab. Override if you want.",
        )
    with col5:
        add_review = st.toggle("Add weekly spaced review", value=True)

    api_key = _get_gemini_key()
    if api_key:
        st.caption(
            "**LLM topic extraction enabled** — Gemini Flash will read your syllabus "
            "and extract topics. Falls back to heuristics if the call fails."
        )
    else:
        st.caption(
            "**Heuristic topic extraction** — add a `GEMINI_API_KEY` to "
            "`.streamlit/secrets.toml` for LLM-extracted topics. The plan still "
            "works without it."
        )

    if st.button("Generate study plan", type="primary", use_container_width=True):
        with st.spinner("Building your plan..."):
            plan = generate_study_plan(
                course_title=course_title or top_course_type,
                course_type=top_course_type,
                primary_technique=primary_technique,
                syllabus_text=last_syllabus,
                weeks=weeks,
                hours_per_week=hours_per_week,
                add_spaced_review=add_review,
                api_key=api_key,
            )
        st.session_state["last_plan"] = plan
        st.session_state["quizzes"] = {}
        st.session_state["grades"] = {}
        st.success(f"Generated {plan.total_sessions} sessions over {plan.weeks} weeks.")

    plan: StudyPlan | None = st.session_state.get("last_plan")
    if plan is None:
        return

    st.divider()
    render_plan_overview(plan)
    st.divider()
    render_plan_sessions(plan)
    st.divider()

    st.markdown("#### Export")
    dcols = st.columns(3)
    md_bytes = plan.to_markdown().encode("utf-8")
    json_bytes = pd.Series(plan.to_dict()).to_json(indent=2).encode("utf-8")
    start = date.today() + timedelta(days=1)
    ics_bytes = plan.to_ics(start_date=start).encode("utf-8")
    safe_name = "".join(c if c.isalnum() else "_" for c in plan.course_title).strip("_") or "study_plan"

    dcols[0].download_button(
        "Download Markdown",
        data=md_bytes,
        file_name=f"{safe_name}_plan.md",
        mime="text/markdown",
        use_container_width=True,
    )
    dcols[1].download_button(
        "Download Calendar (.ics)",
        data=ics_bytes,
        file_name=f"{safe_name}_plan.ics",
        mime="text/calendar",
        use_container_width=True,
        help=f"Sessions scheduled daily starting {start.isoformat()}.",
    )
    dcols[2].download_button(
        "Download JSON",
        data=json_bytes,
        file_name=f"{safe_name}_plan.json",
        mime="application/json",
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Progress tab
# ──────────────────────────────────────────────────────────────────────────────


def _render_weak_topic_cards(weak: list[dict]) -> None:
    """Tech-card grid for weak topics with attempts + avg score + suggestion."""
    for t in weak:
        action = (
            "Re-study fundamentals and retry"
            if t["avg_score"] <= 2.0
            else "Revisit key concepts and try again"
            if t["avg_score"] <= 3.0
            else "Short refresh — close to mastery"
        )
        badge_cls = "badge badge-mixed" if t["avg_score"] <= 2.0 else "badge badge-moderate"
        techniques_pills = " ".join(
            f'<span class="pill">{tech}</span>' for tech in t["techniques"]
        )
        st.markdown(
            f"""
            <div class="tech-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <div class="tech-rank">Weak topic</div>
                        <div class="tech-name">{t['topic']}</div>
                        <div class="tech-meta">
                            <span class="{badge_cls}">avg {t['avg_score']:.2f}/5</span>
                            <span class="pill">{t['attempts']} attempts</span>
                            <span class="pill">last score {t['last_score']}/5</span>
                            {techniques_pills}
                        </div>
                        <div class="tech-summary">{action}.</div>
                    </div>
                    <div style="text-align: right; padding-left: 1.5rem;">
                        <div class="tech-score">{t['avg_score']:.1f}</div>
                        <div class="tech-meta">avg score</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_recommendation_cards(recs: list[dict]) -> None:
    """Top-N 'study this next' cards from PracticeLog.recommended_review()."""
    for i, r in enumerate(recs, start=1):
        st.markdown(
            f"""
            <div class="tech-card tech-card-top">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <div class="tech-rank">Suggestion {i}</div>
                        <div class="tech-name">{r['topic']}</div>
                        <div class="tech-meta">
                            <span class="badge badge-moderate-high">{r['technique'] or 'review'}</span>
                            <span class="pill">avg {r['avg_score']:.2f}/5</span>
                            <span class="pill">last {r['last_score']}/5</span>
                            <span class="pill">{r['attempts']} attempts</span>
                        </div>
                        <div class="tech-summary">{r['reason']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_progress_tab() -> None:
    """Personal practice analytics derived from graded quiz attempts."""
    practice_log: PracticeLog = st.session_state.setdefault("practice_log", PracticeLog())
    stats = practice_log.summary_stats()

    if stats["total_attempted"] == 0:
        st.info(
            "**No practice data yet.** This tab tracks every quiz answer you grade — "
            "running accuracy, average score, weak topics, and what to review next. "
            "Open the **Study Plan** tab, generate a quiz for any session, write an "
            "answer, and click **Check my answer**. Your stats will appear here.",
            icon=":material/insights:",
        )
        return

    # ── Summary metrics ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Questions attempted", stats["total_attempted"])
    m2.metric("Accuracy", f"{stats['accuracy_pct']:.1f}%")
    m3.metric("Average score", f"{stats['avg_score']:.2f} / 5")
    m4.metric("Topics seen", stats["topics_seen"])

    st.divider()

    # ── Score trend ─────────────────────────────────────────────────────
    st.markdown("#### Score trend")
    trend = practice_log.score_trend(window=5)
    if len(trend) >= 1:
        trend_df = pd.DataFrame(trend, columns=["Attempt", "Running average (last 5)"])
        fig = px.line(
            trend_df,
            x="Attempt",
            y="Running average (last 5)",
            markers=True,
            range_y=[0, 5.2],
        )
        fig.update_traces(line=dict(color="#4F46E5", width=3), marker=dict(size=8))
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=20, t=10, b=10),
            plot_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", dtick=1)
        fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Score trend appears after your first graded answer.")

    # ── Per-technique averages ──────────────────────────────────────────
    st.markdown("#### Average score by technique")
    breakdown = practice_log.technique_breakdown()
    if breakdown:
        bdf = pd.DataFrame(breakdown).sort_values("avg_score")
        bar_fig = px.bar(
            bdf,
            x="avg_score",
            y="technique",
            orientation="h",
            text=bdf["avg_score"].apply(lambda v: f"{v:.2f}"),
            hover_data={"attempts": True, "accuracy_pct": True},
            color="avg_score",
            color_continuous_scale=[(0, "#FCA5A5"), (0.5, "#FCD34D"), (1, "#34D399")],
            range_color=[0, 5],
        )
        bar_fig.update_traces(textposition="outside", cliponaxis=False)
        bar_fig.update_layout(
            height=max(220, 60 * len(bdf) + 80),
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Avg score (0-5)",
            yaxis_title=None,
            coloraxis_showscale=False,
            plot_bgcolor="white",
        )
        bar_fig.update_xaxes(range=[0, 5.5], showgrid=True, gridcolor="#F1F5F9")
        bar_fig.update_yaxes(showgrid=False)
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.caption("Practice with a quiz to populate this chart.")

    st.divider()

    # ── Weak topics ─────────────────────────────────────────────────────
    st.markdown("#### Weak topics")
    weak = practice_log.weak_topics(min_attempts=2, max_avg=3.0)
    if weak:
        _render_weak_topic_cards(weak)
    else:
        st.caption(
            "No topics flagged as weak yet — that's either great work, or you need a "
            "couple more attempts before there's enough signal."
        )

    # ── Recommended review ──────────────────────────────────────────────
    st.markdown("#### Recommended next study session")
    recs = practice_log.recommended_review(top_k=3)
    if recs:
        _render_recommendation_cards(recs)
    else:
        st.caption("Once you've graded a few answers, suggestions will appear here.")

    st.divider()

    # ── Export ──────────────────────────────────────────────────────────
    st.markdown("#### Export")
    dl_cols = st.columns([1, 1, 4])
    with dl_cols[0]:
        st.download_button(
            "Download practice log (JSON)",
            data=json.dumps(practice_log.to_dict(), indent=2),
            file_name="practice_log.json",
            mime="application/json",
            use_container_width=True,
            key="dl-practice-log",
        )
    with dl_cols[1]:
        if st.button("Clear log", use_container_width=True, key="clear-practice-log"):
            practice_log.clear()
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    st.markdown('<div class="hero-title">Learning Technique Recommender</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Paste a course syllabus and get the study techniques '
        'most supported for it — grounded in peer-reviewed cognitive-science meta-analyses, '
        'with citations for every recommendation.</div>',
        unsafe_allow_html=True,
    )

    default_text = render_sidebar()

    tab_analyze, tab_plan, tab_progress, tab_explore, tab_about = st.tabs(
        ["Analyze", "Study Plan", "Progress", "Explore Data", "How It Works"]
    )

    # ── Analyze tab ───────────────────────────────────────────────────────────
    with tab_analyze:
        col_left, col_right = st.columns([1.05, 1])

        with col_left:
            st.markdown("#### Syllabus")

            with st.expander("Or upload a syllabus file (PDF, TXT, MD)", expanded=False):
                uploaded_syllabus = st.file_uploader(
                    "Drop a course syllabus",
                    type=["pdf", "txt", "md", "markdown"],
                    accept_multiple_files=False,
                    label_visibility="collapsed",
                    key="syllabus_upload",
                )
                if uploaded_syllabus is not None:
                    if (
                        st.session_state.get("uploaded_filename")
                        != uploaded_syllabus.name
                    ):
                        with st.spinner(f"Extracting text from {uploaded_syllabus.name}..."):
                            extracted = extract_syllabus(
                                file_bytes=uploaded_syllabus.getvalue(),
                                filename=uploaded_syllabus.name,
                                api_key=_get_gemini_key(),
                            )
                        st.session_state["uploaded_filename"] = uploaded_syllabus.name
                        st.session_state["uploaded_text"] = extracted.text
                        st.session_state["uploaded_meta"] = extracted

                if st.session_state.get("uploaded_text"):
                    meta = st.session_state.get("uploaded_meta")
                    if meta is not None:
                        badge = "Gemini vision" if meta.used_llm else "native"
                        st.caption(
                            f"Extracted **{meta.char_count:,}** characters from "
                            f"`{meta.filename}` ({meta.source} · {badge})"
                        )
                        for w in meta.warnings:
                            st.caption(f"_{w}_")

            default_value = (
                st.session_state.get("uploaded_text") or default_text
            )
            syllabus = st.text_area(
                "Syllabus text",
                value=default_value,
                height=320,
                label_visibility="collapsed",
                key="syllabus_input",
            )
            analyze = st.button("Analyze syllabus", type="primary", use_container_width=True)

        if analyze and syllabus.strip():
            system = get_system()
            with st.spinner("Analyzing..."):
                results = system.analyze_syllabus(syllabus, top_n=5)

            st.session_state["last_analysis"] = results
            st.session_state["last_syllabus"] = syllabus

            top_ct, top_score = results["course_type_scores"][0]
            best = results["top_techniques"][0]

            with col_right:
                st.markdown("#### Summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Detected course type", top_ct.split(" ")[0] + "…", help=top_ct)
                m2.metric("Match confidence", f"{top_score * 100:.1f}%")
                m3.metric("Top evidence score", f"{best['evidence_score']:.1f}")

                st.markdown("##### Course type match")
                render_course_type_chart(results["course_type_scores"])

            st.divider()
            st.markdown("### Recommended Techniques")
            render_technique_cards(results["top_techniques"])

            with st.expander("Full breakdown — every technique across every course type"):
                render_detailed_table(results["all_techniques"])

            st.info(
                "Open the **Study Plan** tab to turn these recommendations into a "
                "concrete, session-by-session schedule.",
                icon=":material/event_note:",
            )

        elif analyze:
            st.warning("Please paste a syllabus before analyzing.")
        else:
            with col_right:
                st.info("Pick or paste a syllabus on the left, then click **Analyze syllabus**.")

    # ── Study Plan tab ────────────────────────────────────────────────────────
    with tab_plan:
        render_study_plan_tab()

    # ── Progress tab ──────────────────────────────────────────────────────────
    with tab_progress:
        render_progress_tab()

    # ── Explore Data tab ──────────────────────────────────────────────────────
    with tab_explore:
        st.markdown("#### Browse historical effectiveness data")
        st.caption(
            "These average grades drive the recommendations. Use the sidebar to upload your "
            "own CSV and replace this data with your own."
        )
        render_data_explorer(get_system())

    # ── About tab ─────────────────────────────────────────────────────────────
    with tab_about:
        st.markdown("""
#### How it works

The pipeline has five stages — three deterministic, two LLM-assisted:

1. **Syllabus ingestion** — Paste text or drop a PDF/TXT/Markdown file. Text-based PDFs
   are parsed natively via `pypdf`; scanned or image-based PDFs fall back to Gemini 2.5
   Flash multimodal vision when an API key is configured.
2. **Syllabus classifier** — A TF-IDF vectorizer + cosine similarity scores the syllabus
   against five hand-crafted course-type descriptions. Output: a confidence score for each
   course type in the range [0, 1].
3. **Technique recommender** — For each course type, an evidence score per technique is
   pre-computed from peer-reviewed meta-analyses (see Methodology below). The system ranks
   techniques best-first.
4. **Study plan generation** — The top technique drives a session-by-session schedule.
   Topics are extracted from the syllabus (Gemini if available, regex heuristic otherwise),
   then deterministic per-technique templates produce the concrete actions for each session.
5. **Practice quizzes** — Each non-review session can produce a 5-question quiz tuned to
   its technique (Active Recall → recall prompts, Worked Examples → solve-and-explain,
   Feynman → explain-it-simply, etc.). Generated via Gemini with a JSON schema, or
   falls back to template prompts offline.

The core scoring formula for ranking techniques:

   ```
   combined = course_weight × course_match + (1 − course_weight) × (evidence_score / 100)
   ```

   The slider in the sidebar controls `course_weight`. At `1.0`, only course-type match
   matters; at `0.0`, only the strength of evidence for the technique matters.

#### When the LLM is — and isn't — used

The architecture deliberately keeps LLM calls in two narrow lanes:

- **Topic extraction** from the syllabus (one call per plan generation)
- **Quiz generation** for a specific session (one call per quiz)
- **PDF vision fallback** (only triggered when native PDF parsing yields too little text)

Everything else — classification, scoring, ranking, session scheduling, exports,
evidence lookups — is deterministic code with no external dependencies. The app is
fully usable offline; the LLM features simply unlock when an API key is provided.

#### Methodology — how the evidence scores are derived

Rather than synthetic grade data, this system uses a research-grounded evidence base. For
each technique:

1. **Evidence rating** — a categorical assessment of the strength of meta-analytic support
   (`high` · `moderate-high` · `moderate` · `mixed` · `low`), mapped to numeric weights
   `{1.00, 0.88, 0.75, 0.55, 0.40}`.
2. **Domain alignment** — a percentage in `[0, 100]` capturing how strongly the technique's
   support extends to the specific course type's subfield (e.g., worked examples have strong
   support for procedural mathematics but weak support for second-language acquisition).
3. **Evidence score** — `weight × alignment`, bounded to `[0, 100]`.

Each technique in the recommendation cards links back to its primary research source,
with the effect size and study count when available. Open the "Detailed Breakdown" expander
on the Analyze tab to see all 30 (course type × technique) pairs.
""")

        st.markdown("#### Research sources")
        sources_by_rating: dict[str, list] = {
            "high": [], "moderate-high": [], "moderate": [], "mixed": [], "low": [],
        }
        for evidence in RESEARCH.values():
            sources_by_rating[evidence.evidence_rating].append(evidence)

        for rating in ["high", "moderate-high", "moderate", "mixed", "low"]:
            entries = sources_by_rating.get(rating, [])
            if not entries:
                continue
            st.markdown(f"##### {rating.replace('-', '–').title()} evidence")
            for e in sorted(entries, key=lambda x: x.name):
                effect_str = ""
                if e.effect_size is not None and e.effect_metric:
                    sign = "+" if e.effect_size >= 0 else ""
                    effect_str = f" *({e.effect_metric} {sign}{e.effect_size:.2f}"
                    if e.study_count:
                        effect_str += f", {e.study_count} studies"
                    effect_str += ")*"
                st.markdown(
                    f"- **{e.name}**{effect_str} — {e.short_citation}  \n"
                    f"  <small>{e.summary}</small>",
                    unsafe_allow_html=True,
                )

        st.markdown("#### Project structure")
        st.markdown("""
```
src/
  research.py     — Peer-reviewed evidence base with citations & effect sizes
  classifier.py   — SyllabusClassifier (TF-IDF + cosine similarity)
  recommender.py  — LearningTechniqueRecommender (per-course-type ranking)
  pipeline.py     — LearningStyleSystem (end-to-end orchestration)
  data.py         — Default scores (derived from research.py), example syllabi
app.py            — This Streamlit interface
Learning_Recommendation_Pipeline.ipynb — Notebook walkthrough
```
""")


if __name__ == "__main__":
    main()
