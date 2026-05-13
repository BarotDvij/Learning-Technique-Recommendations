"""
Learning Technique Recommender — Streamlit App.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import (
    DEFAULT_TECHNIQUE_GRADES,
    EXAMPLE_SYLLABI,
    LearningStyleSystem,
    REQUIRED_COLUMNS,
    RESEARCH,
    compute_technique_grades,
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

    tab_analyze, tab_explore, tab_about = st.tabs(["Analyze", "Explore Data", "How It Works"])

    # ── Analyze tab ───────────────────────────────────────────────────────────
    with tab_analyze:
        col_left, col_right = st.columns([1.05, 1])

        with col_left:
            st.markdown("#### Syllabus")
            syllabus = st.text_area(
                "Syllabus text",
                value=default_text,
                height=320,
                label_visibility="collapsed",
                key="syllabus_input",
            )
            analyze = st.button("Analyze syllabus", type="primary", use_container_width=True)

        if analyze and syllabus.strip():
            system = get_system()
            with st.spinner("Analyzing..."):
                results = system.analyze_syllabus(syllabus, top_n=5)

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

        elif analyze:
            st.warning("Please paste a syllabus before analyzing.")
        else:
            with col_right:
                st.info("Pick or paste a syllabus on the left, then click **Analyze syllabus**.")

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

The pipeline has three stages:

1. **Syllabus classifier** — A TF-IDF vectorizer + cosine similarity scores the syllabus
   against five hand-crafted course-type descriptions. Output: a confidence score for each
   course type in the range [0, 1].
2. **Technique recommender** — For each course type, an evidence score per technique is
   pre-computed from peer-reviewed meta-analyses (see Methodology below). The system ranks
   techniques best-first.
3. **Combined scoring** — For each (course type, technique) pair:

   ```
   combined = course_weight × course_match + (1 − course_weight) × (evidence_score / 100)
   ```

   The slider in the sidebar controls `course_weight`. At `1.0`, only course-type match
   matters; at `0.0`, only the strength of evidence for the technique matters.

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
