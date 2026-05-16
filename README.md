# Learning Technique Recommender

A web application that recommends the most effective study techniques for any course, given just the raw syllabus. **Every recommendation is grounded in a peer-reviewed meta-analysis or seminal study** — there is no synthetic data anywhere in the pipeline.

> **Stack:** Python · scikit-learn · pandas · Streamlit · Plotly

---

## What it does

Upload your course syllabus (or paste it) and the system:

1. **Extracts text from PDFs** — native parsing for text-based PDFs, Gemini vision fallback for scanned / image-based PDFs
2. **Classifies** the syllabus into one of five course types using TF-IDF + cosine similarity
3. **Recommends** study techniques most supported for that course type, using a research-grounded evidence base spanning 21 techniques and 22 meta-analyses / seminal studies — every recommendation cites its primary source and reports the effect size
4. **Generates a concrete, session-by-session study plan** for the recommended technique — exportable as Markdown, ICS calendar, or JSON
5. **Generates AI practice quizzes** tailored to each session's technique — Active Recall sessions get factual recall, Worked Example sessions get solve-and-explain, Feynman sessions get explain-it-simply prompts, etc.

You can adjust the scoring weight in real time, browse the underlying evidence base, or upload your own grade CSV to retrain on your data.

---

## Methodology

Rather than rely on synthetic grades, each (course type × technique) score is derived from published research:

```
evidence_score = evidence_weight × domain_alignment
```

- **`evidence_weight`** — a numeric weight derived from each technique's meta-analytic support
  - `high` (1.00) — replicated meta-analyses, e.g. practice testing (Adesope, Trevisan, & Sundararajan 2017; *g* = 0.61, 272 studies)
  - `moderate-high` (0.88) — strong but more bounded support, e.g. self-explanation (Bisra et al. 2018)
  - `moderate` (0.75) — solid framework or single meta-analysis, e.g. simulation-based learning (Vogel et al. 2006)
  - `mixed` (0.55) — meaningful but contested evidence, e.g. deliberate practice in education (Macnamara et al. 2014)
  - `low` (0.40) — minimal or negative effects
- **`domain_alignment`** — a percentage in `[0, 100]` capturing how well the technique's evidence extends to that course type's subfield (e.g. worked examples have strong alignment with calculation-driven learning but weak alignment with language learning)

The recommender then computes a combined score:

```
combined = course_weight × course_match + (1 − course_weight) × (evidence_score / 100)
```

The sidebar slider exposes `course_weight` so users can experiment between "match the syllabus precisely" and "use the universally strongest technique."

### Key research sources

| Technique | Source | Effect |
|---|---|---|
| Active Recall | Adesope, Trevisan, & Sundararajan (2017) | Hedges' g = 0.61 (272 studies) |
| Spaced Repetition | Cepeda et al. (2006) | Hedges' g = 0.42 (254 studies) |
| Feynman / Self-Explanation | Bisra et al. (2018) | Hedges' g = 0.55 (64 studies) |
| Conceptual Mapping | Nesbit & Adesope (2006) | Hedges' g = 0.43 (55 studies) |
| Worked Examples | Renkl (2014); Sweller (1988) | Foundational |
| Project-Based Learning | Chen & Yang (2019) | Hedges' g = 0.71 (46 studies) |
| Comparative Analysis | Alfieri, Nokes-Malach, & Schunn (2013) | Hedges' g = 0.50 (57 studies) |
| Scaffolded Skill Building | Belland et al. (2017) | Hedges' g = 0.46 (144 studies) |
| Open-Ended Exploration | Alfieri et al. (2011) | Cohen's d = −0.08 (164 studies) |
| Synthesis | Dunlosky et al. (2013) | *Psychological Science in the Public Interest* — landmark review of 10 study techniques |

Full APA citations for all 22 sources are embedded in [`src/research.py`](src/research.py).

---

## Run locally

```bash
git clone https://github.com/BarotDvij/Learning-Technique-Recommendations.git
cd Learning-Technique-Recommendations

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. **New app** → pick this repo → main file `app.py` → Deploy

Streamlit Cloud reads `requirements.txt` and `.streamlit/config.toml` automatically.

---

## Project structure

```
Learning-Technique-Recommendations/
├── app.py                              Streamlit UI
├── requirements.txt
├── .streamlit/
│   ├── config.toml                     theme + server settings
│   └── secrets.example.toml            template for GEMINI_API_KEY
├── src/
│   ├── research.py                     peer-reviewed evidence base (citations + effect sizes)
│   ├── classifier.py                   SyllabusClassifier (TF-IDF + cosine similarity)
│   ├── recommender.py                  LearningTechniqueRecommender (per-course-type ranking)
│   ├── pipeline.py                     LearningStyleSystem (end-to-end orchestration)
│   ├── parsing.py                      PDF / TXT extraction with Gemini vision fallback
│   ├── plan.py                         StudyPlan + per-technique session templates
│   ├── quiz.py                         Per-technique AI quiz generation
│   └── data.py                         default scores (derived from research.py), examples
└── Learning_Recommendation_Pipeline.ipynb   notebook walkthrough
```

---

## Course types supported

| Course Type | Best Technique (research-grounded) | Primary Source |
|---|---|---|
| Applied Calculation-Driven Learning | Worked Example Analysis | Renkl (2014); Sweller (1988) |
| Deep Conceptual Learning | Feynman Technique (Self-Explanation) | Bisra et al. (2018) |
| Case-Based & Strategic Learning | Case Study Analysis | Williams (2005) |
| Language & Communication-Based Learning | Immersive Practice | Krashen (1982) |
| Hands-On, Project-Based Learning | Incremental Skill Building | Belland et al. (2017) |

---

## Study plan generation

The **Study Plan** tab turns a recommended technique into a concrete schedule:

- **Topic extraction** — Gemini 2.5 Flash reads the syllabus and pulls out the main study topics. Falls back to regex-based heuristic parsing if no API key is configured, so the app works fully offline.
- **Session templates** — each top-supported technique has its own deterministic template (Worked Example Analysis, Feynman, Conceptual Mapping, Case Study Analysis, Immersive Practice, Incremental Skill Building, Project-Based Learning, Active Recall, Spaced Repetition). Templates produce a session-by-session schedule with concrete sub-steps.
- **Per-session AI quizzes** — click "Generate practice quiz" on any session to get 5 questions tuned to that session's technique. Each question reveals the model answer + an explanation on click. Generated by Gemini 2.5 Flash with a structured JSON schema. Falls back to a template prompt if no API key.
- **Spaced review** — when the primary technique isn't already a review technique, short weekly Spaced Repetition consolidation sessions are added at the end of each week.
- **Exports** — download as Markdown, ICS (drop into Google Calendar / Apple Calendar / Outlook), or raw JSON.

**Architecture principle.** The LLM is used only for the parts that can't be precomputed: topic extraction, quiz generation, and (when needed) vision-based PDF parsing. Session scheduling, technique templates, scoring, and exports are all deterministic code — no token use, no rate limits, no surprises.

### Enabling LLM topic extraction (optional)

1. Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (no credit card required).
2. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and paste your key.
3. Restart the app. The Study Plan tab will display "LLM topic extraction enabled."

Without a key, the app uses a heuristic parser. Plans are still useful, just with simpler topic names.

---

## Use your own data

Upload a CSV from the sidebar with these columns:

- `Course Type`
- `Learning Technique`
- `Grade of Module (%)`

The recommender will recompute average grades per (course type, technique) pair and use those for ranking, bypassing the default evidence-based scores.

---

## Reference
[Data Goons Final Report](https://github.com/user-attachments/files/19886497/Data.Goons.Final.Report.1.pdf)
