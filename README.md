# Learning Technique Recommender

A web application that recommends the most effective study techniques for any course, given just the raw syllabus. Built with scikit-learn for the classification layer, Streamlit + Plotly for the interactive UI, and backed by a statistical analysis of historical student grade data.

> **Stack:** Python · scikit-learn · pandas · Streamlit · Plotly

---

## What it does

Paste a course syllabus and the system:

1. Classifies the syllabus into one of five course types using TF-IDF + cosine similarity
2. Looks up which learning techniques produced the highest grades for that course type in historical data
3. Returns a ranked list of recommended techniques with a combined score that blends course-type match and historical effectiveness

You can adjust the weighting in real time, browse the underlying effectiveness data, or upload your own CSV to retrain the recommender on your data.

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

The app will open at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, pick this repo, set the main file to `app.py`, and deploy

That's it — Streamlit Cloud reads `requirements.txt` and runs the app automatically.

---

## Project structure

```
Learning-Technique-Recommendations/
├── app.py                              Streamlit UI
├── requirements.txt
├── .streamlit/config.toml              theme & server settings
├── src/
│   ├── classifier.py                   SyllabusClassifier — TF-IDF + cosine similarity
│   ├── recommender.py                  LearningTechniqueRecommender — per-course-type ranking
│   ├── pipeline.py                     LearningStyleSystem — end-to-end orchestration
│   └── data.py                         default grades, example syllabi, CSV utilities
└── Learning_Recommendation_Pipeline.ipynb   notebook walkthrough of the pipeline
```

---

## How the pipeline works

### Stage 1 — Classification
`SyllabusClassifier` vectorizes the syllabus with `TfidfVectorizer` (English stop words removed) and computes cosine similarity against five hand-crafted course-type descriptions. Output: a confidence score in `[0, 1]` for each course type.

### Stage 2 — Recommendation
`LearningTechniqueRecommender` is initialized with a `{course_type: {technique: avg_grade}}` dict computed from a historical dataset (5 course types × 24 techniques). For a given course type, it returns techniques ranked by average historical grade.

### Stage 3 — Combined scoring
`LearningStyleSystem` computes:

```
combined = course_weight × course_match + (1 − course_weight) × (expected_grade / 100)
```

The default `course_weight = 0.5` weights both signals equally. The sidebar slider in the UI lets you adjust this live.

---

## Course types supported

| Course Type | Best Technique (default data) |
|---|---|
| Applied Calculation-Driven Learning | Worked Example Analysis |
| Deep Conceptual Learning | Conceptual Mapping |
| Case-Based & Strategic Learning | Case Study Analysis |
| Language & Communication-Based Learning | Immersive Practice |
| Hands-On, Project-Based Learning | Incremental Skill Building |

---

## Use your own data

Upload a CSV from the sidebar with these columns:

- `Course Type`
- `Learning Technique`
- `Grade of Module (%)`

The recommender will recompute average grades per (course type, technique) pair and use those for ranking.

---

## Reference
[Data Goons Final Report](https://github.com/user-attachments/files/19886497/Data.Goons.Final.Report.1.pdf)
