# Learning Technique Recommender

A pipeline that takes a raw course syllabus and recommends the most effective study techniques based on historical grade data.

## How it works

1. **Data Analysis** — load a CSV of student records and compute average grades per technique per course type
2. **Syllabus Classifier** — use TF-IDF + cosine similarity to map a syllabus to one of 5 course types
3. **Recommender** — rank learning techniques by effectiveness for the detected course type
4. **Integrated System** — combine classifier + recommender into a single `analyze_syllabus()` call
5. **Demo** — run the full pipeline on example syllabi

## Usage

Open `Learning_Recommendation_Pipeline.ipynb` in Google Colab or Jupyter and run all cells top to bottom.

- In **Colab**: Part 1 will prompt you to upload a CSV with columns `Course Type`, `Learning Technique`, and `Grade of Module (%)`. If no file is uploaded, pre-analyzed defaults are used.
- **Outside Colab**: the default pre-analyzed dataset is used automatically.

## Course types supported

| Course Type | Best Technique |
|---|---|
| Applied Calculation-Driven Learning | Worked Example Analysis |
| Deep Conceptual Learning | Conceptual Mapping |
| Case-Based & Strategic Learning | Case Study Analysis |
| Language & Communication-Based Learning | Immersive Practice |
| Hands-On, Project-Based Learning | Incremental Skill Building |

## Reference

[Data Goons Final Report](https://github.com/user-attachments/files/19886497/Data.Goons.Final.Report.1.pdf)
