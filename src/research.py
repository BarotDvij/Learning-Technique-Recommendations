"""
Peer-reviewed evidence base for learning technique recommendations.

Each technique is mapped to a representative meta-analysis or seminal study,
with effect sizes (Cohen's d / Hedges' g) reported when available. An
``evidence_rating`` summarizes the overall strength of evidence and is used
as the universal weight in the recommendation score:

    evidence_score(course_type, technique) = EVIDENCE_WEIGHTS[rating]
                                           * COURSE_TYPE_ALIGNMENT[course_type][technique]

The alignment values are domain-fit percentages on [0, 100], reflecting how
much the technique is supported in that specific subfield's instructional
research. Multiplying by the universal evidence weight prevents weakly-evidenced
techniques from dominating purely on alignment.

Primary references (full APA citations are stored per-technique below):

    Adesope, O. O., Trevisan, D. A., & Sundararajan, N. (2017). Rethinking the
        use of tests: A meta-analysis of practice testing.
        Review of Educational Research, 87(3), 659–701.
    Alfieri, L., Brooks, P. J., Aldrich, N. J., & Tenenbaum, H. R. (2011).
        Does discovery-based instruction enhance learning?
        Journal of Educational Psychology, 103(1), 1–18.
    Alfieri, L., Nokes-Malach, T. J., & Schunn, C. D. (2013). Learning through
        case comparisons: A meta-analytic review.
        Educational Psychologist, 48(2), 87–113.
    Belland, B. R., Walker, A. E., Kim, N. J., & Lefler, M. (2017). Synthesizing
        results from empirical research on computer-based scaffolding in STEM
        education: A meta-analysis. Review of Educational Research, 87(2), 309–344.
    Bisra, K., Liu, Q., Nesbit, J. C., Salimi, F., & Winne, P. H. (2018).
        Inducing self-explanation: A meta-analysis.
        Educational Psychology Review, 30, 703–725.
    Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006).
        Distributed practice in verbal recall tasks: A review and quantitative
        synthesis. Psychological Bulletin, 132(3), 354–380.
    Chen, C.-H., & Yang, Y.-C. (2019). Revisiting the effects of project-based
        learning on students' academic achievement: A meta-analysis.
        Educational Research Review, 26, 71–81.
    Collins, A., Brown, J. S., & Newman, S. E. (1989). Cognitive apprenticeship:
        Teaching the crafts of reading, writing, and mathematics.
        In L. B. Resnick (Ed.), Knowing, Learning, and Instruction (pp. 453–494).
    Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T.
        (2013). Improving students' learning with effective learning techniques:
        Promising directions from cognitive and educational psychology.
        Psychological Science in the Public Interest, 14(1), 4–58.
    Graham, S., & Sandmel, K. (2011). The process writing approach: A meta-analysis.
        The Journal of Educational Research, 104(6), 396–407.
    Hamada, Y. (2017). Teaching EFL Learners Shadowing for Listening. Routledge.
    Kolb, D. A. (1984). Experiential Learning: Experience as the Source of Learning
        and Development. Prentice-Hall.
    Krashen, S. (1982). Principles and Practice in Second Language Acquisition.
        Pergamon Press.
    Macnamara, B. N., Hambrick, D. Z., & Oswald, F. L. (2014). Deliberate practice
        and performance in music, games, sports, education, and professions: A
        meta-analysis. Psychological Science, 25(8), 1608–1618.
    Nesbit, J. C., & Adesope, O. O. (2006). Learning with concept and knowledge
        maps: A meta-analysis. Review of Educational Research, 76(3), 413–448.
    Papert, S. (1991). Situating constructionism. In I. Harel & S. Papert (Eds.),
        Constructionism (pp. 1–11). Ablex Publishing.
    Renkl, A. (2014). Toward an instructionally oriented theory of example-based
        learning. Cognitive Science, 38(1), 1–37.
    Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking
        memory tests improves long-term retention. Psychological Science, 17(3),
        249–255.
    Sweller, J. (1988). Cognitive load during problem solving: Effects on learning.
        Cognitive Science, 12(2), 257–285.
    Vogel, J. J., Vogel, D. S., Cannon-Bowers, J., Bowers, C. A., Muse, K., &
        Wright, M. (2006). Computer gaming and interactive simulations for
        learning: A meta-analysis. Journal of Educational Computing Research,
        34(3), 229–243.
    Wood, D., Bruner, J. S., & Ross, G. (1976). The role of tutoring in problem
        solving. Journal of Child Psychology and Psychiatry, 17(2), 89–100.

2023–2025 evidence expansion (new techniques):

    Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of
        interleaved learning and its moderating variables.
        Psychological Bulletin, 145(11), 1029–1052.
    Kulik, J. A., & Fletcher, J. D. (2016). Effectiveness of intelligent tutoring
        systems: A meta-analytic review.
        Review of Educational Research, 86(1), 42–78.
    Schroeder, N. L., Nesbit, J. C., Anguiano, C. J., & Adesope, O. O. (2018).
        Studying and constructing concept maps: A meta-analysis.
        Educational Psychology Review, 30(2), 431–455.
    Yang, C., Luo, L., Vadillo, M. A., Yu, R., & Shanks, D. R. (2021). Testing
        (quizzing) boosts classroom learning: A systematic and meta-analytic review.
        Psychological Bulletin, 147(4), 399–435.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TechniqueEvidence:
    """Evidence record for a single learning technique."""

    name: str
    short_citation: str
    full_citation: str
    effect_size: Optional[float]
    effect_metric: str  # 'Cohen\'s d', 'Hedges\' g', or '' if N/A
    study_count: Optional[int]
    evidence_rating: str  # 'high', 'moderate-high', 'moderate', 'mixed', 'low'
    summary: str
    dunlosky_utility: Optional[str] = None  # 'high', 'moderate', 'low'


# ──────────────────────────────────────────────────────────────────────────────
# Evidence rating → universal weight
# ──────────────────────────────────────────────────────────────────────────────

EVIDENCE_WEIGHTS: Dict[str, float] = {
    "high": 1.00,
    "moderate-high": 0.88,
    "moderate": 0.75,
    "mixed": 0.55,
    "low": 0.40,
}


# ──────────────────────────────────────────────────────────────────────────────
# Technique → research record
# ──────────────────────────────────────────────────────────────────────────────

RESEARCH: Dict[str, TechniqueEvidence] = {
    "Active Recall": TechniqueEvidence(
        name="Active Recall",
        short_citation="Adesope, Trevisan, & Sundararajan (2017)",
        full_citation=(
            "Adesope, O. O., Trevisan, D. A., & Sundararajan, N. (2017). "
            "Rethinking the use of tests: A meta-analysis of practice testing. "
            "Review of Educational Research, 87(3), 659–701."
        ),
        effect_size=0.61,
        effect_metric="Hedges' g",
        study_count=272,
        evidence_rating="high",
        summary=(
            "Retrieving information from memory (rather than re-reading) produces "
            "robust learning gains across domains and ages. The largest single "
            "meta-analysis of practice testing found a substantial mean effect."
        ),
        dunlosky_utility="high",
    ),
    "Spaced Repetition": TechniqueEvidence(
        name="Spaced Repetition",
        short_citation="Cepeda et al. (2006)",
        full_citation=(
            "Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. "
            "(2006). Distributed practice in verbal recall tasks: A review and "
            "quantitative synthesis. Psychological Bulletin, 132(3), 354–380."
        ),
        effect_size=0.42,
        effect_metric="Hedges' g",
        study_count=254,
        evidence_rating="high",
        summary=(
            "Spreading study sessions over time yields dramatically better "
            "long-term retention than massed practice. One of the most "
            "replicated findings in memory research."
        ),
        dunlosky_utility="high",
    ),
    "Feynman Technique": TechniqueEvidence(
        name="Feynman Technique",
        short_citation="Bisra et al. (2018)",
        full_citation=(
            "Bisra, K., Liu, Q., Nesbit, J. C., Salimi, F., & Winne, P. H. "
            "(2018). Inducing self-explanation: A meta-analysis. "
            "Educational Psychology Review, 30, 703–725."
        ),
        effect_size=0.55,
        effect_metric="Hedges' g",
        study_count=64,
        evidence_rating="moderate-high",
        summary=(
            "Explaining material to yourself in plain language (or to an "
            "imagined novice) forces conceptual integration and surfaces gaps. "
            "Especially effective for principled, theoretical content."
        ),
        dunlosky_utility="moderate",
    ),
    "Conceptual Mapping": TechniqueEvidence(
        name="Conceptual Mapping",
        short_citation="Nesbit & Adesope (2006)",
        full_citation=(
            "Nesbit, J. C., & Adesope, O. O. (2006). Learning with concept and "
            "knowledge maps: A meta-analysis. "
            "Review of Educational Research, 76(3), 413–448."
        ),
        effect_size=0.43,
        effect_metric="Hedges' g",
        study_count=55,
        evidence_rating="moderate-high",
        summary=(
            "Constructing visual concept maps improves comprehension and "
            "retention of structured knowledge by making relationships explicit."
        ),
    ),
    "Worked Example Analysis": TechniqueEvidence(
        name="Worked Example Analysis",
        short_citation="Renkl (2014); Sweller (1988)",
        full_citation=(
            "Renkl, A. (2014). Toward an instructionally oriented theory of "
            "example-based learning. Cognitive Science, 38(1), 1–37.\n"
            "Sweller, J. (1988). Cognitive load during problem solving: "
            "Effects on learning. Cognitive Science, 12(2), 257–285."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="high",
        summary=(
            "Studying fully worked-out solutions before attempting new problems "
            "reduces extraneous cognitive load and accelerates skill acquisition "
            "in procedural domains. Robust effect for novices in math/STEM."
        ),
    ),
    "Simulation & Visualization": TechniqueEvidence(
        name="Simulation & Visualization",
        short_citation="Vogel et al. (2006)",
        full_citation=(
            "Vogel, J. J., Vogel, D. S., Cannon-Bowers, J., Bowers, C. A., "
            "Muse, K., & Wright, M. (2006). Computer gaming and interactive "
            "simulations for learning: A meta-analysis. "
            "Journal of Educational Computing Research, 34(3), 229–243."
        ),
        effect_size=0.32,
        effect_metric="Cohen's d",
        study_count=32,
        evidence_rating="moderate",
        summary=(
            "Interactive simulations and dynamic visualizations support transfer "
            "in conceptually-rich domains, particularly when paired with guided "
            "reflection. Effect is moderate and depends on instructional support."
        ),
    ),
    "Deliberate Practice": TechniqueEvidence(
        name="Deliberate Practice",
        short_citation="Macnamara, Hambrick, & Oswald (2014)",
        full_citation=(
            "Macnamara, B. N., Hambrick, D. Z., & Oswald, F. L. (2014). "
            "Deliberate practice and performance in music, games, sports, "
            "education, and professions: A meta-analysis. "
            "Psychological Science, 25(8), 1608–1618."
        ),
        effect_size=None,
        effect_metric="",
        study_count=88,
        evidence_rating="mixed",
        summary=(
            "Targeted, feedback-driven practice improves performance, but the "
            "effect size in education is far smaller than originally claimed. "
            "Helpful for procedural mastery; insufficient on its own."
        ),
    ),
    "Reverse Engineering": TechniqueEvidence(
        name="Reverse Engineering",
        short_citation="Renkl (2014)",
        full_citation=(
            "Renkl, A. (2014). Toward an instructionally oriented theory of "
            "example-based learning. Cognitive Science, 38(1), 1–37."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate-high",
        summary=(
            "Working backward from a finished solution to its constituent steps "
            "is a worked-example variant that fosters self-explanation and "
            "structural understanding."
        ),
    ),
    "Iterative Writing & Editing": TechniqueEvidence(
        name="Iterative Writing & Editing",
        short_citation="Graham & Sandmel (2011)",
        full_citation=(
            "Graham, S., & Sandmel, K. (2011). The process writing approach: "
            "A meta-analysis. The Journal of Educational Research, 104(6), "
            "396–407."
        ),
        effect_size=0.34,
        effect_metric="Cohen's d",
        study_count=29,
        evidence_rating="moderate",
        summary=(
            "Drafting, peer review, and revision cycles produce reliably better "
            "writing quality than single-shot writing. Effect is moderate but "
            "domain-specific to writing instruction."
        ),
    ),
    "Case Study Analysis": TechniqueEvidence(
        name="Case Study Analysis",
        short_citation="Williams (2005)",
        full_citation=(
            "Williams, B. (2005). Case based learning—a review of the "
            "literature: Is there scope for this educational paradigm in "
            "prehospital education? Emergency Medicine Journal, 22(8), 577–581."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate-high",
        summary=(
            "Analyzing rich, contextualized cases builds transferable strategic "
            "reasoning. Strong support in medical, legal, and business education."
        ),
    ),
    "Project-Based Learning": TechniqueEvidence(
        name="Project-Based Learning",
        short_citation="Chen & Yang (2019)",
        full_citation=(
            "Chen, C.-H., & Yang, Y.-C. (2019). Revisiting the effects of "
            "project-based learning on students' academic achievement: A "
            "meta-analysis. Educational Research Review, 26, 71–81."
        ),
        effect_size=0.71,
        effect_metric="Hedges' g",
        study_count=46,
        evidence_rating="moderate-high",
        summary=(
            "Sustained, authentic projects outperform traditional instruction "
            "for academic achievement, particularly in STEM. Effect is large "
            "but variable across implementation quality."
        ),
    ),
    "Open-Ended Exploration": TechniqueEvidence(
        name="Open-Ended Exploration",
        short_citation="Alfieri et al. (2011)",
        full_citation=(
            "Alfieri, L., Brooks, P. J., Aldrich, N. J., & Tenenbaum, H. R. "
            "(2011). Does discovery-based instruction enhance learning? "
            "Journal of Educational Psychology, 103(1), 1–18."
        ),
        effect_size=-0.08,
        effect_metric="Cohen's d",
        study_count=164,
        evidence_rating="mixed",
        summary=(
            "Unguided discovery learning produces near-zero or negative effects; "
            "structured guided discovery is moderately beneficial. Effectiveness "
            "is highly dependent on scaffolding."
        ),
    ),
    "Experiential Learning": TechniqueEvidence(
        name="Experiential Learning",
        short_citation="Kolb (1984)",
        full_citation=(
            "Kolb, D. A. (1984). Experiential Learning: Experience as the "
            "Source of Learning and Development. Prentice-Hall."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate",
        summary=(
            "Cycle of concrete experience, reflection, abstraction, and "
            "experimentation. Well-established framework with strong support in "
            "adult education and professional training."
        ),
    ),
    "Comparative Analysis": TechniqueEvidence(
        name="Comparative Analysis",
        short_citation="Alfieri, Nokes-Malach, & Schunn (2013)",
        full_citation=(
            "Alfieri, L., Nokes-Malach, T. J., & Schunn, C. D. (2013). "
            "Learning through case comparisons: A meta-analytic review. "
            "Educational Psychologist, 48(2), 87–113."
        ),
        effect_size=0.50,
        effect_metric="Hedges' g",
        study_count=57,
        evidence_rating="moderate",
        summary=(
            "Side-by-side comparison of two or more cases or examples surfaces "
            "underlying structure and improves transfer to novel problems."
        ),
    ),
    "First-Principles Thinking": TechniqueEvidence(
        name="First-Principles Thinking",
        short_citation="Bisra et al. (2018)",
        full_citation=(
            "Bisra, K., Liu, Q., Nesbit, J. C., Salimi, F., & Winne, P. H. "
            "(2018). Inducing self-explanation: A meta-analysis. "
            "Educational Psychology Review, 30, 703–725."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate",
        summary=(
            "Decomposing problems to their foundational truths and rebuilding "
            "from there is a generative reasoning strategy related to "
            "self-explanation."
        ),
    ),
    "Immersive Practice": TechniqueEvidence(
        name="Immersive Practice",
        short_citation="Krashen (1982)",
        full_citation=(
            "Krashen, S. (1982). Principles and Practice in Second Language "
            "Acquisition. Pergamon Press."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate-high",
        summary=(
            "Sustained, comprehensible input in a target language environment "
            "drives implicit acquisition. Foundational framework in second-language "
            "research; strong evidence for naturalistic language learning."
        ),
    ),
    "Storytelling Frameworks": TechniqueEvidence(
        name="Storytelling Frameworks",
        short_citation="Graham & Sandmel (2011)",
        full_citation=(
            "Graham, S., & Sandmel, K. (2011). The process writing approach: "
            "A meta-analysis. The Journal of Educational Research, 104(6), "
            "396–407."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate",
        summary=(
            "Embedding content in narrative structures (setup, conflict, "
            "resolution) improves engagement and recall in communication-heavy "
            "domains."
        ),
    ),
    "Active Recall & Shadowing": TechniqueEvidence(
        name="Active Recall & Shadowing",
        short_citation="Hamada (2017)",
        full_citation=(
            "Hamada, Y. (2017). Teaching EFL Learners Shadowing for Listening. Routledge."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate-high",
        summary=(
            "Verbatim repetition of native-speaker audio combined with active "
            "retrieval improves phonological encoding and listening fluency in "
            "second-language learners."
        ),
    ),
    "Incremental Skill Building": TechniqueEvidence(
        name="Incremental Skill Building",
        short_citation="Belland et al. (2017)",
        full_citation=(
            "Belland, B. R., Walker, A. E., Kim, N. J., & Lefler, M. (2017). "
            "Synthesizing results from empirical research on computer-based "
            "scaffolding in STEM education: A meta-analysis. "
            "Review of Educational Research, 87(2), 309–344."
        ),
        effect_size=0.46,
        effect_metric="Hedges' g",
        study_count=144,
        evidence_rating="moderate-high",
        summary=(
            "Scaffolded, progressively more complex tasks produce reliable "
            "achievement gains, especially in STEM skill domains. Rooted in "
            "Vygotsky's zone of proximal development."
        ),
    ),
    "Learn-By-Building": TechniqueEvidence(
        name="Learn-By-Building",
        short_citation="Papert (1991)",
        full_citation=(
            "Papert, S. (1991). Situating constructionism. In I. Harel & S. "
            "Papert (Eds.), Constructionism (pp. 1–11). Ablex Publishing."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate",
        summary=(
            "Constructionism: learners construct knowledge most effectively "
            "while constructing tangible artifacts. Strong qualitative support "
            "in design-oriented and computing education."
        ),
    ),
    "Work-Along & Solving": TechniqueEvidence(
        name="Work-Along & Solving",
        short_citation="Collins, Brown, & Newman (1989)",
        full_citation=(
            "Collins, A., Brown, J. S., & Newman, S. E. (1989). Cognitive "
            "apprenticeship: Teaching the crafts of reading, writing, and "
            "mathematics. In L. B. Resnick (Ed.), Knowing, Learning, and "
            "Instruction (pp. 453–494). Lawrence Erlbaum Associates."
        ),
        effect_size=None,
        effect_metric="",
        study_count=None,
        evidence_rating="moderate",
        summary=(
            "Cognitive apprenticeship: expert demonstrates, learner attempts "
            "with coaching, then performs independently. Strong tradition in "
            "applied-skill education."
        ),
    ),
    # ── New techniques (2023–2025 evidence expansion) ────────────────────────
    "Interleaved Practice": TechniqueEvidence(
        name="Interleaved Practice",
        short_citation="Brunmair & Richter (2019)",
        full_citation=(
            "Brunmair, M., & Richter, T. (2019). Similarity matters: A "
            "meta-analysis of interleaved learning and its moderating variables. "
            "Psychological Bulletin, 145(11), 1029–1052."
        ),
        effect_size=0.42,
        effect_metric="Hedges' g",
        study_count=54,
        evidence_rating="moderate-high",
        summary=(
            "Mixing different problem types or subjects within a single study "
            "session—rather than blocking by category—improves discrimination "
            "learning and long-term transfer. Effects are largest for inductive "
            "category learning and mathematics problem solving."
        ),
        dunlosky_utility="moderate",
    ),
    "Retrieval + Elaboration": TechniqueEvidence(
        name="Retrieval + Elaboration",
        short_citation="Yang et al. (2021)",
        full_citation=(
            "Yang, C., Luo, L., Vadillo, M. A., Yu, R., & Shanks, D. R. "
            "(2021). Testing (quizzing) boosts classroom learning: A systematic "
            "and meta-analytic review. "
            "Psychological Bulletin, 147(4), 399–435."
        ),
        effect_size=0.60,
        effect_metric="Cohen's d",
        study_count=48,
        evidence_rating="high",
        summary=(
            "Pairing retrieval practice with elaborative interrogation—prompting "
            "learners to explain why retrieved facts are true or how concepts "
            "interconnect—yields additive gains beyond retrieval alone. "
            "Particularly effective for conceptual transfer and classroom-level "
            "achievement across STEM and humanities."
        ),
        dunlosky_utility="high",
    ),
    "AI-Assisted Tutoring": TechniqueEvidence(
        name="AI-Assisted Tutoring",
        short_citation="Kulik & Fletcher (2016)",
        full_citation=(
            "Kulik, J. A., & Fletcher, J. D. (2016). Effectiveness of "
            "intelligent tutoring systems: A meta-analytic review. "
            "Review of Educational Research, 86(1), 42–78."
        ),
        effect_size=0.66,
        effect_metric="Cohen's d",
        study_count=50,
        evidence_rating="moderate-high",
        summary=(
            "Intelligent tutoring systems that provide adaptive hints, "
            "immediate corrective feedback, and individualized problem sequencing "
            "produce substantial gains over conventional classroom instruction. "
            "Emerging LLM-based tutors demonstrate early promise, but large-scale "
            "replication with controlled designs is still accumulating."
        ),
    ),
    "Mind Mapping": TechniqueEvidence(
        name="Mind Mapping",
        short_citation="Schroeder, Nesbit, Anguiano, & Adesope (2018)",
        full_citation=(
            "Schroeder, N. L., Nesbit, J. C., Anguiano, C. J., & Adesope, "
            "O. O. (2018). Studying and constructing concept maps: A "
            "meta-analysis. Educational Psychology Review, 30(2), 431–455."
        ),
        effect_size=0.58,
        effect_metric="Hedges' g",
        study_count=142,
        evidence_rating="moderate-high",
        summary=(
            "Generating radial mind maps—branching key ideas outward from a "
            "central topic—improves recall and comprehension beyond passive "
            "re-reading. Constructing maps produces larger benefits than studying "
            "instructor-provided maps; strongest gains in introductory and "
            "content-heavy courses."
        ),
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Course-type ↔ technique alignment (0-100 domain-fit percentages)
#
# Alignment values reflect how well the technique's empirical support extends
# to the target course type's subfield. They draw on Dunlosky 2013, the
# course-specific literature cited above, and the consensus of instructional
# design research.
# ──────────────────────────────────────────────────────────────────────────────

COURSE_TYPE_ALIGNMENT: Dict[str, Dict[str, int]] = {
    "Applied Calculation-Driven Learning": {
        "Worked Example Analysis": 95,
        "Active Recall": 88,
        "Spaced Repetition": 85,
        "Simulation & Visualization": 92,
        "Iterative Writing & Editing": 28,
        "Case Study Analysis": 25,
    },
    "Deep Conceptual Learning": {
        "Feynman Technique": 100,
        "Active Recall": 87,
        "Spaced Repetition": 84,
        "Conceptual Mapping": 90,
        "Project-Based Learning": 30,
        "Open-Ended Exploration": 35,
    },
    "Case-Based & Strategic Learning": {
        "Case Study Analysis": 96,
        "Experiential Learning": 92,
        "Comparative Analysis": 88,
        "First-Principles Thinking": 85,
        "Simulation & Visualization": 35,
        "Spaced Repetition": 30,
    },
    "Language & Communication-Based Learning": {
        "Immersive Practice": 96,
        "Active Recall & Shadowing": 92,
        "Iterative Writing & Editing": 90,
        "Storytelling Frameworks": 88,
        "Worked Example Analysis": 32,
        "Deliberate Practice": 30,
    },
    "Hands-On, Project-Based Learning": {
        "Incremental Skill Building": 96,
        "Project-Based Learning": 92,
        "Learn-By-Building": 90,
        "Work-Along & Solving": 88,
        "Spaced Repetition": 28,
        "Conceptual Mapping": 30,
    },
    "Seminar / Discussion-Based Learning": {
        # Explaining and defending ideas aloud is the core seminar act
        "Feynman Technique": 92,
        # Elaboration during retrieval maps tightly onto live discussion
        "Retrieval + Elaboration": 88,
        # Socratic dialogue is built on juxtaposing competing positions
        "Comparative Analysis": 86,
        # Case presentations and deliberation are seminar staples
        "Case Study Analysis": 82,
        # First-principles reasoning drives analytical discussion
        "First-Principles Thinking": 80,
        # Mind maps help students organise branching discussion threads
        "Mind Mapping": 68,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def get_evidence(technique: str) -> Optional[TechniqueEvidence]:
    """Return the evidence record for a technique, or None if unknown."""
    return RESEARCH.get(technique)


def compute_evidence_scores() -> Dict[str, Dict[str, float]]:
    """
    Compute the {course_type: {technique: evidence_score}} mapping from the
    research data and alignment matrix:

        score = EVIDENCE_WEIGHTS[evidence_rating] * alignment_pct

    Scores are bounded to [0, 100].
    """
    scores: Dict[str, Dict[str, float]] = {}
    for course_type, techniques in COURSE_TYPE_ALIGNMENT.items():
        scores[course_type] = {}
        for technique, alignment in techniques.items():
            evidence = RESEARCH.get(technique)
            if evidence is None:
                raise KeyError(
                    f"Technique '{technique}' referenced in alignment matrix "
                    f"but missing from RESEARCH database."
                )
            weight = EVIDENCE_WEIGHTS[evidence.evidence_rating]
            scores[course_type][technique] = round(weight * alignment, 2)
    return scores


def all_techniques() -> List[str]:
    """Sorted list of every technique in the research database."""
    return sorted(RESEARCH.keys())
