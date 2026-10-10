"""Chapter 15: When the Problem Isn't Tabular. A second-model reviewer: base only, review everything, or review low-confidence answers."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from sklearn.linear_model import LogisticRegression

from kaggle_companion.activities._common import clean

SEED = 15
REPLICATES = 30            # independent constructed question sets; every estimate is their mean
CLASSES, DIM = 5, 6        # five possible answers, six features per view
BASE_NOISE = 1.7           # noise in the base model's view of the question
N_TRAIN, N_DEV, N_TEST = 1500, 1000, 6000


def generate(rng, n, means_base, means_review, review_noise):
    """Each question has one correct answer. The base model and the reviewer see different noisy views of it."""
    answer = rng.integers(0, CLASSES, n)
    base_view = means_base[answer] + rng.normal(0, BASE_NOISE, (n, DIM))
    review_view = means_review[answer] + rng.normal(0, review_noise, (n, DIM))
    return base_view, review_view, answer


def run(review_noise):
    keys = ["base", "review_all", "routed"]
    acc = {k: [] for k in keys}
    fixed = {k: [] for k in keys}
    introduced = {k: [] for k in keys}
    calls, reviewer_alone, subset_base, subset_review = [], [], [], []
    for rep in range(REPLICATES):
        rng = np.random.default_rng(SEED * 100 + rep)
        means_base, means_review = rng.normal(size=(CLASSES, DIM)), rng.normal(size=(CLASSES, DIM))
        train, dev, test = [generate(rng, n, means_base, means_review, review_noise) for n in (N_TRAIN, N_DEV, N_TEST)]
        base = LogisticRegression(max_iter=500).fit(train[0], train[2])
        reviewer = LogisticRegression(max_iter=500).fit(train[1], train[2])

        def answers(data):
            proba = base.predict_proba(data[0])
            return proba.argmax(axis=1), proba.max(axis=1), reviewer.predict(data[1])

        base_dev, conf_dev, review_dev = answers(dev)
        base_test, conf_test, review_test = answers(test)
        # Routed review: ask the reviewer only when the base model's confidence is below a threshold chosen on the development questions.
        thresholds = np.quantile(conf_dev, np.linspace(0, 1, 21))
        dev_accuracy = [np.mean(np.where(conf_dev < t, review_dev, base_dev) == dev[2]) for t in thresholds]
        threshold = thresholds[int(np.argmax(dev_accuracy))]
        y = test[2]
        policies = {"base": np.zeros(N_TEST, bool), "review_all": np.ones(N_TEST, bool), "routed": conf_test < threshold}
        for name, asked in policies.items():
            final = np.where(asked, review_test, base_test)       # a correction replaces the base answer; no correction keeps it
            acc[name].append(np.mean(final == y))
            fixed[name].append(np.mean((base_test != y) & (final == y)))
            introduced[name].append(np.mean((base_test == y) & (final != y)))
        asked = policies["routed"]
        calls.append(asked.mean())
        reviewer_alone.append(np.mean(review_test == y))
        subset_base.append(np.mean(base_test[asked] == y[asked]) if asked.any() else np.nan)
        subset_review.append(np.mean(review_test[asked] == y[asked]) if asked.any() else np.nan)
    mean = lambda v: float(np.nanmean(v))
    return clean({
        "review_noise": review_noise, "reviewer_alone": mean(reviewer_alone), "routed_calls": mean(calls),
        "accuracy": {k: mean(v) for k, v in acc.items()},
        "fixed": {k: mean(v) for k, v in fixed.items()}, "introduced": {k: mean(v) for k, v in introduced.items()},
        "routed_subset": {"base": mean(subset_base), "reviewer": mean(subset_review)},
        "share_beating_base": {"review_all": float(np.mean(np.array(acc["review_all"]) > np.array(acc["base"]))),
                               "routed": float(np.mean(np.array(acc["routed"]) > np.array(acc["base"])))},
        "replicates": REPLICATES, "test_questions": N_TEST,
    })
# notebook-end


SPEC = {
    "chapter": 15,
    "chapter_title": "When the Problem Isn't Tabular",
    "subtitle": "Evaluation discipline carries over: compare whole policies, including the errors a second model introduces.",
    "summary": ("A second-model reviewer can fix a base answer or break it. One demonstration measures three policies on held-out questions "
                "as the reviewer's quality changes: no review, review everything, and review only the low-confidence answers."),
    "title": "Review everything, or route only the doubtful answers",
    "question": "How good must a reviewer be before reviewing every answer beats the base model, and does routing change that?",
    "why": ("The chapter asks for the base answer, the reviewer policy and any uncertainty routing to be compared on untouched labelled "
            "questions, counting the errors the reviewer introduces and the calls it costs."),
    "method": ("A constructed five-answer task. A logistic regression base model and a logistic regression reviewer each fit a different noisy "
               "view of the same question (the base view has fixed noise, the reviewer's noise is the control). Policies are scored "
               "on 6,000 held-out questions: base only; review every question and accept the reviewer's answer when it differs; "
               "and review only questions whose base confidence is below a threshold chosen on 1,000 development questions. "
               "Thirty independent question sets are averaged."),
    "control": {"key": "review_noise", "label": "Reviewer's view noise (larger means a weaker reviewer)",
                "values": [2.6, 1.9, 1.4, 0.9], "default": 1.9,
                "value_labels": ["2.6 (reviewer alone about 0.45)", "1.9 (about 0.55)", "1.4 (about 0.66)", "0.9 (about 0.84)"]},
    "source_section": "Visual Question Answering: Autopilot 2026",
    "symbols": ("Fixed is the share of questions the base answered wrongly and the policy answers correctly, introduced the share the base "
                "answered correctly and the policy gets wrong; accuracy changes by fixed minus introduced."),
    "explanation": ("A reviewer that is independent of the base model is right about as often on a doubtful question as on any other, "
                    "while the base model is much less accurate on the questions it is unsure about. Reviewing everything therefore "
                    "overturns many correct base answers, and the net can be a loss. Routing only the doubtful questions to the reviewer "
                    "keeps a large share of the fixes while overturning fewer correct answers, with fewer calls. The threshold chosen on "
                    "development questions routes few questions to a weak reviewer and nearly all of them to a strong one."),
    "application": ("Compare base only, review everything and routed review on the same untouched labelled questions, with the number of "
                    "reviewer calls and the count of correct answers the reviewer overturned. Choose the confidence threshold on a "
                    "development set, not on the test questions."),
    "assumptions": ("Constructed data with two logistic regressions standing in for a multimodal base model and a reviewer. The reviewer's errors "
                    "are independent of the base model's, and it answers from its own view rather than judging the base answer; a real "
                    "reviewer may anchor on the answer it is shown, or return text that is not a valid answer. Confidence here is a real model "
                    "probability, and the schema-validity check the chapter describes is not modelled."),
    "prediction": "The reviewer alone is slightly less accurate than the base model (0.545 against 0.574). What does reviewing every question do to accuracy?",
    "prediction_options": ["Raises it, as a second opinion", "Leaves it about the same", "Lowers it by about 3 points"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "Review-all scores 0.545 against 0.574 for the base: it fixes 23.4 per 100 questions and breaks 26.2.",
        "incorrect": "Review-all scores 0.545 against 0.574 for the base: it fixes 23.4 per 100 questions but breaks 26.2.",
    },
    "check": "Routed review helps even though the reviewer is weaker overall. Why?",
    "answer": ("Among the questions it routes (51% of all questions), the base model is right only 0.426 of the time while the reviewer is right "
               "0.547 of the time, because the base model's low confidence marks the questions it is worst at. Elsewhere the base model is "
               "better, so routing leaves those answers alone."),
    "provenance": "Constructed example: seeded synthetic questions and two logistic regressions, measured by the chapter activity.",
    "apply": [
        "Score the base answer, the review-everything policy and the routed policy on the same untouched labelled questions.",
        "Count the correct answers each policy overturned next to the wrong answers it fixed; a policy is judged by the net.",
        "Pick the routing threshold on development questions, and log the reviewer calls so the cost is part of the comparison.",
        "Accept a reviewer's correction only when it is valid under the answer schema, otherwise keep the base answer.",
    ],
    "honesty": ("Constructed data; the reviewer is independent of the base model and answers from its own view, which makes the "
                "benefit of routing cleaner than it would be with a reviewer that anchors on the shown answer."),
}

EQUATIONS = [{"tex": r"\Delta\,\mathrm{accuracy} = \mathrm{fixed} - \mathrm{introduced}",
              "alt": "change in accuracy equals the share of answers fixed minus the share of answers introduced as errors",
              "basis": "The activity's own accounting for a reviewer policy (Visual Question Answering: Autopilot 2026); the chapter has no display equation."}]
NCOLS = 2
HEIGHT = 4.4
POLICIES = [("base", "Base only", COLORS["light"]), ("review_all", "Review all", COLORS["terracotta"]),
            ("routed", "Routed", COLORS["teal"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = range(len(POLICIES))
    values = [result["accuracy"][k] for k, _, _ in POLICIES]
    left.bar(xs, values, width=0.55, color=[c for _, _, c in POLICIES], edgecolor=COLORS["ink"], lw=0.6)
    for x, v in zip(xs, values):
        left.text(x, v + 0.006, fmt(v), ha="center", va="bottom", fontsize=10)
    left.axhline(result["reviewer_alone"], color=COLORS["gold"], ls=(0, (4, 3)), lw=1.4,
                 label=f"Reviewer alone: {fmt(result['reviewer_alone'])}")
    left.set_xticks(list(xs), [name for _, name, _ in POLICIES])
    lo = min(values + [result["reviewer_alone"]])
    left.set_ylim(max(0.0, lo - 0.12), max(values + [result["reviewer_alone"]]) + 0.1)
    left.set_ylabel("Accuracy on 6,000 held-out questions")
    left.set_xlabel("Policy")
    left.legend(loc="upper left", frameon=False, fontsize=10)

    keys = [("review_all", "Review all"), ("routed", f"Routed\n({round(100 * result['routed_calls'])}% of calls)")]
    for x, (k, _) in enumerate(keys):
        right.bar(x - 0.2, 100 * result["fixed"][k], width=0.35, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6,
                  label="Wrong answers fixed" if x == 0 else None)
        right.bar(x + 0.2, -100 * result["introduced"][k], width=0.35, color=COLORS["terracotta"], edgecolor=COLORS["ink"], lw=0.6,
                  label="Correct answers overturned" if x == 0 else None)
        right.text(x - 0.2, 100 * result["fixed"][k] + 1, fmt(100 * result["fixed"][k], 1), ha="center", va="bottom", fontsize=10)
        right.text(x + 0.2, -100 * result["introduced"][k] - 1, fmt(100 * result["introduced"][k], 1), ha="center", va="top", fontsize=10)
    right.axhline(0, color=COLORS["grey"], lw=0.8)
    right.set_xticks([0, 1], [n for _, n in keys])
    top = 100 * max(result["fixed"]["review_all"], result["fixed"]["routed"])
    bottom = 100 * max(result["introduced"]["review_all"], result["introduced"]["routed"])
    right.set_ylim(-bottom - 9, top + 16)
    right.set_ylabel("Questions per 100")
    right.set_xlabel("Policy")
    right.legend(loc="upper right", frameon=False, fontsize=10)


def diff(a, b):
    """Difference of the displayed (3 decimal) values, so the hand calculation in the text adds up."""
    return fmt(round(a, 3) - round(b, 3))


def explain(result, parameter):
    a, f, i = result["accuracy"], result["fixed"], result["introduced"]
    sub = result["routed_subset"]
    interpretation = (
        f"With reviewer noise {parameter} the reviewer alone scores {fmt(result['reviewer_alone'])} against {fmt(a['base'])} for the base model. "
        f"Reviewing everything scores {fmt(a['review_all'])}: it fixes {fmt(100 * f['review_all'], 1)} and overturns {fmt(100 * i['review_all'], 1)} correct "
        f"answers per 100 questions, so {fmt(a['review_all'])} - {fmt(a['base'])} = {diff(a['review_all'], a['base'])}. Routing the {round(100 * result['routed_calls'])}% "
        f"least confident questions scores {fmt(a['routed'])}, {signed(round(a['routed'], 3) - round(a['base'], 3))} against the base, and beat the base in "
        f"{round(100 * result['share_beating_base']['routed'])}% of the {result['replicates']} question sets.")
    steps = [
        f"Review all minus base: {fmt(a['review_all'])} - {fmt(a['base'])} = {diff(a['review_all'], a['base'])} accuracy.",
        f"Routed minus base: {fmt(a['routed'])} - {fmt(a['base'])} = {diff(a['routed'], a['base'])} accuracy.",
        f"Routed, per 100 questions: {fmt(100 * f['routed'], 1)} fixed - {fmt(100 * i['routed'], 1)} overturned = {fmt(100 * (f['routed'] - i['routed']), 1)}.",
        f"On the routed questions the base is right {fmt(sub['base'])} of the time and the reviewer {fmt(sub['reviewer'])}.",
    ]
    metrics = {"Base only": fmt(a["base"]), "Review all": fmt(a["review_all"]), "Routed": fmt(a["routed"]),
               "Reviewer alone": fmt(result["reviewer_alone"]), "Routed calls": f"{round(100 * result['routed_calls'])}%"}
    alt = (f"Left: accuracy of base only ({fmt(a['base'])}), review all ({fmt(a['review_all'])}) and routed review ({fmt(a['routed'])}) with reviewer "
           f"noise {parameter}. Right: wrong answers fixed and correct answers overturned per 100 questions for review all and routed review.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    order = sorted(results, reverse=True)           # weakest reviewer first
    alone = [results[v]["reviewer_alone"] for v in order]
    assert all(b > a for a, b in zip(alone, alone[1:])), "reviewer quality should rise as noise falls"
    for v, lo, hi in ((2.6, 0.40, 0.50), (1.9, 0.50, 0.60), (1.4, 0.61, 0.71), (0.9, 0.79, 0.89)):
        assert lo < results[v]["reviewer_alone"] < hi, f"value label for reviewer noise {v}"
    for v, res in results.items():
        a = res["accuracy"]
        assert a["routed"] > a["base"] + 0.01, f"routing should beat the base at {v}"
        assert a["routed"] >= a["review_all"] - 0.002, f"routing should be at least as good as review-all at {v}"
        assert res["share_beating_base"]["routed"] > 0.8, f"routing should beat the base in most question sets at {v}"
        assert res["routed_subset"]["reviewer"] > res["routed_subset"]["base"], f"reviewer should beat base on routed questions at {v}"
    weak, mid = results[1.9], results[1.4]
    assert weak["reviewer_alone"] < weak["accuracy"]["base"] and weak["accuracy"]["review_all"] < weak["accuracy"]["base"] - 0.01, \
        "a weaker reviewer reviewing everything should lose"
    assert results[2.6]["accuracy"]["review_all"] < results[2.6]["accuracy"]["base"] - 0.05
    assert results[0.9]["accuracy"]["review_all"] > results[0.9]["accuracy"]["base"] + 0.1, "a strong reviewer reviewing all should win"
    assert results[0.9]["routed_calls"] < 1.0 and results[2.6]["routed_calls"] < 0.5
    calls = [results[v]["routed_calls"] for v in order]
    assert all(b > a for a, b in zip(calls, calls[1:])), "explanation: a stronger reviewer is routed more questions"
    for v, res in results.items():
        assert res["introduced"]["routed"] < res["introduced"]["review_all"], f"routing overturns fewer correct answers at {v}"
    assert weak["introduced"]["review_all"] > weak["fixed"]["review_all"], "review-all breaks more than it fixes at 1.9"
    assert mid["accuracy"]["review_all"] > mid["accuracy"]["base"], "reviewer 0.66 should beat base 0.57 when reviewing all"
    assert fmt(weak["accuracy"]["review_all"]) == "0.545" and fmt(weak["accuracy"]["base"]) == "0.574", "prediction feedback numbers"
    assert fmt(100 * weak["fixed"]["review_all"], 1) == "23.4" and fmt(100 * weak["introduced"]["review_all"], 1) == "26.2"
    assert 2.5 < 100 * (weak["accuracy"]["base"] - weak["accuracy"]["review_all"]) < 3.5, "prediction option says about 3 points"
    assert round(100 * weak["routed_calls"]) == 51, "check answer numbers"
    assert fmt(weak["routed_subset"]["base"]) == "0.426" and fmt(weak["routed_subset"]["reviewer"]) == "0.547", "check answer numbers"
