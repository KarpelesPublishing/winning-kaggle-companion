# Chapter 15: When the Problem Isn't Tabular

**How good must a reviewer be before reviewing every answer beats the base model, and does routing change that?**

The chapter asks for the base answer, the reviewer policy and any uncertainty routing to be compared on untouched labelled questions, counting the errors the reviewer introduces and the calls it costs.

## The experiment

A constructed five-answer task. A logistic regression base model and a logistic regression reviewer each fit a different noisy view of the same question (the base view has fixed noise, the reviewer's noise is the control). Policies are scored on 6,000 held-out questions: base only; review every question and accept the reviewer's answer when it differs; and review only questions whose base confidence is below a threshold chosen on 1,000 development questions. Thirty independent question sets are averaged.

Control: Reviewer's view noise (larger means a weaker reviewer) (2.6 (reviewer alone about 0.45), 1.9 (about 0.55), 1.4 (about 0.66), 0.9 (about 0.84); default 1.9).

## Measured results

| Measure | 2.6 (reviewer alone about 0.45) | 1.9 (about 0.55) | 1.4 (about 0.66) | 0.9 (about 0.84) |
|---|---|---|---|---|
| Base only | 0.574 | 0.574 | 0.574 | 0.574 |
| Review all | 0.447 | 0.545 | 0.661 | 0.839 |
| Routed | 0.596 | 0.636 | 0.706 | 0.848 |
| Reviewer alone | 0.447 | 0.545 | 0.661 | 0.839 |
| Routed calls | 32% | 51% | 69% | 87% |

## What the result says (default, reviewer's view noise (larger means a weaker reviewer) = 1.9)

With reviewer noise 1.9 the reviewer alone scores 0.545 against 0.574 for the base model. Reviewing everything scores 0.545: it fixes 23.4 and overturns 26.2 correct answers per 100 questions, so 0.545 - 0.574 = -0.029. Routing the 51% least confident questions scores 0.636, +0.062 against the base, and beat the base in 100% of the 30 question sets.

- Review all minus base: 0.545 - 0.574 = -0.029 accuracy.
- Routed minus base: 0.636 - 0.574 = 0.062 accuracy.
- Routed, per 100 questions: 16.0 fixed - 9.7 overturned = 6.3.
- On the routed questions the base is right 0.426 of the time and the reviewer 0.547.

## Apply it to a competition

- Score the base answer, the review-everything policy and the routed policy on the same untouched labelled questions.
- Count the correct answers each policy overturned next to the wrong answers it fixed; a policy is judged by the net.
- Pick the routing threshold on development questions, and log the reviewer calls so the cost is part of the comparison.
- Accept a reviewer's correction only when it is valid under the answer schema, otherwise keep the base answer.

## Assumptions and limits

Constructed data with two logistic regressions standing in for a multimodal base model and a reviewer. The reviewer's errors are independent of the base model's, and it answers from its own view rather than judging the base answer; a real reviewer may anchor on the answer it is shown, or return text that is not a valid answer. Confidence here is a real model probability, and the schema-validity check the chapter describes is not modelled.

Constructed data; the reviewer is independent of the base model and answers from its own view, which makes the benefit of routing cleaner than it would be with a reviewer that anchors on the shown answer.

## Reproduce it

The chapter notebook `notebooks/15-when-the-problem-isnt-tabular.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch15` (`run`, `explain`, `draw`).

```python
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
```

Book location: Chapter 15, Visual Question Answering: Autopilot 2026. Constructed example: seeded synthetic questions and two logistic regressions, measured by the chapter activity.
