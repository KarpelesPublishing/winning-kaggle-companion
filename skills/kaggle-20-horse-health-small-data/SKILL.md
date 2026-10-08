---
name: kaggle-20-horse-health-small-data
description: "Apply Chapter 20, Horse Health: Small Data, Big Discipline, to a Kaggle modeling project; explain the method, implement a scoped change and evaluate it on suitable held-out data."
---

# Horse Health: Small Data, Big Discipline

Start from the reader's actual request and project. Explain the method when asked to learn it; inspect and edit their pipeline when asked to implement it. Read [the companion method](references/method.md) and consult [the conditional lesson records](references/lessons.json) for this chapter. The full chapter text is available in the book, not in this public bundle. The activity is one illustration, not complete chapter coverage. Select a lesson by its conditions, not its value grade alone.

Verify the target is single-label and all classes are included in micro averaging. Map class identities consistently across every probability matrix. Fit preprocessing inside each training fold and confirm row sampling is actually enabled. Compare regularized baselines under the official metric. If tuning class-score multipliers, fit them inside the selection boundary and assess on outer predictions or the untouched local holdout. Report accuracy and micro-F1 identity, plus class-specific diagnostics where useful. Reject apparent gains measured only on the tuning labels. Label local and official test splits consistently and preserve uncertainty from repeated small-sample comparisons rather than assuming ten folds are automatically more reliable.

Use the project's metric direction, class order, prediction-time feature availability and validation population. Keep every target-derived operation inside the relevant training boundary. Compare with the incumbent using identical held-out rows. Run a small, bounded test, then record the hypothesis, changed files, seed, data and fold identity, scores and limitations. A training score does not establish an improvement. Treat constructed examples and reported competition results separately.

For additional fitting stages, consult [technical corrections](references/errata.md). Use the existing local runtime and reader-authorized budget. Submission, paid compute and external publication require a request for those actions. Do not claim an execution that did not occur.

[Chapter notebook](https://github.com/KarpelesPublishing/winning-kaggle-companion/blob/main/notebooks/20-horse-health-small-data.ipynb) · [Book](https://karpeles.com/publishing/winning-kaggle-the-reproducible-way)

## Apply and close the decision

For a relevant lesson, retain its trigger, action, comparison, limits and source. Translate the action into the current project only after checking required inputs and prediction/label timing. Trace upstream fitted stages as well as the final estimator. Preserve an incumbent; measure paired eligible predictions, complete runtime/memory and affected subgroups. Record keep, reject or defer, with actual artifacts and source/lesson IDs. A source reporting no ablation does not support an invented gain. If a local result motivates new book guidance, record it as a proposed adaptation for editorial review; do not silently change the canonical source.
