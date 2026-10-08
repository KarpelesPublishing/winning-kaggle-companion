---
name: kaggle-23-sensor-and-har
description: "Apply Chapter 23, Sensors and HAR: The Cross-Subject Problem, to a Kaggle modeling project; explain the method, implement a scoped change and evaluate it on suitable held-out data."
---

# Sensors and HAR: The Cross-Subject Problem

Start from the reader's actual request and project. Explain the method when asked to learn it; inspect and edit their pipeline when asked to implement it. Read [the companion method](references/method.md) and consult [the conditional lesson records](references/lessons.json) for this chapter. The full chapter text is available in the book, not in this public bundle. The activity is one illustration, not complete chapter coverage. Select a lesson by its conditions, not its value grade alone.

Inspect person, recording-session and timestamp identifiers in the current sensor pipeline. Define the actual test holdout and split subjects before window construction; assert no identity or prohibited overlapping-window reuse. Establish a GBM baseline with fold-local scaling and a small temporal CNN baseline using identical subject folds. Test subject normalization only when its batch availability matches inference. Compare macro-F1, per-subject dispersion, confusion matrices and runtime. Reject a remedy if gains disappear on new subjects, if normalization needs unavailable future windows, or if compute exceeds the submission budget. Preserve the failing random-split result as diagnostic evidence rather than a model-selection score.

Use the project's metric direction, class order, prediction-time feature availability and validation population. Keep every target-derived operation inside the relevant training boundary. Compare with the incumbent using identical held-out rows. Run a small, bounded test, then record the hypothesis, changed files, seed, data and fold identity, scores and limitations. A training score does not establish an improvement. Treat constructed examples and reported competition results separately.

For additional fitting stages, consult [technical corrections](references/errata.md). Use the existing local runtime and reader-authorized budget. Submission, paid compute and external publication require a request for those actions. Do not claim an execution that did not occur.

[Chapter notebook](https://github.com/KarpelesPublishing/winning-kaggle-companion/blob/main/notebooks/23-sensor-and-har.ipynb) · [Book](https://karpeles.com/publishing/winning-kaggle-the-reproducible-way)

## Apply and close the decision

For a relevant lesson, retain its trigger, action, comparison, limits and source. Translate the action into the current project only after checking required inputs and prediction/label timing. Trace upstream fitted stages as well as the final estimator. Preserve an incumbent; measure paired eligible predictions, complete runtime/memory and affected subgroups. Record keep, reject or defer, with actual artifacts and source/lesson IDs. A source reporting no ablation does not support an invented gain. If a local result motivates new book guidance, record it as a proposed adaptation for editorial review; do not silently change the canonical source.
