---
name: kaggle-14-what-doesnt-work
description: "Apply Chapter 14, What Doesn't Work, to a Kaggle modeling project; explain the method, implement a scoped change and evaluate it on suitable held-out data."
---

# What Doesn't Work

Start from the reader's actual request and project. Explain the method when asked to learn it; inspect and edit their pipeline when asked to implement it. Read [the companion method](references/method.md) and consult [the conditional lesson records](references/lessons.json) for this chapter. The full chapter text is available in the book, not in this public bundle. The activity is one illustration, not complete chapter coverage. Select a lesson by its conditions, not its value grade alone.

Classify each discouraged technique as an observed case, a compute tradeoff or an invalid information dependency before applying the chapter's prohibition. For pseudo-labeling, trace teacher training IDs and labels for each outer fold. Generate pseudo labels from that fold's training-only teacher, cap their influence, retain confidence and provenance, and assess the student only on genuine held-out labels. Compare against no-pseudo training using the same splits. Reject gains relying on validation-trained teachers or pseudo-label-based scoring. Inspect class balance and calibration before raising confidence thresholds. For other techniques, preserve measured negative conditions rather than treating a few failures as universal evidence.

Use the project's metric direction, class order, prediction-time feature availability and validation population. Keep every target-derived operation inside the relevant training boundary. Compare with the incumbent using identical held-out rows. Run a small, bounded test, then record the hypothesis, changed files, seed, data and fold identity, scores and limitations. A training score does not establish an improvement. Treat constructed examples and reported competition results separately.

For additional fitting stages, consult [technical corrections](references/errata.md). Use the existing local runtime and reader-authorized budget. Submission, paid compute and external publication require a request for those actions. Do not claim an execution that did not occur.

[Chapter notebook](https://github.com/KarpelesPublishing/winning-kaggle-companion/blob/main/notebooks/14-what-doesnt-work.ipynb) · [Book](https://karpeles.com/publishing/winning-kaggle-the-reproducible-way)

## Apply and close the decision

For a relevant lesson, retain its trigger, action, comparison, limits and source. Translate the action into the current project only after checking required inputs and prediction/label timing. Trace upstream fitted stages as well as the final estimator. Preserve an incumbent; measure paired eligible predictions, complete runtime/memory and affected subgroups. Record keep, reject or defer, with actual artifacts and source/lesson IDs. A source reporting no ablation does not support an invented gain. If a local result motivates new book guidance, record it as a proposed adaptation for editorial review; do not silently change the canonical source.
