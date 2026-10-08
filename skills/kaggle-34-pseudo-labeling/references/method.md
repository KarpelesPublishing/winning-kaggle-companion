# Chapter 34: Pseudo-Labeling

Teacher selection is a dependency question before it is a confidence-threshold question. Three folds here contain two labeled rows each. Teacher zero trains on folds one and two; teacher one trains on zero and two; teacher two trains on zero and one. Select the fold you intend to evaluate. Audit which teacher training sets overlap its labels. The manuscript’s displayed selection excludes the matching teacher, leaving teachers that saw the validation labels. The corrected selection uses the teacher trained without that fold. Trace this path before producing any pseudo-labels, and keep teacher early stopping, calibration and later pseudo-label rounds inside the same outer training boundary. Select the current validation fold and audit teacher-training intersections for the manuscript teacher selection versus the corrected selection. For outer fold k, teacher k excludes validation k. Excluding teacher k instead selects precisely the teachers that saw those labels. Confidence filtering cannot repair the teacher-selection error. Audit the label dependency first; only then consider thresholding, pseudo-row caps and class distribution. Otherwise an improved validation score can reward information you intended to exclude.

This constructed activity illustrates the chapter topic. The revised chapter develops the full fitting and assessment boundaries.

## Worked example

Which teacher excludes every validation label for the current fold?

Three folds have distinct numeric row IDs. Teachers train on the complement of their own held-out fold.

```python
parameter = 0
import json, math
assert parameter in (0,1,2)
folds=[{0,1},{2,3},{4,5}]; universe=set.union(*folds)
teachers=[universe-f for f in folds]; k=int(parameter)
bad=[i for i in range(3) if i!=k]; good=[k]
bad_overlap=len(set.union(*(teachers[i] for i in bad))&folds[k])
good_overlap=len(teachers[k]&folds[k])
assert good_overlap==0 and bad_overlap==len(folds[k])
print(json.dumps({'validation_fold':k,'manuscript_teacher_ids':bad,'corrected_teacher_ids':good,'manuscript_validation_label_overlap':bad_overlap,'corrected_validation_label_overlap':good_overlap}))

```

For outer fold k, teacher k excludes validation k. Excluding teacher k instead selects precisely the teachers that saw those labels.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
