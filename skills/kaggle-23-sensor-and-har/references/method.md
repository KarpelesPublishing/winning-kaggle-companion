# Chapter 23: Sensors and HAR: The Cross-Subject Problem

A sensor row is a window, but the person producing it may be the real unit of generalization. Imagine four people contributing three windows each. Select the interleaved split and count how many identities appear on both sides. Then select the subject holdout and repeat the count. Neither calculation trains a classifier: the purpose is to make the evaluation boundary visible before a model can hide it behind a reassuring score. Ask what your competition holds out: people, recording sessions, future time, or some combination. A split can exclude the target row while still retaining almost identical neighboring windows. Record both checks separately. The control chooses an interleaved row split or a subject-held-out split. The table counts shared identities, which directly tests the structural promise. Zero shared subjects establishes identity separation only; it cannot establish matching activity prevalence or an accurate leaderboard estimate. If test subjects are unseen, record how many subjects support each activity and whether a held-out person can lack a class. This may change the split and metric aggregation even after identity overlap reaches zero.

## Worked example

How many subjects cross your train/validation boundary?

Constructed 12 windows belong to four subjects. Window leakage and subject leakage are different checks.

```python
parameter = 1
import json, math
assert parameter in (0,1)
groups=[s for s in range(4) for _ in range(3)]
val=[i for i,g in enumerate(groups) if (g==3 if parameter else i%3==0)]
train=[i for i in range(12) if i not in val]
shared=set(groups[i] for i in train)&set(groups[i] for i in val)
if parameter: assert not shared
print(json.dumps({'train_windows':len(train),'validation_windows':len(val),'shared_subject_count':len(shared),'validation_subjects':len(set(groups[i] for i in val))}))

```

Zero shared subjects establishes identity separation only; it cannot establish matching activity prevalence or an accurate leaderboard estimate.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
