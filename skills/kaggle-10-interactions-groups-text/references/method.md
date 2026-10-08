# Chapter 10: Interactions, Groups, and Text

A reusable search helper must know whether larger or smaller numbers mean improvement. AUC usually increases; RMSE decreases. One shared acceptance inequality can keep precisely the regression features you intended to reject.

Change the candidate RMSE delta. Negative values improve each illustrative fold; positive values worsen them. Compare erroneous and correct acceptance flags and verify mean improvement by hand. At zero, neither accepts the candidate. The activity isolates direction without fitting hundreds of models.

Before an interaction search, record metric name, direction, fixed split identities and baseline predictions. Inspect paired fold changes to distinguish one unusual fold from a consistent pattern. Repeatedly screening candidates on the same folds also tunes to those folds. Reserve an untouched comparison or use an outer evaluation when a large search shapes your conclusion. This gives a correct screening rule and validation plan, not proof that stepwise selection eliminates leakage or multiple-testing optimism. Ratios, group aggregates and text reductions also need explicit availability and fit boundaries before their score comparisons become meaningful.

## Worked example

Respect the metric direction in feature search.

Paired constructed RMSE; smaller is better.

```python
parameter = -0.02
import json
assert -0.1 <= parameter <= 0.1
base=[.4,.5,.6];candidate=[v+parameter for v in base];b=sum(base)/3;c=sum(candidate)/3
result={'baseline_rmse':b,'candidate_rmse':c,'improvement':b-c,'wrong_acceptance':int(c>b+1e-5),'correct_acceptance':int(c<b-1e-5),'paired_fold_deltas':[x-y for x,y in zip(candidate,base)]}
print(json.dumps(result))

```

Correct screening is necessary but repeated CV reuse remains adaptive.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
