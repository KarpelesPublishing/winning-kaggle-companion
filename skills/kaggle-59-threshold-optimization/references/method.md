# Chapter 59: Threshold Optimization

The chapter’s strongest point is that the final decision rule should match the score and that a fitted threshold has a distribution boundary. I can follow the binary grid search. I would reread the prior-adjustment function and the rare-class threshold example because their arithmetic contradicts the prose. The companion shows selection and audit separately rather than promising a free improvement. Apply the selected cutoff to two constructed eight-row probability/label tables. Compute true positives, false positives, false negatives, precision, recall, and F1 separately. Compute the best tuning-set threshold from its exact probability breakpoints, then evaluate that frozen choice on the audit set. The tuning-best cutoff need not be audit-best. Reporting a fitted tuning score as expected gain hides post-processing selection bias. The separate audit also distinguishes threshold improvement from probability calibration. Compare the tuning-best threshold against the default on the audit rows. Explain the difference using false-positive and false-negative counts, and keep the fitted tuning gain separate from independent evaluation.

## Worked example

How does a threshold selected on one labeled subset behave on a separate audit subset?

Two illustrative subsets have different error patterns. Audit labels never select the reported tuning-best threshold. Single-class positive F1, notmacro F1 or MAP.

```python
parameter = 0.5
import json
assert .05<=parameter<=.95
tune=[(.9,1),(.8,1),(.65,0),(.55,1),(.4,1),(.3,0),(.2,0),(.1,0)]
audit=[(.9,1),(.8,0),(.65,0),(.55,1),(.4,0),(.3,1),(.2,0),(.1,0)]
def stats(data,t):
 tp=sum(p>=t and y==1 for p,y in data); fp=sum(p>=t and y==0 for p,y in data); fn=sum(p<t and y==1 for p,y in data)
 f1=2*tp/max(1,2*tp+fp+fn)
 return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':f1}
candidates=sorted({0.0,1.0}|{p for p,y in tune})
best=max(candidates,key=lambda t:(stats(tune,t)['f1'],-t))
assert 0<=stats(tune,parameter)['f1']<=1
print(json.dumps({'cutoff':parameter,'tuning_best_cutoff':best,'tuning_best_f1':stats(tune,best)['f1'],'frozen_cutoff_audit_f1':stats(audit,best)['f1'],'table':[dict(subset=0,**stats(tune,parameter)),dict(subset=1,**stats(audit,parameter))]}))

```

The tuning-best cutoff need not be audit-best. Reporting a fitted tuning score as expected gain hides post-processing selection bias. The separate audit also distinguishes threshold improvement from probability calibration.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
