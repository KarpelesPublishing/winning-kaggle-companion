# Chapter 13: Hyperparameter Tuning with Optuna

Hyperparameter search selects the candidate that looks best on the data it repeatedly examines. That selected score also contains the benefit of selection. It is not automatically an independent estimate of future performance, even if each candidate used cross-validation.

Increase the number of examined trials in this fixed hypothetical record. The chosen search score can improve while its audit score stays flat or worsens. The audit scores are revealed after selection and never used to choose a trial. Follow the chosen trial ID to see that the candidate winning one comparison need not win the other.

For your own study, define search space, metric direction, split identities and budget before running. Keep an assessment outside the adaptive selection loop, or use nested evaluation when your evidence requirements justify it. Record pruned trials and the step at which pruning occurs. After selecting parameters, apply the frozen training recipe to the independent assessment once. If that result is worse, inspect selection optimism and split mismatch instead of extending trials until the audit also looks better. Retraining on all data requires a documented iteration plan; neither a universal floor nor a multiplier proves the correct model size.

## Worked example

Separate search best from independent assessment.

Fixed hypothetical candidate scores; audit not used for selection.

```python
parameter = 4
import json
assert 1 <= parameter <= 8
assert int(parameter) == parameter
candidates=[{'trial':i+1,'search_score':s,'audit_score':a} for i,(s,a) in enumerate([(.880,.879),(.884,.880),(.882,.881),(.887,.878),(.885,.882),(.889,.879),(.886,.883),(.890,.880)])][:int(parameter)]
selected=max(candidates,key=lambda r:r['search_score'])
result={'trials_examined':len(candidates),'chosen_trial':selected['trial'],'chosen_search_score':selected['search_score'],'chosen_audit_score':selected['audit_score'],'search_minus_audit':selected['search_score']-selected['audit_score']}
print(json.dumps(result))

```

More search can improve search best without improving audit performance.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
