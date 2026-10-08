# Chapter 4: Tracking the CV-LB Gap

A gap is a difference between scores, not a diagnosis by itself. Public observations and local folds differ, so a negative gap on a loss can arise when nothing useful changes. Blending can also improve predictions, but that improvement needs evidence beyond the sign of the gap.

Change the public adjustment here. Both scenarios receive the same gap. One has lower paired local loss than the baseline; the other keeps the baseline's errors. Read the local-improvement column before deciding which scenario supports a blending benefit. Holding the public adjustment equal separates arithmetic from the explanation attached to it.

For your campaign, retain predictions on identical validation rows and compare errors by row, group or season. Report variation alongside the aggregate direction. Examine the public score as another observation with its own uncertainty. If signals disagree, record competing explanations and the check that could separate them. A stable gap can support confidence in a workflow, but cannot certify every feature or identify the cause of the next leaderboard movement. Choosing a champion requires the complete evidence record.

## Worked example

Can identical gaps imply different mechanisms?

Constructed local errors and identical public adjustment.

```python
parameter = 0.002
import json
assert 0 <= parameter <= 0.01
base=[.04,.01,.04,.01];blend=[.025,.015,.025,.015];mean=lambda a:sum(a)/len(a)
rows=[{'scenario':name,'cv':mean(a),'lb':mean(a)-parameter,'gap':round(-parameter,8),'paired_local_improvement':round(mean(base)-mean(a),8)} for name,a in [('blend',blend),('unchanged',base)]]
assert rows[0]['gap']==rows[1]['gap']
result={'rows':rows}
print(json.dumps(result))

```

Gap sign cannot identify causal mechanism.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
