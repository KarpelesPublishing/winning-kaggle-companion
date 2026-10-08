# Chapter 44: Loss Functions for CV

Focal loss changes how training effort is allocated across examples. The activity supplies three probabilities assigned to the true class: 0.1, 0.5 and 0.9. The first is hard for the model, the last easy. Increase gamma and compare ordinary cross-entropy with focal loss on each row. Notice that every loss falls, but the easy example’s loss falls much more sharply. At gamma zero the two losses coincide. This boundary is also a useful implementation test. A smaller displayed loss does not mean the model is better: it means you changed the objective. Keep an untouched evaluation metric and investigate whether hard rows represent useful minority signal or mislabeled examples. Increase gamma in -(1-p_t)^gamma*log(p_t) and compare easy/hard losses. At gamma zero focal loss equals CE; positive gamma suppresses easy cases more. Loss values alone do not verify a custom gradient or improved learning. Some hard rows are mislabeled rather than informative rare events. Focusing more on them can amplify noise. Inspect those errors and their provenance before treating increasing gamma as an automatic response to imbalance.

## Worked example

How does focal gamma change the relative weight of easy and hard examples?

Three fixed true-class probabilities define per-example CE and focal losses. No custom gradient or model training is performed.

```python
parameter = 2
import json, math
assert 0<=parameter<=4
probs=[0.1,0.5,0.9]
table=[{'true_class_probability':p,'cross_entropy':-math.log(p),'focal_loss':-(1-p)**parameter*math.log(p)} for p in probs]
if parameter==0: assert all(abs(r['cross_entropy']-r['focal_loss'])<1e-12 for r in table)
assert all(r['focal_loss']<=r['cross_entropy']+1e-12 for r in table)
print(json.dumps({'gamma':parameter,'table':table,'easy_to_hard_loss_ratio':table[2]['focal_loss']/table[0]['focal_loss']}))

```

At gamma zero focal loss equals CE; positive gamma suppresses easy cases more. Loss values alone do not verify a custom gradient or improved learning.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
