# Chapter 5: Adversarial Validation

A domain classifier asks whether features distinguish train rows from test rows. It does not directly ask whether your target model will fail. A shifted feature can remain useful when its target relationship is stable. An undetected shift can still matter when the domain model is weak or conditional target behavior changes.

Assume the domain probabilities here are calibrated on held-out observations and domain sampling is balanced. Change one probability. Compare it with the corresponding odds, then inspect capped and normalized weights. Probabilities near one produce much larger odds, making support overlap and weight concentration consequential.

In your competition, identify what moved and whether test construction explains it. Do not remove a feature solely because it separates domains. Compare target ablations on an appropriate split. Before weighting, document stable conditional target behavior and obtain cross-fitted domain predictions. Inspect concentration and compare capped weighting with an unweighted baseline. This operation demonstrates neither an expected gain nor a requirement to force AUC to one half. Missing support means reweighting cannot manufacture relevant training examples.

## Worked example

Turn domain probabilities into qualified weights.

Calibrated cross-fitted domain probabilities; equal domain priors. Stable conditional target behavior and overlapping support.

```python
parameter = 0.7
import json
assert 0.05 <= parameter <= 0.95
probs=[.2,.4,float(parameter),.8];odds=[min(p/(1-p),10) for p in probs];avg=sum(odds)/4;weights=[v/avg for v in odds]
assert abs(sum(weights)/4-1)<1e-12
result={'probabilities':probs,'capped_density_ratios':odds,'normalized_weights':weights,'maximum_weight':max(weights)}
print(json.dumps(result))

```

Weights valid only under assumptions; no predicted improvement.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
