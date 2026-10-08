# Chapter 12: The GBM Trio

A robust loss changes how strongly large residuals affect a training update. Huber loss is quadratic near zero and linear beyond a transition delta. Linear loss does not mean every useful gradient vanishes: residual signs can still provide information about which predictions should move up or down.

Change delta and inspect losses and gradients for five residuals. Large positive and negative residuals have capped gradient magnitude with opposite signs. Small residuals retain a gradient proportional to their size. Moving delta changes both the transition and the relative influence of outliers. It does not create an automatic rule that a model must stop at zero iterations.

In an actual regression experiment, inspect residual scale, target units and the framework's exact loss definition. Compare squared error and a robust alternative on the same held-out rows using the competition metric. A dollar target and a standardized target need different interpretation of a numerical delta. If training stops immediately, check data, objective support and stopping behavior rather than assigning one cause from scale alone. Preserve predictions and diagnostics so the outcome can be reproduced across framework versions.

## Worked example

Inspect the effect of a Huber transition.

Mathematical Huber loss; not a specific CatBoost implementation.

```python
parameter = 2
import json
assert 0.5 <= parameter <= 10
residuals=[-8.,-2.,0.,2.,8.]
def huber(r,d):
 return .5*r*r if abs(r)<=d else d*(abs(r)-.5*d)
grad=[max(-parameter,min(parameter,r)) for r in residuals]
assert grad[0]<0<grad[-1]
result={'rows':[{'residual':r,'loss':huber(r,parameter),'gradient':g} for r,g in zip(residuals,grad)],'nonzero_gradient_count':sum(g!=0 for g in grad)}
print(json.dumps(result))

```

Linear-region gradients can still be nonzero and informative.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
