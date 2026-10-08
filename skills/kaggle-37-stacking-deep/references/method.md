# Chapter 37: Stacking Deep

The useful currency of an ensemble is complementary error. This activity holds targets fixed and supplies two small regression prediction vectors. Change the weight of model B and compare the blend’s error with both individual errors. Inspect the residual covariance alongside those scores: it describes whether deviations around each model’s average error tend to move together. It is not the same quantity as correlation between raw predictions, which can be high simply because both models track the target. Before adding another stacking level, ask where its training inputs came from and whether every reported evaluation prediction was held out from the complete fitted pipeline. Vary the second-model weight and inspect MSE and residual covariance. Blend performance depends on error magnitudes and covariance, not prediction correlation alone. These are constructed comparisons requiring independent validation. A final meta learner can learn from OOF inputs while its own fitted outputs remain in-sample. Describe both levels separately, and keep the score used to choose complexity apart from the score used to assess it.

This constructed activity illustrates the chapter topic. The revised chapter develops the full fitting and assessment boundaries.

## Worked example

How does residual covariance explain the error of a two-model blend?

Four fixed regression targets and two prediction sets define a blend; no meta learner is trained.

```python
parameter = 0.5
import json, math
assert 0<=parameter<=1
y=[0.,1.,2.,3.]; a=[0.4,0.8,2.4,2.8]; b=[-0.2,1.4,1.8,3.4]
ra=[x-t for x,t in zip(a,y)]; rb=[x-t for x,t in zip(b,y)]
ma=sum(ra)/4; mb=sum(rb)/4
cov=sum((x-ma)*(z-mb) for x,z in zip(ra,rb))/4
blend=[(1-parameter)*x+parameter*z for x,z in zip(a,b)]
mse=lambda p:sum((x-t)**2 for x,t in zip(p,y))/4
print(json.dumps({'weight':parameter,'model_a_mse':mse(a),'model_b_mse':mse(b),'blend_mse':mse(blend),'residual_covariance':cov}))

```

Blend performance depends on error magnitudes and covariance, not prediction correlation alone. These are constructed comparisons requiring independent validation.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
