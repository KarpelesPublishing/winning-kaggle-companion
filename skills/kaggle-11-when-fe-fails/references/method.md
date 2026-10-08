# Chapter 11: When Feature Engineering Fails

Reduction can retain almost all variance while discarding the target distinction. Variance measures movement across examples. Prediction depends on which movements distinguish outcomes. Those questions can align, but need not.

Change the signal scale. The nuisance direction stays at plus or minus ten, while a smaller direction determines the labels. Compare variances and sign-rule accuracy. The nuisance accounts for nearly all variance yet predicts half the labels correctly. The small coordinate preserves the target distinction at every positive scale. Here the centered orthogonal axes make the contrast transparent without an eigensolver.

Question explained-variance cutoffs in your embedding pipeline. Compare raw embeddings, several reductions and a supervised head on identical held-out rows. Fit learned reductions within each fold and record runtime with the metric. Useful compression is a tradeoff. This does not show PCA always fails or SVD always succeeds. It shows why variance alone cannot certify predictive information survived. To explain a historical failure causally, retain the actual representation and compare the proposed mechanism with alternative ablations instead of inferring it from one score drop.

## Worked example

Separate variance from label information.

Centered orthogonal axes; small axis determines labels.

```python
parameter = 0.1
import json
assert 0.01 <= parameter <= 1
x=[-10.,-10.,10.,10.];z=[-parameter,parameter,-parameter,parameter];y=[0,1,0,1];var=lambda a:sum(v*v for v in a)/4;acc=lambda a:sum(int(v>0)==t for v,t in zip(a,y))/4
assert var(x)>var(z)
result={'nuisance_variance':var(x),'signal_variance':var(z),'nuisance_only_accuracy':acc(x),'signal_only_accuracy':acc(z),'nuisance_variance_fraction':var(x)/(var(x)+var(z))}
print(json.dumps(result))

```

Variance retained is not task information retained.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
