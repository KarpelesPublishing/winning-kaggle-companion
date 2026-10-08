# Chapter 17: Stacking and Hill-Climbing

Stacking learns a combination of model predictions. Regularization limits how strongly the fitted combination follows the training meta-data. A ridge penalty shrinks coefficients toward zero; with correlated predictors, individual coefficient interpretations also depend on scaling and the other inputs.

Change alpha in the centered one-feature illustration. Inspect the coefficient, predictions, training MSE and penalty. Increasing alpha shrinks the coefficient and typically sacrifices training fit. That tradeoff does not tell you which alpha generalizes best: the displayed errors are measured on the same examples used to fit the coefficient.

In your actual ensemble, collect first-level predictions without each observation's label influencing its own training path. To estimate the complete stack honestly, also ensure first-level features used for meta training do not depend on the meta assessment labels. A separate outer evaluation can enclose first-level fitting, preprocessing and meta selection. Fit the selected meta model only after its choices are fixed, and compare the complete stack with a simple blend on untouched data. Inspect nonfinite inputs directly; do not mask pipeline faults merely because a numerical sanitizer makes fitting continue.

## Worked example

Fit a regularized meta coefficient.

One centered meta feature; squared loss and L2 penalty. Training error is not an independent stacked-model estimate.

```python
parameter = 1
import json
assert 0 <= parameter <= 10
x=[-.4,-.2,.2,.4];y=[-.5,-.1,.1,.5];xy=sum(a*b for a,b in zip(x,y));xx=sum(a*a for a in x);beta=xy/(xx+parameter);pred=[beta*v for v in x];mse=sum((a-b)**2 for a,b in zip(y,pred))/4
assert abs(beta)<=abs(xy/xx)+1e-12
result={'ridge_coefficient':beta,'training_mse':mse,'predictions':pred,'penalty':parameter*beta*beta}
print(json.dumps(result))

```

Larger alpha shrinks coefficient; evaluate separately to choose alpha.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
