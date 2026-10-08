# Chapter 53: Financial Competition Patterns

I understand the chapter’s aim as protecting a model from regime and factor dependencies that may fail out of sample. The residualization helper clarifies what removing exposure actually means. I cannot apply the advice confidently until the scored target is identified. The companion uses two targets to make that dependence explicit rather than implying neutralization is always beneficial. Construct six cross-sectional factor exposures and predictions with a factor-driven component plus an orthogonal residual. Fit the one-factor projection analytically, subtract the chosen fraction of that projection, and compare exposure slope and MSE against two explicit targets: factor-neutral and factor-retaining. Neutralization reduces factor exposure and helps the neutral target in this construction. It can worsen the factor-retaining target. Shared exposure is not automatically leakage; correctness depends on the available information and scored target. Move the neutralization fraction to both endpoints. Choose which target matches a hypothetical score, then explain why the opposite target gives a different recommendation for the same predictions.

## Worked example

What does partial factor neutralization remove, and when can it remove predictive signal?

Synthetic single-date cross-section; both targets are known only for demonstration. Residual component is centered and orthogonal to the factor. The competition metric determines which target is relevant.

```python
parameter = 0.5
import json
assert 0<=parameter<=1
x=[-3,-2,-1,1,2,3]; residual=[1,-1,0,0,-1,1]
p=[2*a+b for a,b in zip(x,residual)]
assert sum(a*b for a,b in zip(x,residual))==0
slope=sum(a*b for a,b in zip(x,p))/sum(a*a for a in x)
q=[b-parameter*slope*a for a,b in zip(x,p)]
def mse(y): return sum((a-b)**2 for a,b in zip(q,y))/len(q)
exposure=sum(a*b for a,b in zip(x,q))/sum(a*a for a in x)
assert abs(exposure-2*(1-parameter))<1e-10
print(json.dumps({'fraction':parameter,'exposure_slope':exposure,'neutral_target_mse':mse(residual),'raw_target_mse':mse(p),'table':[{'factor':a,'prediction':b,'neutralized':c} for a,b,c in zip(x,p,q)]}))

```

Neutralization reduces factor exposure and helps the neutral target in this construction. It can worsen the factor-retaining target. Shared exposure is not automatically leakage; correctness depends on the available information and scored target.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
