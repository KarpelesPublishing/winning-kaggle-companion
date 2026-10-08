# Chapter 29: Advanced Feature Selection

Feature importance needs a comparison point before it can help you prune. This activity supplies three invented real importances and five null values for each feature. A null value represents the kind of importance that might arise after breaking the target relationship. Move the percentile cutoff and identify which features survive. Notice that equal raw importance would not imply equal evidence: each feature has its own null behavior. Then ask whether your data permit arbitrary target shuffling. Time and grouped observations may require a different permutation scheme. The cutoff is a screening choice, followed by an ablation in data that did not determine the feature set. Raise the empirical percentile cutoff and inspect each feature against its own null distribution. Survival nominates a feature for an independent ablation; it does not control multiple-testing error or certify signal. Check the cost of removing a correlated substitute as well as an isolated weak feature. Low individual importance can coexist with redundancy, and pruning several columns at once can change which surviving feature becomes useful.

## Worked example

Which features survive a heuristic null-importance cutoff?

Fixed invented null importance values illustrate a diagnostic, not fitted model importances or statistical proof.

```python
parameter = 75
import json, math
assert 50<=parameter<=95
nulls=[[2,3,4,5,6],[8,9,10,11,12],[1,2,2,3,4]]
real=[8,10,3]
def quantile(a,p):
    v=(len(a)-1)*p/100; i=int(v); j=min(i+1,len(a)-1)
    return a[i]+(a[j]-a[i])*(v-i)
table=[]
for i,(a,r) in enumerate(zip(nulls,real)):
    c=quantile(a,parameter)
    table.append({'feature':i,'real_importance':r,'null_cutoff':c,'keep':int(r>c)})
print(json.dumps({'percentile':parameter,'table':table,'kept_count':sum(r['keep'] for r in table)}))

```

Survival nominates a feature for an independent ablation; it does not control multiple-testing error or certify signal.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
