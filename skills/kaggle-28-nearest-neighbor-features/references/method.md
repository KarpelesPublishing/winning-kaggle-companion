# Chapter 28: Nearest-Neighbor Features

A neighborhood feature answers a local question using a clearly defined set of eligible donors. The query here is at position 0.3 and has no label. Five labeled training points lie along one common numeric axis. Increase K from one to five and inspect both the target mean and the average distance. The estimate becomes less local as distant rows enter. Explain whether that change reduces noise or erases a useful boundary; the table alone cannot decide. In a real fold, the eligible set is the training portion available to that prediction, including every earlier transformation used to define distance. A rowwise OOF feature is not automatically safe for a second model’s evaluation. Choose K and inspect the mean target and mean distance among the nearest eligible points. Large K smooths the local summary. This does not establish predictive gain; eligible donors must also exclude outer validation labels. When K grows, inspect whether distant neighbors cross an important domain boundary. Geographic proximity, product similarity and standardized Euclidean distance need not describe the same neighborhood or support the same local target average.

## Worked example

How does neighborhood size change a local target summary without self-inclusion?

The query is unlabeled; five fixed training points are eligible donors. One-dimensional distances use a common scale.

```python
parameter = 2
import json, math
assert parameter in range(1,6)
points=[(0.0,0),(0.2,0),(0.8,1),(1.0,1),(1.5,1)]
query=0.3
nbr=sorted([(abs(x-query),i,y) for i,(x,y) in enumerate(points)])[:int(parameter)]
assert len({i for _,i,_ in nbr})==parameter
print(json.dumps({'k':int(parameter),'neighbor_ids':[i for _,i,_ in nbr],'target_mean':sum(y for _,_,y in nbr)/parameter,'mean_distance':sum(d for d,_,_ in nbr)/parameter}))

```

Large K smooths the local summary. This does not establish predictive gain; eligible donors must also exclude outer validation labels.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
