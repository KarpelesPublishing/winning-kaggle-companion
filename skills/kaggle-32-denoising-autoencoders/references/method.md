# Chapter 32: Denoising Autoencoders for Tabular Pretraining

Swap noise changes a cell using a value that already occurs in the same column. The six constructed rows here obey a perfect relationship: the second feature is twice the first. Increase the number of swapped cells and compare the corrupted rows with the originals. Each replacement remains a plausible second-column value, yet it may become implausible beside that row’s first feature. The relation error makes the distinction between marginal validity and joint structure visible. A denoising model would receive corrupted inputs and try to reconstruct the untouched rows. Decide whether your real features share useful relationships before assuming that good reconstruction will help the supervised task. Increase the number of second-column cells replaced by donor values and compare reconstruction inconsistency. Every swapped value remains column-valid while joint row structure deteriorates. That tension motivates denoising rather than proving downstream gain. Notice that corruption targets remain clean even when input cells change. If your real columns have almost no useful dependence, reconstructing a replaced cell may encourage a bland mean instead of a valuable representation.

## Worked example

What happens to cross-feature relationships when valid cells are swapped?

Six rows obey x2=2*x1. Fixed donor rotation is a deterministic illustration of swap corruption, not random DAE training.

```python
parameter = 2
import json, math
assert parameter in range(7)
original=[[float(i),2.0*i] for i in range(1,7)]
corrupted=[r[:] for r in original]
for i in range(int(parameter)): corrupted[i][1]=original[(i+2)%6][1]
assert all(r[1] in [x[1] for x in original] for r in corrupted)
error=sum((r[1]-2*r[0])**2 for r in corrupted)/6
print(json.dumps({'swapped_cells':int(parameter),'relation_mse':error,'original':original,'corrupted':corrupted}))

```

Every swapped value remains column-valid while joint row structure deteriorates. That tension motivates denoising rather than proving downstream gain.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
