# Chapter 33: Data Augmentation for Tabular Data

Interpolation creates a mathematical training example, so ask exactly which properties it preserves. Row A contains income ten and category indicators [1, 0]; row B contains income thirty and [0, 1]. Change the mixing weight and inspect both the numeric feature and indicators. The income always stays between the original values. The indicators still sum to one, but most mixed rows no longer describe a single observed category. The soft target also sums to one and must be consumed by a loss that supports probability targets. Write one joint constraint from your own dataset and test whether interpolation respects it before copying an augmentation loop. Move the interpolation weight and inspect feature values, a one-hot-validity flag and the soft class target. Nonnegativity survives convex mixing, but a fractional indicator vector is no longer an observed discrete category. Decide whether this mathematical regularizer is appropriate. A fractional category can be a deliberate regularizer without representing an observed customer. Make that choice explicit and ensure the training loss consumes a normalized probability target rather than treating class numbers as measurements.

## Worked example

What does Mixup preserve, and which joint constraints can it violate?

Two nonnegative rows have mutually exclusive category indicators and different class targets.

```python
parameter = 0.4
import json, math
assert 0<=parameter<=1
a=[10.,1.,0.]; b=[30.,0.,1.]
mixed=[parameter*x+(1-parameter)*z for x,z in zip(a,b)]
soft=[parameter,1-parameter]
assert all(x>=0 for x in mixed) and abs(sum(soft)-1)<1e-12
onehot=int(mixed[1:] in ([1.,0.],[0.,1.]))
print(json.dumps({'mix_weight':parameter,'features':mixed,'soft_target':soft,'observed_one_hot_category':onehot}))

```

Nonnegativity survives convex mixing, but a fractional indicator vector is no longer an observed discrete category. Decide whether this mathematical regularizer is appropriate.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
