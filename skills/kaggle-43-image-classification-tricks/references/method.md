# Chapter 43: Image Classification Tricks

A smoothing coefficient is incomplete without its target convention. This activity has three classes, true class zero and prediction probabilities [0.7, 0.2, 0.1]. One convention keeps 1−epsilon on the true class and spreads epsilon over the wrong classes. Another mixes the hard target with a uniform distribution over all classes. Change epsilon and compare the targets and their cross-entropies. Both targets are valid probability vectors, but they assign different mass to the correct class. Explain why matching a reported coefficient does not necessarily reproduce the reported training objective. Record the exact target construction beside any Mixup, CutMix or smoothing configuration you test. Compare mass assigned only to incorrect classes with uniform mixing across all classes, then calculate their cross-entropies. The same epsilon names different targets under these conventions. Match the implementation you use rather than relying on the label alone. When the loss changes, the same logits can produce different displayed loss values without any different prediction. Use the competition metric to judge the trained model and save the smoothing convention with its checkpoint.

## Worked example

Do two label-smoothing conventions produce the same target?

Three classes, target class zero, fixed probabilities and a controlled smoothing coefficient.

```python
parameter = 0.1
import json, math
assert 0<=parameter<=0.3
p=[0.7,0.2,0.1]
wrong=[1-parameter,parameter/2,parameter/2]
uniform=[1-parameter+parameter/3,parameter/3,parameter/3]
ce=lambda target:-sum(t*math.log(q) for t,q in zip(target,p))
assert abs(sum(wrong)-1)<1e-12 and abs(sum(uniform)-1)<1e-12
print(json.dumps({'epsilon':parameter,'wrong_classes_only_target':wrong,'uniform_mix_target':uniform,'wrong_classes_only_ce':ce(wrong),'uniform_mix_ce':ce(uniform)}))

```

The same epsilon names different targets under these conventions. Match the implementation you use rather than relying on the label alone.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
