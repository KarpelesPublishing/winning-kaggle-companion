# Chapter 40: Knowledge Distillation

Distillation targets describe what the teacher predicts, including its uncertainty and its mistakes. This activity softens probabilities in logit space rather than interpolating them directly. The fixed teacher outputs are 0.1, 0.5 and 0.9. Increase temperature from one to four and inspect the softened targets and their average binary entropy. Ordering survives, but confidence decreases. Explain why a student trained only to match these targets cannot receive evidence absent from the teacher and why hard labels may still deserve weight. Then trace who trained the teacher that supplied each student-training target. A teacher prediction being OOF for its own row does not automatically exclude every label in the student’s evaluation fold. Increase temperature and inspect softened probabilities and binary entropy. The targets become less extreme while retaining order. Greater entropy does not establish better calibration or a more accurate student. The soft target at temperature four is a calculable probability, not a confidence guarantee. Ask whether that target retains useful teacher distinctions and whether your student evaluation excludes all labels that shaped it.

## Worked example

What information remains when a teacher probability is softened?

Binary teacher probabilities are fixed; temperature operates in logit space. This does not fit a student.

```python
parameter = 2
import json, math
assert 1<=parameter<=4
raw=[0.1,0.5,0.9]
soft=[1/(1+math.exp(-math.log(p/(1-p))/parameter)) for p in raw]
entropy=lambda p:-(p*math.log(p)+(1-p)*math.log1p(-p))
assert soft==sorted(soft)
print(json.dumps({'temperature':parameter,'teacher_probabilities':raw,'soft_targets':soft,'mean_soft_target_entropy':sum(map(entropy,soft))/3,'p09_softened':soft[2]}))

```

The targets become less extreme while retaining order. Greater entropy does not establish better calibration or a more accurate student.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
