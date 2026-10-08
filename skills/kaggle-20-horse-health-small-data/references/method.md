# Chapter 20: Horse Health: Small Data, Big Discipline

For exhaustive single-label multiclass prediction, micro-F1 equals accuracy. Each correct row contributes one true positive. Each wrong row contributes one false positive and one false negative across the classes. Substituting those counts into micro-F1 gives the fraction of correct rows.

Change the multiplier for class two. The activity adjusts class scores, takes one argmax label per row and computes both metrics. Their values move together and remain equal. The multiplier can change predictions because the probabilities may be imperfect; it does not reveal a difference between accuracy and micro-F1 under these assumptions.

In a small-data experiment, fit class adjustments on one set of predictions and assess them on another. An improvement reported on the same OOF labels used to tune the adjustments is an in-sample postprocessing result. Preserve a separate local holdout or wrap the adjustment fit inside outer evaluation. Inspect minority-class recall separately when that outcome matters, while keeping the official metric clear. Distinguish the local 988/247 split from the official test set. Small samples demand careful evidence, not a new formula for micro-F1.

## Worked example

Check multiclass micro-F1 and accuracy together.

Exhaustive single-label multiclass; all classes included.

```python
parameter = 1
import json
assert 0.5 <= parameter <= 2
y=[0,1,2,0,2];probs=[[.6,.3,.1],[.2,.5,.3],[.2,.35,.45],[.4,.2,.4],[.3,.4,.3]];pred=[max(range(3),key=lambda c:row[c]*(parameter if c==2 else 1)) for row in probs];tp=sum(a==b for a,b in zip(y,pred));fp=fn=len(y)-tp;micro=2*tp/(2*tp+fp+fn);accuracy=tp/len(y)
assert abs(micro-accuracy)<1e-12
result={'predictions':pred,'accuracy':accuracy,'micro_f1':micro,'correct_count':tp,'total_count':len(y)}
print(json.dumps(result))

```

Micro-F1 equals accuracy under stated assumptions at every scale.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
