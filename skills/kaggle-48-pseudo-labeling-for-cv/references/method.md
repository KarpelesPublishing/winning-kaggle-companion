# Chapter 48: Pseudo-Labeling for CV

I understand pseudo-labeling as amplifying a model’s existing beliefs rather than acquiring independent truth. The class-count printout is worth preserving because it turns a failure mode into an observable check. I would reread how pseudo-labels enter the student dataset: the two snippets use different identifier columns. The activity focuses on confidence’s limits before adding a training loop. Filter ten constructed predictions by maximum-class confidence. Hidden evaluation labels are supplied only for this diagnostic so you can count correctness, class balance, and accepted errors. Report acceptance fraction and error fraction, including a flag when no examples pass. Raising confidence trades coverage for a different selected population, not guaranteed correctness. A high-confidence systematic error can survive the strictest useful setting. Class counts and error counts explain why independent real-label validation remains necessary. Try a cutoff that leaves only the most confident rows. Identify the surviving mistake and explain what extra evidence would justify treating that pseudo-label as training truth.

## Worked example

Can stricter pseudo-label confidence reduce coverage without removing systematic errors?

Oracle labels are for demonstration auditing only, never used to select labels in production. Confidence is deliberately imperfect and includes a confidently wrong example.

```python
parameter = 0.85
import json
assert .5<=parameter<=1
items=[(.99,0,1),(.98,0,0),(.95,0,0),(.92,1,1),(.89,0,0),(.86,0,1),(.82,1,1),(.76,1,0),(.67,1,1),(.55,0,1)]
sel=[r for r in items if r[0]>=parameter]; errors=sum(p!=y for c,p,y in sel)
counts=[sum(p==k for c,p,y in sel) for k in (0,1)]
assert sum(counts)==len(sel) and errors<=len(sel)
print(json.dumps({'threshold':parameter,'accepted':len(sel),'acceptance_fraction':len(sel)/len(items),'errors':errors,'error_fraction':errors/max(1,len(sel)),'has_selected':int(bool(sel)),'table':[{'class':k,'count':v} for k,v in enumerate(counts)]}))

```

Raising confidence trades coverage for a different selected population, not guaranteed correctness. A high-confidence systematic error can survive the strictest useful setting. Class counts and error counts explain why independent real-label validation remains necessary.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
