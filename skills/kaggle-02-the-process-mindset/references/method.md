# Chapter 2: The Process Mindset

Build a log around the comparison you will need tomorrow. Before training, state one hypothesis and identify the baseline it challenges. Afterward, record metric direction, validation split and the artifact that produced the score. A version name alone will not recover changed data or fold assignments.

These four entries use an illustrative loss, so smaller is better. Change the gap tolerance and inspect marked rows. Every gap is leaderboard score minus CV score. Recalculate one by hand before interpreting the pattern. A positive gap means the public score is worse on this loss; its practical meaning reverses for a metric rewarding larger values.

A flag does not prove a feature is bad. Public and local samples differ, and scores can move by chance. Inspect paired fold changes and several comparable runs. Finish by writing a next action for one marked row: retain the champion, repeat a paired comparison, or audit a suspected leak. The useful output is a decision tied to a recoverable run, not just a longer spreadsheet or another leaderboard submission.

## Worked example

Recompute the logged CV-LB gap.

Lower-is-better illustrative losses.

```python
parameter = 0.0003
import json
assert 0 <= parameter <= 0.001
runs=[('baseline',.0245,.0248),('blend',.0223,.0221),('feature',.02215,.0224),('recency',.0221,.0225)]
rows=[{'version':v,'cv':cv,'lb':lb,'gap':round(lb-cv,8),'flag':int(abs(lb-cv)>parameter)} for v,cv,lb in runs]
assert rows[-1]['gap']==.0004
result={'rows':rows,'flagged_count':sum(r['flag'] for r in rows)}
print(json.dumps(result))

```

Flag means investigate, not proven overfit.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
