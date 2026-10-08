# Chapter 18: Final Submission Selection

Final selection begins with the competition's actual rules and a record of validated candidates. A backup can hedge a different failure mode, but the lowest correlation alone does not maximize the chance of a good private result.

Change the CV tolerance around the primary score. Only candidates close enough in this illustrative registry are eligible. Among those, the activity picks a lower-correlation backup for further inspection. At zero tolerance there may be no eligible backup. Read that outcome as missing evidence, not an instruction to invent a second entry.

Now inspect the proposed backup's error slices and assumptions. Are its differences concentrated in a few groups, due to calibration, or caused by a pipeline bug? Confirm comparable validation conditions before interpreting diversity. A high-correlation improved challenger may deserve promotion when evidence supports it, but it does not become a diversity hedge merely because it is new. Freeze selected artifacts and check row IDs, columns, ranges and provenance before the deadline. Treat the tolerance and diversity rule as a transparent heuristic, not a probability model of private leaderboard survival.

## Worked example

Choose a backup within an explicit score tolerance.

Higher-is-better scores; tolerance-qualified backups; correlation only heuristic.

```python
parameter = 0.005
import json
assert 0 <= parameter <= 0.02
primary={'id':1,'cv':.885,'correlation':1.};candidates=[{'id':2,'cv':.884,'correlation':.99},{'id':3,'cv':.881,'correlation':.90},{'id':4,'cv':.870,'correlation':.60}]
eligible=[r for r in candidates if primary['cv']-r['cv']<=parameter+1e-12];backup=min(eligible,key=lambda r:r['correlation']) if eligible else None
if backup: assert primary['cv']-backup['cv']<=parameter+1e-12
result={'primary_id':1,'eligible_count':len(eligible),'backup_id':backup['id'] if backup else -1,'backup_cv':backup['cv'] if backup else 0,'backup_correlation':backup['correlation'] if backup else 0,'decision':'inspect distinct error slices' if backup else 'no evidence-qualified backup'}
print(json.dumps(result))

```

Heuristic hedge, not optimized private-survival probability.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
