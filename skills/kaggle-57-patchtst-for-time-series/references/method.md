# Chapter 57: PatchTST for Time Series and HAR

The chapter teaches me to turn local signal windows into manageable attention tokens. The unfolding code is concrete enough to follow with Python-array reasoning. I would reread the token reduction claim because overlapping stride changes the count. I also need to know whether the forecasting head emits one target or every channel. The companion isolates patch construction and coverage. Patch a 48-point constructed signal with stride half the selected even patch length. Compute patch starts and means, count how often each time point is covered, and compare squared token counts against 48 single-step tokens. Report uncovered tail points and overlap multiplicity. Overlapping patches generally produce more than L/P tokens. Coverage can leave a tail when division is uneven. Token-pair reduction describes one attention term; the projection and prediction-head costs remain separate. Compare patch lengths four and sixteen. Calculate tokens using the displayed stride rather than length divided by patch size, and explain how uncovered tail samples would affect real signal preparation.

## Worked example

How do patch length and overlapping stride change tokens, coverage, and attention-pair count?

Even patch lengths 4 to 16 and stride=P/2. Patch means stand in for inspection, not learned token projections. Attention-pair count is a per-channel illustration, not whole-model runtime.

```python
parameter = 8
import math, json
assert int(parameter)==parameter and 4<=parameter<=16 and int(parameter)%2==0
p=int(parameter); stride=p//2; signal=[math.sin(2*math.pi*t/12)+.03*t for t in range(48)]
starts=list(range(0,49-p,stride)); cover=[0]*48; rows=[]
for start in starts:
 for t in range(start,start+p): cover[t]+=1
 rows.append({'start':start,'end_exclusive':start+p,'mean':sum(signal[start:start+p])/p})
assert len(starts)==(48-p)//stride+1
print(json.dumps({'patch_length':p,'stride':stride,'tokens':len(starts),'single_step_pairs':48**2,'patch_pairs':len(starts)**2,'uncovered_points':sum(c==0 for c in cover),'maximum_coverage':max(cover),'table':rows}))

```

Overlapping patches generally produce more than L/P tokens. Coverage can leave a tail when division is uneven. Token-pair reduction describes one attention term; the projection and prediction-head costs remain separate.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
