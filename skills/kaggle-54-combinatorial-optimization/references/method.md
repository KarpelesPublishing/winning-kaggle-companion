# Chapter 54: Combinatorial Optimization

The chapter changes my modeling frame: optimize a legal solution directly rather than fit a predictor. The simulated-annealing code is useful because the acceptance rule is visible. I would need more detail about feasibility before copying the scheduling example. The companion keeps every tour legal, so I can study acceptance and best-so-far tracking independently of constraint handling. Run a seeded forty-step search on a six-city tour with segment-reversal neighbors. Keep the temperature fixed at the selected value to isolate acceptance behavior. Report starting, final-current, and best tour cost, plus accepted worsening moves and the retained best permutation. Higher temperature can accept more worsening moves, but one finite seeded run need not improve monotonically with temperature. Best-so-far retention prevents returning a worse answer than the initial tour even when the current tour deteriorates. Compare the current tour and the retained best tour at two temperatures. Identify whether accepted worsening moves changed exploration, and state what repeated seeds would add to this single path.

## Worked example

How does annealing temperature affect exploration while preserving the best feasible tour?

Constructed Euclidean closed tour; every reversal remains feasible. Fixed temperature is a mechanism experiment, not a full cooling schedule. Temperature has the same units as tour-cost changes.

```python
parameter = 1
import math, random, json
assert .01<=parameter<=5
rng=random.Random(42); pts=[(0,0),(3,0),(3,3),(0,3),(1,1),(2,2)]
def cost(t): return sum(math.dist(pts[t[i]],pts[t[(i+1)%len(t)]]) for i in range(len(t)))
current=[0,2,1,3,5,4]; initial=cost(current); best=current[:]; bestcost=initial; bad=accepted=0
for _ in range(40):
 i,j=sorted(rng.sample(range(6),2)); cand=current[:i]+current[i:j+1][::-1]+current[j+1:]
 delta=cost(cand)-cost(current)
 if delta<=0 or rng.random()<math.exp(-delta/parameter):
  bad+=int(delta>1e-12); accepted+=1; current=cand
  if cost(current)<bestcost: best=current[:]; bestcost=cost(best)
assert sorted(best)==list(range(6)) and bestcost<=initial+1e-12
print(json.dumps({'temperature':parameter,'initial_cost':initial,'current_cost':cost(current),'best_cost':bestcost,'accepted_moves':accepted,'accepted_worse':bad,'best_tour':best}))

```

Higher temperature can accept more worsening moves, but one finite seeded run need not improve monotonically with temperature. Best-so-far retention prevents returning a worse answer than the initial tour even when the current tour deteriorates.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
