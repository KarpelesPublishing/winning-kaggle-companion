# Chapter 15: When the Problem Isn't Tabular

A search algorithm must return a legal completed solution, not only a promising estimate. In prefix-flip sorting, the state is the pancake order and an action reverses the first k items. The reported cost should count actions in a path that actually reaches sorted order.

Change the depth limit for the three-item instance. The bounded breadth-first search reports whether it found the goal, its legal flips and visited nodes. Replay the flips mentally. At a small limit, failure means no goal was found within that budget; it does not mean sorting is impossible. When a path is found, the example replays it and asserts sorted order.

Before scaling a search, separate an achievable greedy completion cost from an admissible lower bound. A greedy procedure normally supplies an upper bound because it constructs a feasible solution. A beam-search ranking heuristic may be useful without being admissible, but that limits what you can claim about optimality. Profile actual expansions on small instances and increase budgets gradually. The activity demonstrates goal checking and legal-path accounting, not a benchmark for beam search, NMCS or a competition-winning search strategy.

## Worked example

Return a legal goal-reaching prefix-flip path.

Exact bounded BFS on three distinct pancake sizes; not NMCS benchmark.

```python
parameter = 3
import json
assert 0 <= parameter <= 5
assert int(parameter) == parameter
from collections import deque
start=(3,1,2);goal=(1,2,3);queue=deque([(start,[])]);seen={start};found=None;visited=0
while queue:
 state,path=queue.popleft();visited+=1
 if state==goal: found=path;break
 if len(path)>=int(parameter): continue
 for k in range(2,len(state)+1):
  nxt=state[:k][::-1]+state[k:]
  if nxt not in seen: seen.add(nxt);queue.append((nxt,path+[k]))
end=start
if found is not None:
 for k in found:end=end[:k][::-1]+end[k:]
 assert end==goal
result={'goal_found':int(found is not None),'flip_count':len(found) if found is not None else -1,'legal_path':found or [],'nodes_visited':visited,'depth_limit':int(parameter)}
print(json.dumps(result))

```

Goal/path verified; failure means not found within depth, not impossible.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
