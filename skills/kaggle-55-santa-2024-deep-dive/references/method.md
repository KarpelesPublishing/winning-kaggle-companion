# Chapter 55: Pancake Sorting: A Constructed Search Example

I read this as a lesson in checking algorithm suitability before committing a compute budget. The failure story and branching calculations sustain attention. I would reread the heuristic claim because misplaced items can all be fixed by a single reversal. The companion separates an illustrative legal search from the unverified competition history and makes completion an explicit output. Search a five-pancake permutation with legal prefix reversals. Keep only the selected number of states per depth, deduplicate states, and use deterministic tie-breaking with a zero heuristic. Return solved status, flip sequence, expanded-state count, and replayed final state; demonstrate the four-mismatch one-flip counterexample separately. A returned path is usable only when replay reaches the declared goal. Beam width changes retained alternatives and expansion work. The mismatch counterexample explains why a plausible score is not automatically an admissible lower bound. Try a narrow beam and a wide beam. Read the solved flag before judging cost, replay the moves yourself, and explain why a lower heuristic value cannot certify a completed solution.

This constructed activity illustrates the chapter topic. The revised chapter develops the full fitting and assessment boundaries.

## Worked example

How does beam width affect whether a legal pancake-flip path reaches the goal within a fixed budget?

This is classic pancake sorting, explicitly not verified Santa 2024 rules. Five depth expansions; narrow beams may fail. No optimality is claimed for a pruned beam.

```python
parameter = 5
import json
assert int(parameter)==parameter and 1<=parameter<=30
width=int(parameter); start=(3,1,5,2,4); goal=tuple(range(1,6)); beam=[(start,[])]; seen={start}; expanded=0; solved=None
for depth in range(6):
 for state,path in beam:
  if state==goal: solved=(state,path); break
 if solved or depth==5: break
 candidates=[]
 for state,path in beam:
  for k in range(2,6):
   nxt=state[:k][::-1]+state[k:]; expanded+=1
   if nxt not in seen: seen.add(nxt); candidates.append((nxt,path+[k]))
 beam=sorted(candidates,key=lambda a:(a[0],a[1]))[:width]
 if not beam: break
state,path=solved if solved else (beam[0] if beam else (start,[]))
replay=start
for k in path: replay=replay[:k][::-1]+replay[k:]
assert replay==state
counter=(4,3,2,1); assert counter[::-1]==(1,2,3,4)
print(json.dumps({'beam_width':width,'solved':int(state==goal),'expanded':expanded,'flips':path,'final_state':list(state),'counterexample_mismatches':4,'counterexample_flips':1}))

```

A returned path is usable only when replay reaches the declared goal. Beam width changes retained alternatives and expansion work. The mismatch counterexample explains why a plausible score is not automatically an admissible lower bound.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
