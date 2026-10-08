# Chapter 56: Reinforcement Learning for Game Competitions

The chapter’s practical advice is to build a competitive rule baseline and add learning where replay evidence shows uncertainty. I can use that timeline. Reward shaping is the point where I would slow down: useful intermediate progress is not necessarily a faithful objective. The activity makes the warning concrete by letting a losing policy collect more of the rewarded resource. Compare two three-step constructed policies: one collects resource rewards but loses, and one collects less and wins. Multiply resource gains by the chosen shaping weight and add terminal reward plus/minus 5. Return both shaped returns, true win outcomes, and the policy selected by shaped reward. At some shaping weights the losing resource collector is preferred. This demonstrates objective mismatch, not the inadequacy of all shaping. To preserve optimal policies, consider a properly specified potential-based shaping rule or validate against real win rate. Find where the preferred policy changes as you increase the resource weight. Explain how the reward could become easier to earn than a win, and name a replay symptom you would inspect.

## Worked example

When can an intermediate reward make a losing policy look better than a winning one?

Illustrative returns, not a trained RL agent. No discounting; resource accumulation is not assumed to cause winning. Terminal outcome remains the actual objective.

```python
parameter = 0.5
import json
assert 0<=parameter<=2
policies=[{'id':0,'resources':[8,8,8],'win':0},{'id':1,'resources':[1,1,1],'win':1}]
rows=[]
for p in policies:
 terminal=5 if p['win'] else -5
 shaped=parameter*sum(p['resources'])+terminal
 rows.append({'policy':p['id'],'resources':sum(p['resources']),'terminal':terminal,'win':p['win'],'shaped_return':shaped})
chosen=max(rows,key=lambda r:(r['shaped_return'],-r['policy']))
assert rows[1]['terminal']>rows[0]['terminal']
print(json.dumps({'shaping_weight':parameter,'chosen_policy':chosen['policy'],'chosen_wins':chosen['win'],'table':rows}))

```

At some shaping weights the losing resource collector is preferred. This demonstrates objective mismatch, not the inadequacy of all shaping. To preserve optimal policies, consider a properly specified potential-based shaping rule or validate against real win rate.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
