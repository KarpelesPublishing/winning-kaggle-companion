# Chapter 1: What Actually Wins on Kaggle

Start with a competition you understand well enough to name the target, the row, and the object or person behind that row. Write those three things separately. In the transaction setting, the entity is a customer and the outcome occurs later. That distinction changes both possible features and observations you should keep apart during validation.

Select a next action and read its evidence requirement. Auditing entities needs identifiers and timestamps. An amount-deviation feature needs history available before prediction. Tuning needs an evaluation scheme you trust. A modality-specific model needs evidence its representation adds information your baseline misses.

Record one experiment for your competition: what information it adds, how you will test it, and what observation would make you abandon it. The worksheet predicts no rank or score gain. It makes your first decision explicit and exposes missing evidence before a long run. Keep that decision beside the baseline so you can later distinguish a model-choice problem from an evaluation-assumption problem. If you cannot identify the prediction boundary, resolve that uncertainty before choosing more features.

## Worked example

Choose a justified first action.

Decision IDs index actions, not expected performance.

```python
parameter = 0
import json
assert 0 <= parameter <= 3
assert int(parameter) == parameter
choices=[{'action':'audit entities and future holdout','needed':'IDs, timestamps, label availability','next':'build isolated chronological split'},{'action':'derive historical amount deviation','needed':'past-only histories','next':'test one feature on frozen folds'},{'action':'tune baseline','needed':'stable folds and error analysis','next':'bound a small search'},{'action':'try modality-specific representation','needed':'raw text/images inform target','next':'compare a pretrained baseline'}]
result={'selected_id':int(parameter),'decision':choices[int(parameter)],'choice_count':4}
print(json.dumps(result))

```

Choice creates a testable plan; no invented gain.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
