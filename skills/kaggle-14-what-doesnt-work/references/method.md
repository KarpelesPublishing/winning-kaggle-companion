# Chapter 14: What Doesn't Work

Pseudo-labels are predictions used as temporary training labels. Their safety in a validation experiment depends on how the teacher was trained. Sharing a pseudo-label set is not automatically contamination; sharing a teacher that consumed the held-out labels can be.

Toggle whether the teacher includes validation labels. The activity tracks its training dependencies for two illustrative pseudo-rows. Inspect the count of held-out label dependencies and the safe-for-this-fold flag. A zero count is necessary for this boundary, although it does not establish pseudo-label accuracy or usefulness.

For a real outer fold, fit the teacher from its training slice, generate pseudo-labels without validation targets, and augment only that slice. Keep confidence filtering and pseudo-row limits within the same selection boundary. Evaluate the student on genuine held-out labels and compare with a no-pseudo baseline. Do not include pseudo-label agreement as if it were another ground-truth validation score. If a globally trained teacher has already consumed held-out labels, rebuild the experiment rather than trying to cure the dependency with a tighter confidence threshold. Document both information availability and host permission for using unlabeled test inputs.

## Worked example

Check pseudo-label teacher dependencies.

Teacher label-dependency audit; pseudo examples are illustrative.

```python
parameter = 0
import json
assert 0 <= parameter <= 1
assert int(parameter) == parameter
validation={4,5};train={0,1,2,3};teacher=train|validation if parameter else train
pseudo=[{'row':6,'label':1,'teacher_label_ids':sorted(teacher)},{'row':7,'label':0,'teacher_label_ids':sorted(teacher)}]
count=len(teacher&validation)
if not parameter: assert count==0
result={'teacher_validation_dependencies':count,'safe_for_this_outer_fold':int(count==0),'pseudo_rows':pseudo}
print(json.dumps(result))

```

Shared teacher is unsafe if it used this validation slice; correlation alone is not the defect.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
