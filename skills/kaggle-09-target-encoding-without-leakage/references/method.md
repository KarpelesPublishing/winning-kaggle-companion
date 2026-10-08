# Chapter 9: Target Encoding Without Leakage

Excluding a row's own target is necessary, but not the complete validation boundary. During an outer fold, every feature used to train the model must exclude that fold's validation targets. A globally precomputed OOF column can satisfy self-exclusion while violating this stronger requirement.

A and B are training rows; C and D are validation. Toggle C's target and inspect training row A's unsafe encoding. It changes because its sources include validation rows. The nested encoding uses only B and remains fixed. No fitted model is needed to expose this dependency.

Freeze the outer split first. Within its training slice, create inner cross-fitted training encodings. Fit a lookup on the complete outer training slice and use it to transform validation. Preserve group or time restrictions in both levels. After evaluation choices are fixed, produce final-fit training encodings and a full-training test lookup. Check that changing held-out labels cannot alter training features. Smoothing can reduce the size of contamination while leaving the dependency intact; therefore smoothing is not a substitute for containment.

This constructed activity illustrates the chapter topic. The revised chapter develops the full fitting and assessment boundaries.

## Worked example

Trace outer-validation label contamination.

A/B train, C/D validation; same category.

```python
parameter = 1
import json
assert 0 <= parameter <= 1
assert int(parameter) == parameter
y=[0,1,int(parameter),0];unsafe_sources=[2,3];safe_sources=[1]
unsafe=sum(y[i] for i in unsafe_sources)/2;safe=sum(y[i] for i in safe_sources)
assert safe==1 and not set(safe_sources)&{2,3}
result={'validation_target':parameter,'unsafe_training_encoding':unsafe,'nested_training_encoding':safe,'unsafe_validation_dependencies':2,'safe_validation_dependencies':0}
print(json.dumps(result))

```

Outer training must exclude all outer-validation targets.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
