# Chapter 22: Vision and Multimodal: When You Need a VLM

A model's parameter count provides a lower bound on memory, not its complete inference footprint. Four-bit weights require half a byte per parameter before quantization metadata, activations, temporary buffers, vision inputs and generation caches. Multiple GPUs add placement constraints even when aggregate arithmetic appears to fit.

Change the parameter count. Inspect raw weight GiB, the explicit overhead assumption and the aggregate-budget result. The example also calculates serial runtime for five thousand images at an assumed rate. These are budget calculations, not measured throughput or proof a checkpoint can load on two devices.

For your VLM pipeline, profile a representative image and response workload on the exact hardware and software versions. Include loading, preprocessing, second opinions, postprocessing and safety margin. Constrain responses to the taxonomy, define how class confidence is obtained and validate it before using an uncertainty threshold. Measure rule overrides on untouched labels, including both corrections and introduced mistakes. A disagreement flag indicates different answers, not verified difficulty or correctness. Resolve which model wins conflicts before claiming a pipeline result, and preserve that policy with the submission artifact.

## Worked example

Budget VLM weights and runtime headroom.

4-bit raw weights; assumed6GiB overhead;30GiB aggregate budget. No guarantee of per-device placement, kernels or measured throughput.

```python
parameter = 32
import json
assert 7 <= parameter <= 72
assert int(parameter) == parameter
params=parameter*1e9;bits=4;raw_gib=params*bits/8/(1024**3);overhead_gib=6.;available_gib=30.;rate=.3;count=5000
result={'raw_weight_gib':raw_gib,'assumed_runtime_overhead_gib':overhead_gib,'estimated_total_gib':raw_gib+overhead_gib,'available_aggregate_gib':available_gib,'aggregate_budget_pass':int(raw_gib+overhead_gib<=available_gib),'base_inference_minutes':count*rate/60,'assumptions':'aggregate lower-bound worksheet; per-device fit requires profiling'}
print(json.dumps(result))

```

Aggregate arithmetic is necessary but insufficient for feasible sharding.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
