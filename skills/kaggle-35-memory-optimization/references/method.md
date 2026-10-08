# Chapter 35: Memory Optimization for Large Datasets

Saving memory can change what your model is able to distinguish. The three example values are consecutive large integers stored as floats. Select 64-bit storage and then 32-bit storage. Compare the byte estimate, maximum conversion error, and number of unique values. The smaller representation has plenty of range for these numbers, yet it cannot preserve all of them distinctly. That matters for identifiers, narrow thresholds, and features where small differences carry signal. Before downcasting a real column, name the tolerated error and inspect collisions as well as the minimum and maximum. Then profile peak memory through the entire load, transformation, conversion and model-training sequence. Change the numeric storage width and inspect estimated bytes and unique-value collisions. Range compatibility is not precision compatibility. The byte estimate excludes dataframe objects, model buffers and temporary copies. Inspect identifiers separately from measurements. An identifier collision can merge two entities, while a small measurement error may be harmless. The tolerable representation therefore depends on how the column enters later operations.

## Worked example

How much memory do you save, and which values stop being distinguishable?

IEEE binary32 conversion is demonstrated using struct. Fixed values near 2^24 reveal precision loss.

```python
parameter = 32
import json, math
import struct
assert parameter in (32,64)
values=[16777216.,16777217.,16777218.]
converted=[struct.unpack('f',struct.pack('f',x))[0] for x in values] if parameter==32 else values[:]
errors=[abs(a-b) for a,b in zip(values,converted)]
print(json.dumps({'storage_bits':int(parameter),'estimated_array_bytes':len(values)*int(parameter)//8,'original_unique':len(set(values)),'converted_unique':len(set(converted)),'max_absolute_error':max(errors),'converted':converted}))

```

Range compatibility is not precision compatibility. The byte estimate excludes dataframe objects, model buffers and temporary copies.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
