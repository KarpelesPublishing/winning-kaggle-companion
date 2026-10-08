# Chapter 58: Sensor Embeddings and Augmentation

I understand the chapter as making anatomical sensor identity part of the model and applying symmetry consistently at training and inference. That is a meaningful addition to generic time-series modeling. The complete pipeline is where I cannot proceed: four location IDs and twelve axis channels do not match the encoder’s contract. The activity makes sensor triples and mirrored IDs inspectable. Represent four sensors as named x, y, z triples. Reflect the first selected number of sensors by negating lateral x and swapping their left/right IDs. Apply the same transform twice and return original, once-reflected, and recovered triples and ID values. Flag duplicate resulting IDs when only one side is reflected. A double reflection should recover each selected triple and ID, but partial reflection can create duplicate anatomical assignments. A mechanically reversible transform does not prove a physiologically valid sample; pair completeness, axis alignment, and label invariance still need verification. Reflect one sensor and then all four. Explain why exact recovery after two reflections is compatible with an invalid intermediate anatomical configuration, and state what coordinate evidence you still need.

This constructed activity illustrates the chapter topic. The revised chapter develops the full fitting and assessment boundaries.

## Worked example

Does a sensor reflection preserve a coherent channel/location contract and undo itself when applied twice?

Coordinates are an explicitly assumed anatomical frame. Partial reflection is a diagnostic, not a recommended augmentation. Activity-label invariance is not demonstrated by an involution alone.

```python
parameter = 4
import json
assert int(parameter)==parameter and 0<=parameter<=4
n=int(parameter); data=[(1,[1,2,3]),(2,[4,5,6]),(3,[7,8,9]),(4,[10,11,12])]; swap={1:2,2:1,3:4,4:3}
def mirror(rows): return [(swap[k],[-v[0],v[1],v[2]]) if i<n else (k,v[:]) for i,(k,v) in enumerate(rows)]
once=mirror(data); twice=mirror(once)
assert twice==data
rows=[{'sensor_index':i,'original_id':a[0],'mirrored_id':b[0],'original_xyz':a[1],'mirrored_xyz':b[1],'recovered_xyz':c[1]} for i,(a,b,c) in enumerate(zip(data,once,twice))]
print(json.dumps({'reflected_sensors':n,'recovered_exactly':int(twice==data),'duplicate_ids':4-len(set(k for k,v in once)),'table':rows}))

```

A double reflection should recover each selected triple and ID, but partial reflection can create duplicate anatomical assignments. A mechanically reversible transform does not prove a physiologically valid sample; pair completeness, axis alignment, and label invariance still need verification.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
