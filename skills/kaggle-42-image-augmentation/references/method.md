# Chapter 42: Image Augmentation

A classification prediction has no pixel coordinates, but a segmentation mask does. The constructed mask has two rows and three columns, with foreground concentrated toward the left. A horizontal flip moves that foreground to the right. Select two-view averaging and inspect what happens if you average masks before returning the flipped prediction to the original frame. Then compare the correctly aligned result. This activity assumes each view is predicted perfectly, so any error comes entirely from coordinate handling. Use the same test for rotations and resized outputs. Separately decide whether the transform preserves the task’s label: correct inversion cannot make an invalid augmentation appropriate. Choose identity-only or two-view averaging and compare averaging before and after reversing the flipped mask. Correct inverse mapping preserves the original mask; unaligned averaging mixes different pixel locations. For a rotation followed by a flip, undo the flip before reversing the rotation. Write the operations in order, then test an asymmetric mask so an accidental alignment shortcut cannot pass unnoticed.

## Worked example

Why must segmentation predictions be returned to the original coordinate frame?

A fixed binary mask is predicted perfectly under identity and horizontal flip. It illustrates alignment only.

```python
parameter = 2
import json, math
assert parameter in (1,2)
mask=[[1.,0.,0.],[1.,1.,0.]]
flipped=[r[::-1] for r in mask]
views=[mask] if parameter==1 else [mask,flipped]
raw=[[sum(v[i][j] for v in views)/parameter for j in range(3)] for i in range(2)]
aligned_views=[mask] if parameter==1 else [mask,[r[::-1] for r in flipped]]
aligned=[[sum(v[i][j] for v in aligned_views)/parameter for j in range(3)] for i in range(2)]
mse=lambda a:sum((a[i][j]-mask[i][j])**2 for i in range(2) for j in range(3))/6
assert mse(aligned)==0
print(json.dumps({'view_count':int(parameter),'unaligned_mse':mse(raw),'aligned_mse':mse(aligned),'unaligned_average':raw,'aligned_average':aligned}))

```

Correct inverse mapping preserves the original mask; unaligned averaging mixes different pixel locations.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
