# Chapter 46: Segmentation Architectures

I would use this chapter to build a segmentation baseline and avoid attention shape errors. The two sc SE placements teach a distinction I can apply immediately. The loss code is less transparent: flattening erases the image boundary even though a competition may average scores by image. The activity makes that hidden choice tangible with six pixels per image. Threshold two small probability maps against two explicit binary masks: one large foreground and one tiny foreground. Compute Dice separately for each image, their arithmetic mean, and a pooled Dice from summed intersections and mask sizes. Show all three side by side with foreground counts. Pooled Dice weights images through their mask sizes, whereas mean per-image Dice weights each image equally. A threshold may change the two scores differently. Choose the exact competition reduction before using this activity to reason about a loss. Compare the threshold at which the tiny object disappears with the one that improves the larger mask. Decide which score would select your threshold if your host weights images equally.

## Worked example

When do pooled Dice and mean per-image Dice disagree after probability-map thresholding?

Binary masks use threshold >= parameter. Empty-empty masks have Dice 1 by declared convention. No neural network or nnU-Net is trained.

```python
parameter = 0.5
import json
assert 0.1<=parameter<=0.9
truth=[[1,1,1,1,0,0],[1,0,0,0,0,0]]
prob=[[.9,.8,.7,.4,.3,.2],[.45,.4,.35,.2,.1,.05]]
rows=[]; ti=tp=tt=0
for i,(y,p) in enumerate(zip(truth,prob)):
 b=[int(v>=parameter) for v in p]; inter=sum(a*c for a,c in zip(y,b)); denom=sum(y)+sum(b)
 d=2*inter/denom if denom else 1.0
 assert 0<=d<=1
 rows.append({'image':i,'truth_pixels':sum(y),'predicted_pixels':sum(b),'dice':d})
 ti+=inter; tp+=sum(b); tt+=sum(y)
print(json.dumps({'threshold':parameter,'mean_image_dice':sum(r['dice'] for r in rows)/2,'pooled_dice':2*ti/(tp+tt),'table':rows}))

```

Pooled Dice weights images through their mask sizes, whereas mean per-image Dice weights each image equally. A threshold may change the two scores differently. Choose the exact competition reduction before using this activity to reason about a loss.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
