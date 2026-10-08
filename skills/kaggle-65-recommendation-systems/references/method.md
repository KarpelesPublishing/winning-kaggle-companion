# Chapter 65: Recommendation Systems

The chapter gives me a usable system shape: retrieve plausible items, then build a familiar ranker on realistic candidates. The cold-start routing discussion changes what I would implement. I would reread the recall-ceiling formula because its aggregation differs from MAP. The companion keeps retrieval fixed and derives a ranking bound under one explicit metric rather than transferring recall numbers blindly. For three constructed users, union ordered history, co-visitation, and popularity lists while deduplicating. Keep the selected candidate cap. Compute micro and macro recall, and the oracle AP@3 if every retrieved relevant item were ranked first. Return per-user truth counts, hits, recall, and oracle AP. Micro recall and macro AP need not share a numeric ceiling because their denominators differ. Candidate coverage constrains ranking, but the bound must respect the metric’s units. A larger union before truncation can still lose useful items if source priority pushes them outside the cap. Compare candidate caps one, four, and eight. Identify which source contributes each captured item, then explain why a correct item lost during truncation cannot be recovered by a reranker. Trace the denominators before using a candidate ceiling in a modeling decision.

## Worked example

How does a capped union of candidate sources change recall, and what is the correct metric-specific ranking ceiling?

Binaryrelevanceand AP@3 denominatormin(numberoftrueitems, 3). Oracle is an analytical upperboundfor this exact candidate set, not a trainedranker. Sourceorderdeterminestruncationpriority.

```python
parameter = 4
import json
assert int(parameter)==parameter and 1<=parameter<=8
cap=int(parameter); truth=[{2},{4,5,6,7},{8,9}]
sources=[[[1,2],[3,4],[5,6]],[[4,1],[5,6],[7,8]],[[1,8],[9,2],[3,4]]]
rows=[]; totalhits=tottruth=0
for user,(ys,lists) in enumerate(zip(truth,sources)):
 union=list(dict.fromkeys(item for items in lists for item in items)); chosen=union[:cap]; hits=len(set(chosen)&ys)
 oracle=min(hits,3)/min(len(ys),3)
 rows.append({'user':user,'truth_items':len(ys),'hits':hits,'candidates':chosen,'recall':hits/len(ys),'oracle_ap_at_3':oracle})
 totalhits+=hits; tottruth+=len(ys)
assert all(0<=r['oracle_ap_at_3']<=1 for r in rows)
print(json.dumps({'cap':cap,'micro_recall':totalhits/tottruth,'macro_recall':sum(r['recall'] for r in rows)/3,'oracle_map_at_3':sum(r['oracle_ap_at_3'] for r in rows)/3,'table':rows}))

```

Micro recall and macro APneednot share a numeric ceiling because their denominators differ. Candidate coverage constrains ranking, the bound must respect metric units. A larger union before truncation can still lose useful items if source priority pushes them outside the cap.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
