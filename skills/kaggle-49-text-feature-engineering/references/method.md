# Chapter 49: Text Feature Engineering

The chapter tells me to inspect text coverage before paying for a transformer, then combine several cheap and learned representations. I can use the word-overlap diagnostic immediately, but its single cutoff seems more certain than the evidence provided. Character n-grams offer a useful counterexample: misspelled words may still share subword structure. The activity tests that distinction. Build character n-gram vocabularies from three training product phrases. For three test phrases containing spelling variants, compute both unique-type coverage and occurrence-weighted character coverage. Show ordinary word coverage for the same texts and vary character n-gram width. Type coverage gives rare and common n-grams equal weight; occurrence coverage counts every occurrence. Different widths expose different kinds of spelling overlap. Neither statistic identifies a universal threshold at which embeddings are required. Compare width one with width five. Explain whether the change comes from common letters, longer shared substrings, or exact word matches, and identify which coverage statistic hides repetition.

## Worked example

How can character n-grams preserve coverage when word vocabulary misses a typo?

Lowercase letters and spaces are retained for the constructed character representation. Coverage measures available vocabulary, not predictive quality.

```python
parameter = 3
import json
assert int(parameter)==parameter and 1<=parameter<=5
n=int(parameter)
train=['red running shoe','blue walking shoes','black sports shoe']
test=['red runing shoe','blu walking shoe','black sport shoes']
def grams(s): return [s[i:i+n] for i in range(len(s)-n+1)]
vocab={g for s in train for g in grams(s)}
w={v for s in train for v in s.split()}
rows=[]
for s in test:
 g=grams(s); u=set(g); tokens=s.split()
 a=len(u&vocab)/max(1,len(u)); b=sum(x in vocab for x in g)/max(1,len(g)); c=sum(x in w for x in tokens)/len(tokens)
 assert all(0<=x<=1 for x in (a,b,c))
 rows.append({'type_coverage':a,'occurrence_coverage':b,'word_coverage':c,'grams':len(g)})
print(json.dumps({'width':n,'training_types':len(vocab),'table':rows}))

```

Type coverage gives rare and common n-grams equal weight; occurrence coverage counts every occurrence. Different widths expose different kinds of spelling overlap. Neither statistic identifies a universal threshold at which embeddings are required.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
