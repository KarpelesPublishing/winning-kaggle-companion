# Chapter 31: Deep Learning for Tabular Data

Model choice and ensemble value are different questions. A neural model can have a worse individual score while correcting rows where the tree model struggles. This illustration gives two fixed sets of binary probabilities and four labels. Change the neural weight, inspect the resulting predictions, and compare mean squared error with both components. Describe which row benefits and which row deteriorates as the weight rises. The calculation uses a lower-is-better metric, so any search must minimize error directly. Before spending GPU time in your own competition, specify the training budget, valid split, preprocessing boundary, and evidence that the new component could contribute something beyond your current predictions. Change the neural-component weight and compare blend MSE with both component MSE values. The minimum is specific to these rows and this lower-is-better metric. Use fresh evaluation after choosing a weight. A small useful blend can still be too expensive for your final inference limit. Record prediction time as well as score, and choose the component weight only after confirming matching row and class-column order.

## Worked example

Can a weaker individual model improve a blend through complementary errors?

Two fixed predictions and true values illustrate squared error. This is an arithmetic blend, not neural training.

```python
parameter = 0.5
import json, math
assert 0<=parameter<=1
y=[0,1,0,1]; gbm=[0.1,0.6,0.1,0.9]; neural=[0.3,0.9,0.3,0.9]
mse=lambda p:sum((a-b)**2 for a,b in zip(y,p))/len(y)
blend=[(1-parameter)*g+parameter*d for g,d in zip(gbm,neural)]
assert all(0<=p<=1 for p in blend)
print(json.dumps({'weight':parameter,'gbm_mse':mse(gbm),'neural_mse':mse(neural),'blend_mse':mse(blend),'predictions':blend}))

```

The minimum is specific to these rows and this lower-is-better metric. Use fresh evaluation after choosing a weight.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
