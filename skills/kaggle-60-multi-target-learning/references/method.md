# Chapter 60: Multi-Target Learning

I take away that more labels help only when their shared learning signal supports the primary task. The shared-trunk model and primary-target checklist are useful. I would pause at the named loss-balancing algorithms because the code omits some of their defining machinery. The one-parameter companion makes alignment and cancellation visible without needing a training framework. Use one shared scalar initialized at zero. The primary quadratic loss prefers value 1; compare auxiliary losses preferring 1 and -1. Make one gradient step of size 0.1 with the selected auxiliary weight, then report gradients, updated parameter, and primary loss for both cases. An aligned auxiliary pushes the parameter toward the primary optimum; a conflicting auxiliary cancels or reverses that step as its weight grows. This exposes one local mechanism without claiming every OOF degradation has that cause. Find the weight where opposing gradients cancel. Explain why a nonzero auxiliary loss can produce no shared update and why one gradient measurement cannot establish long-run incompatibility.

## Worked example

How does auxiliary-loss weight change a shared parameter when tasks agree or conflict?

One-step analyticquadratic model, not a trained neural network. Loss scale and gradient direction are explicit. A primary performance drop does not identify gradient conflict by itself.

```python
parameter = 0.5
import json
assert 0<=parameter<=2
rows=[]
for aux_target in (1,-1):
 theta=0.0; primary_grad=theta-1; aux_grad=theta-aux_target
 total=primary_grad+parameter*aux_grad; updated=theta-.1*total
 rows.append({'aux_target':aux_target,'primary_gradient':primary_grad,'aux_gradient':aux_grad,'gradient_product':primary_grad*aux_grad,'combined_gradient':total,'updated_parameter':updated,'primary_loss_before':.5,'primary_loss_after':.5*(updated-1)**2})
assert rows[0]['gradient_product']>0 and rows[1]['gradient_product']<0
print(json.dumps({'auxiliary_weight':parameter,'table':rows}))

```

An aligned auxiliary pushes the parameter toward the primary optimum; a conflicting auxiliary cancels or reverses that step as its weight grows. This exposes one local mechanism without claiming every OOF degradation has that cause.

This is a constructed teaching example. Its results describe these supplied inputs, not a measured competition gain.
