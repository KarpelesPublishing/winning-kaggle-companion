# Chapter 32: Denoising Autoencoders for Tabular Pretraining

**Does a lower reconstruction error mean the pretrained features help a small labeled model more than an untrained encoder would?**

Reconstruction loss is the number pretraining produces for free, so it is the number people optimize. It says how well the pretext task was fitted, not whether the target's information survived, and the untrained control is rarely run.

## The experiment

Fourteen constructed datasets: 20 features driven by four hidden factors plus noise, 3,000 unlabeled rows, 300 labeled rows and 3,000 new rows. The label depends on a product of two factors and one more factor. A 64-unit hidden-layer network (wider than the input) is trained on two swap-noise copies of every unlabeled row to rebuild the clean row; the control is the swap probability (0 gives a plain autoencoder). Its hidden activations are appended to the 20 features of a logistic regression fitted on the 300 labeled rows and scored by AUC on the new rows. Two controls: the raw features alone, and the hidden activations of an untrained network with the same shape.

Control: Swap probability per cell (0: plain autoencoder, 0.1: the chapter's usual rate, 0.3, 0.6; default 0.1).

## Measured results

| Measure | 0: plain autoencoder | 0.1: the chapter's usual rate | 0.3 | 0.6 |
|---|---|---|---|---|
| Reconstruction, trained | 0.003 | 0.037 | 0.083 | 0.184 |
| Reconstruction, untrained | 1.405 | 1.405 | 1.405 | 1.405 |
| AUC raw features | 0.676 | 0.676 | 0.676 | 0.676 |
| AUC + untrained encoder | 0.765 | 0.765 | 0.765 | 0.765 |
| AUC + trained encoder | 0.751 | 0.768 | 0.770 | 0.776 |

## What the result says (default, swap probability per cell = 0.1)

At swap probability 0.1 the trained autoencoder reconstructs new rows with error 0.037 against 1.405 for the untrained network. Its features give a downstream AUC of 0.768; the raw features alone give 0.676 and an untrained encoder of the same shape 0.765. So 0.768 - 0.765 = 0.003 is what pretraining adds beyond a random encoder (one standard deviation over datasets: 0.011), while 0.768 - 0.676 = 0.092 is the gain over raw features.

- Reconstruction: 0.037 trained against 1.405 untrained (mean squared error, standardized units).
- Downstream AUC: trained 0.768, untrained encoder 0.765, raw features 0.676.
- Pretraining beyond a random encoder: 0.768 - 0.765 = +0.003.
- Gain over raw features: 0.768 - 0.676 = +0.092.

## Apply it to a competition

- Always fit the same-shape untrained network beside the pretrained one and compare the downstream metric; a gain over raw features alone is not evidence for pretraining.
- Do not choose corruption strength by reconstruction loss; choose it by the downstream metric on matched labeled folds.
- Pretrain only on the permitted unlabeled population for the assessment you are running (outer-training rows for an inductive assessment).
- Record a failed transfer as a result, with the pretraining data identity and corruption settings.

## Assumptions and limits

Constructed data and a scikit-learn network with one hidden layer, trained for 30 epochs, standing in for the chapter's deeper PyTorch autoencoder; the downstream model is a logistic regression with a fixed regularization. A deeper autoencoder, a different head or a different target could favour pretraining; this setup measures that pretraining added little beyond an untrained encoder here, which is a result about this generator, not a verdict on DAEs.

Constructed data; pretraining adds little over an untrained encoder in this generator, which is a property of the setup, not a general claim about denoising autoencoders.

## Reproduce it

The chapter notebook `notebooks/32-denoising-autoencoders.ipynb` runs this code for every control value. It is importable as `kaggle_companion.activities.ch32` (`run`, `explain`, `draw`).

```python
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from kaggle_companion.activities._common import clean

SEED = 32
REPLICATES = 14         # independent constructed datasets; every estimate is their mean
FEATURES, FACTORS, HIDDEN = 20, 4, 64
N_UNLABELED, N_LABELED, N_NEW = 3000, 300, 3000


def generate(rng):
    """Twenty noisy features driven by four hidden factors. The label depends on a product of two factors, so a
    plain linear model on the features cannot read it. Unlabeled rows are plentiful; labeled rows are few."""
    loading = rng.normal(size=(FACTORS, FEATURES))

    def rows(n):
        f = rng.normal(size=(n, FACTORS))
        return f, f @ loading + 0.5 * rng.normal(size=(n, FEATURES))

    def label(f):
        logit = 1.6 * f[:, 0] * f[:, 1] + 1.0 * f[:, 2]
        return (rng.random(len(f)) < 1 / (1 + np.exp(-logit))).astype(int)

    _, X_unlabeled = rows(N_UNLABELED)
    f_lab, X_lab = rows(N_LABELED)
    f_new, X_new = rows(N_NEW)
    return X_unlabeled, (X_lab, label(f_lab)), (X_new, label(f_new))


def swap_noise(X, p_swap, rng):
    """Replace each cell with probability p_swap by the same column's value from a random donor row."""
    corrupted = X.copy()
    mask = rng.random(X.shape) < p_swap
    donors = rng.integers(0, len(X), X.shape)
    corrupted[mask] = X[donors[mask], np.nonzero(mask)[1]]
    return corrupted


def autoencoder(seed, trained, X_in=None, X_target=None):
    """One hidden layer of 64 ReLU units (wider than the 20 inputs), fitted to map corrupted rows to clean rows.
    trained=False takes one step at a negligible learning rate, which leaves the random initial weights in place."""
    net = MLPRegressor(hidden_layer_sizes=(HIDDEN,), batch_size=256, learning_rate_init=0.003 if trained else 1e-9,
                       max_iter=30 if trained else 1, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # the iteration cap is deliberate
        net.fit(X_in, X_target)
    return net


def encode(net, X):
    """The hidden-layer activations: the learned (or, untrained, random) representation of each row."""
    return np.maximum(0, X @ net.coefs_[0] + net.intercepts_[0])


def downstream_auc(train_features, y, new_features, y_new):
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(C=0.3, max_iter=500).fit(scaler.transform(train_features), y)
    return roc_auc_score(y_new, model.predict_proba(scaler.transform(new_features))[:, 1])


def one_replicate(p_swap, seed):
    rng = np.random.default_rng(seed)
    X_unl, (X_lab, y_lab), (X_new, y_new) = generate(rng)
    scaler = StandardScaler().fit(X_unl)
    X_unl, X_lab, X_new = scaler.transform(X_unl), scaler.transform(X_lab), scaler.transform(X_new)
    clean_rows = np.vstack([X_unl, X_unl])                                         # two corrupted copies of every unlabeled row
    corrupted = np.vstack([swap_noise(X_unl, p_swap, rng), swap_noise(X_unl, p_swap, rng)])
    dae = autoencoder(seed, True, corrupted, clean_rows)
    untrained = autoencoder(seed + 7, False, corrupted[:300], clean_rows[:300])

    def auc_with_encoder(net):                                                      # raw features plus the encoder's activations
        return downstream_auc(np.hstack([X_lab, encode(net, X_lab)]), y_lab, np.hstack([X_new, encode(net, X_new)]), y_new)

    return {
        "reconstruction_dae": float(np.mean((dae.predict(X_new) - X_new) ** 2)),       # clean new rows in, clean rows out
        "reconstruction_untrained": float(np.mean((untrained.predict(X_new) - X_new) ** 2)),
        "auc_raw": downstream_auc(X_lab, y_lab, X_new, y_new),
        "auc_untrained": auc_with_encoder(untrained),
        "auc_dae": auc_with_encoder(dae),
    }


def run(p_swap):
    reps = [one_replicate(p_swap, SEED * 100 + i) for i in range(REPLICATES)]
    mean = lambda key: float(np.mean([r[key] for r in reps]))
    keys = ("reconstruction_dae", "reconstruction_untrained", "auc_raw", "auc_untrained", "auc_dae")
    paired = [r["auc_dae"] - r["auc_untrained"] for r in reps]
    return clean({
        "p_swap": p_swap, **{k: mean(k) for k in keys},
        "dae_minus_untrained": float(np.mean(paired)), "dae_minus_untrained_sd": float(np.std(paired)),
        "per_replicate_auc": {k: [r["auc_" + k] for r in reps] for k in ("raw", "untrained", "dae")},
        "replicates": REPLICATES, "unlabeled_rows": N_UNLABELED, "labeled_rows": N_LABELED, "new_rows": N_NEW,
    })
```

Book location: Chapter 32, Setting Realistic Expectations. Constructed example: fourteen seeded synthetic datasets, a small scikit-learn autoencoder with swap noise and a logistic regression, measured by the chapter activity.
