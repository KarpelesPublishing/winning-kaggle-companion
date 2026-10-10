"""Chapter 32: Denoising Autoencoders for Tabular Pretraining. Swap-noise pretraining: reconstruction error against downstream AUC."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
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
# notebook-end


SPEC = {
    "chapter": 32,
    "chapter_title": "Denoising Autoencoders for Tabular Pretraining",
    "subtitle": "Judge a pretrained representation by a matched downstream comparison, not by its reconstruction loss.",
    "summary": ("A denoising autoencoder is trained to rebuild clean rows from corrupted ones. One demonstration trains one with "
                "swap noise at several strengths and measures both how well it reconstructs and what its features add to a "
                "classifier fitted on a few labeled rows, against an untrained network of the same shape."),
    "title": "Swap-noise pretraining: reconstruction error against downstream value",
    "question": "Does a lower reconstruction error mean the pretrained features help a small labeled model more than an untrained encoder would?",
    "why": ("Reconstruction loss is the number pretraining produces for free, so it is the number people optimize. It says how well the "
            "pretext task was fitted, not whether the target's information survived, and the untrained control is rarely run."),
    "method": ("Fourteen constructed datasets: 20 features driven by four hidden factors plus noise, 3,000 unlabeled rows, 300 labeled rows "
               "and 3,000 new rows. The label depends on a product of two factors and one more factor. A 64-unit hidden-layer network "
               "(wider than the input) is trained on two swap-noise copies of every unlabeled row to rebuild the clean row; the control "
               "is the swap probability (0 gives a plain autoencoder). Its hidden activations are appended to the 20 features of a "
               "logistic regression fitted on the 300 labeled rows and scored by AUC on the new rows. Two controls: the raw features alone, and "
               "the hidden activations of an untrained network with the same shape."),
    "control": {"key": "p_swap", "label": "Swap probability per cell",
                "values": [0, 0.1, 0.3, 0.6], "default": 0.1,
                "value_labels": ["0: plain autoencoder", "0.1: the chapter's usual rate", "0.3", "0.6"]},
    "source_section": "Setting Realistic Expectations",
    "symbols": ("x is a clean row, x-tilde its swap-corrupted copy, f the encoder (the hidden layer) and g the decoder. The pretraining loss "
                "is the mean squared difference between x and g(f(x-tilde)); the downstream score is the AUC of a classifier on the "
                "labeled rows."),
    "explanation": ("With no corruption a wider-than-input network can copy its input, so reconstruction is nearly perfect and the hidden "
                    "layer learns little: its features score a little below those of an untrained network. Swap noise stops the copy and "
                    "forces the network to use the dependencies between features, so reconstruction error rises while downstream AUC "
                    "rises too, reaching about 0.01 above the untrained encoder at the highest rate. Most of the gain over raw features "
                    "is the nonlinear expansion any hidden layer provides, which the untrained control captures as well."),
    "application": ("Run the untrained control (same shape, random weights) beside every pretrained representation, and compare "
                    "all of them with the raw-feature model on matched labeled folds. Keep a pretrained encoder only if it beats "
                    "the control on the downstream metric, and record the result when it does not."),
    "assumptions": ("Constructed data and a scikit-learn network with one hidden layer, trained for 30 epochs, standing in for the "
                    "chapter's deeper PyTorch autoencoder; the downstream model is a logistic regression with a fixed regularization. "
                    "A deeper autoencoder, a different head or a different target could favour pretraining; this setup measures that "
                    "pretraining added little beyond an untrained encoder here, which is a result about this generator, not a verdict on DAEs."),
    "prediction": "At the usual swap rate of 0.1, how do the pretrained features compare with those of an untrained network of the same shape?",
    "prediction_options": ["About 0.05 AUC better", "Within about 0.01 AUC of it", "Clearly worse, because the autoencoder saw only 3,000 rows"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "They score 0.768 against 0.765 for the untrained encoder, a difference of 0.003, although reconstruction error fell from 1.405 to 0.037. Raw features alone score 0.676.",
        "incorrect": "They score 0.768 against 0.765 for the untrained encoder, a difference of 0.003, although reconstruction error fell from 1.405 to 0.037. The pretext task was learned; the extra value was close to nil. Raw features alone score 0.676.",
    },
    "check": "Without any corruption the autoencoder reconstructs almost perfectly. What does that tell you about its downstream value?",
    "answer": ("Nothing good: its reconstruction error is 0.003, the best of all four settings, yet its features score 0.751, below the "
               "untrained encoder's 0.765 and below every swap-noise setting. The network learned to copy its input, and "
               "reconstruction loss ranks the corruption levels in the opposite order from downstream AUC."),
    "provenance": "Constructed example: fourteen seeded synthetic datasets, a small scikit-learn autoencoder with swap noise and a logistic regression, measured by the chapter activity.",
    "apply": [
        "Always fit the same-shape untrained network beside the pretrained one and compare the downstream metric; a gain over raw features alone is not evidence for pretraining.",
        "Do not choose corruption strength by reconstruction loss; choose it by the downstream metric on matched labeled folds.",
        "Pretrain only on the permitted unlabeled population for the assessment you are running (outer-training rows for an inductive assessment).",
        "Record a failed transfer as a result, with the pretraining data identity and corruption settings.",
    ],
    "honesty": "Constructed data; pretraining adds little over an untrained encoder in this generator, which is a property of the setup, not a general claim about denoising autoencoders.",
}

EQUATIONS = [{"tex": r"\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} \bigl\| x_i - g\bigl(f(\tilde x_i)\bigr) \bigr\|^2",
              "alt": "the pretraining loss is the average over rows of the squared distance between the clean row x i and the decoder g applied to the encoder f applied to the corrupted row",
              "basis": "The denoising objective described in words (What a Denoising Autoencoder Does: the target is the original, uncorrupted row); written as an equation by the activity."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    ax, ax2 = axes
    recon = [result["reconstruction_untrained"], result["reconstruction_dae"]]
    ax.bar([0, 1], recon, width=0.55, color=[COLORS["light"], COLORS["teal"]], edgecolor=COLORS["ink"], lw=0.6)
    for x, v in zip([0, 1], recon):
        ax.text(x, v + 0.02 * max(recon), fmt(v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1], ["Untrained\nnetwork", "Trained with\nswap noise"])
    ax.set_ylim(0, max(recon) * 1.15)
    ax.set_ylabel("Reconstruction error on new rows")
    ax.set_xlabel("Autoencoder (lower is better)")

    names = [("raw", "Raw features\nonly"), ("untrained", "Raw + untrained\nencoder"), ("dae", "Raw + trained\nencoder")]
    values = [result["auc_" + k] for k, _ in names]
    colors = [COLORS["light"], COLORS["gold"], COLORS["teal"]]
    pos = np.arange(3)
    ax2.bar(pos, values, width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for p, (k, _) in zip(pos, names):
        dots = result["per_replicate_auc"][k]
        ax2.scatter([p + (i - (len(dots) - 1) / 2) * 0.05 for i in range(len(dots))], dots, s=14, color=COLORS["ink"], zorder=3,
                    label="One dataset" if p == 0 else None)
        ax2.text(p, max(max(dots), result["auc_" + k]) + 0.008, fmt(result["auc_" + k]), ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(pos, [n for _, n in names], fontsize=10)
    ax2.set_ylim(0.5, 0.92)
    ax2.set_ylabel("AUC of the 300-row classifier")
    ax2.set_xlabel(f"Features given to the classifier, swap probability {parameter}")
    ax2.legend(loc="upper left", frameon=False, fontsize=10)


def explain(result, parameter):
    rd, ru = result["reconstruction_dae"], result["reconstruction_untrained"]
    raw, un, dae = result["auc_raw"], result["auc_untrained"], result["auc_dae"]
    gap = float(fmt(dae)) - float(fmt(un))
    over_raw = float(fmt(dae)) - float(fmt(raw))
    interpretation = (
        f"At swap probability {parameter} the trained autoencoder reconstructs new rows with error {fmt(rd)} against {fmt(ru)} for the "
        f"untrained network. Its features give a downstream AUC of {fmt(dae)}; the raw features alone give {fmt(raw)} and an untrained "
        f"encoder of the same shape {fmt(un)}. So {fmt(dae)} - {fmt(un)} = {fmt(gap)} is what pretraining adds beyond a random encoder "
        f"(one standard deviation over datasets: {fmt(result['dae_minus_untrained_sd'])}), while {fmt(dae)} - {fmt(raw)} = {fmt(over_raw)} "
        f"is the gain over raw features.")
    steps = [
        f"Reconstruction: {fmt(rd)} trained against {fmt(ru)} untrained (mean squared error, standardized units).",
        f"Downstream AUC: trained {fmt(dae)}, untrained encoder {fmt(un)}, raw features {fmt(raw)}.",
        f"Pretraining beyond a random encoder: {fmt(dae)} - {fmt(un)} = {signed(gap)}.",
        f"Gain over raw features: {fmt(dae)} - {fmt(raw)} = {signed(over_raw)}.",
    ]
    metrics = {"Reconstruction, trained": fmt(rd), "Reconstruction, untrained": fmt(ru), "AUC raw features": fmt(raw),
               "AUC + untrained encoder": fmt(un), "AUC + trained encoder": fmt(dae)}
    alt = (f"Left: reconstruction error of an untrained network ({fmt(ru)}) and a swap-noise autoencoder ({fmt(rd)}). Right: classifier AUC with "
           f"raw features ({fmt(raw)}), plus an untrained encoder ({fmt(un)}) and plus the trained encoder ({fmt(dae)}), with a dot per dataset.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    values = (0, 0.1, 0.3, 0.6)
    recon = [results[p]["reconstruction_dae"] for p in values]
    auc = [results[p]["auc_dae"] for p in values]
    assert recon[0] < recon[1] < recon[2] < recon[3], "reconstruction error should rise with swap probability"
    assert auc[0] < auc[1] < auc[3] and auc[0] < auc[2], "downstream AUC should rise with swap probability from the plain autoencoder"
    assert 0.003 < results[0.6]["dae_minus_untrained"] < 0.02, "the best swap rate should add about 0.01 beyond the untrained encoder"
    assert abs(results[0.1]["dae_minus_untrained"]) < 0.015, "the usual swap rate should be within about 0.01 of the untrained encoder"
    for p, res in results.items():
        assert res["reconstruction_untrained"] > 0.8 and res["reconstruction_dae"] < 0.5 * res["reconstruction_untrained"], \
            f"the autoencoder should clearly learn the pretext task at {p}"
        assert res["auc_dae"] > res["auc_raw"] + 0.03, f"trained features should beat raw features at {p}"
        assert res["auc_dae"] < res["auc_untrained"] + 0.02, f"pretraining should add little beyond the untrained encoder at {p}"
    assert results[0]["auc_untrained"] - results[0]["auc_dae"] > 0.005, "plain autoencoder should score below the untrained encoder"
    r = results[0.1]
    assert fmt(r["auc_dae"]) == "0.768" and fmt(r["auc_untrained"]) == "0.765" and fmt(r["auc_raw"]) == "0.676", "prediction feedback numbers"
    assert fmt(float(fmt(r["auc_dae"])) - float(fmt(r["auc_untrained"]))) == "0.003", "prediction feedback gap"
    assert fmt(r["reconstruction_untrained"]) == "1.405" and fmt(r["reconstruction_dae"]) == "0.037", "prediction feedback reconstruction"
    z = results[0]
    assert fmt(z["reconstruction_dae"]) == "0.003" and fmt(z["auc_dae"]) == "0.751" and fmt(z["auc_untrained"]) == "0.765", "check answer numbers"
