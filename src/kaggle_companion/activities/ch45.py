"""Chapter 45: Metric Learning and ArcFace. An additive angular margin against a cross-entropy descriptor, scored on identities never trained on."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np

from kaggle_companion.activities._common import clean

SEED = 45
DRAWS = 16                 # independent worlds; every estimate is a mean over draws and every gain is paired
D_SIG, D_NUIS = 8, 24      # identity-bearing dimensions and nuisance (pose, light) dimensions of the 32 image features
TRAIN_IDS, TRAIN_IMAGES = 40, 6
TEST_IDS = 40              # unseen identities, three gallery images and three query images each
SCALE = 16.0               # the chapter's s (feature scale); 16 is enough for 40 classes
EPOCHS = 300


def images(rotation, centers, ids, rng):
    """Image features: identity signal plus larger nuisance noise, mixed by a fixed rotation (a stand-in for backbone features)."""
    signal = centers[ids] + rng.normal(size=(len(ids), D_SIG))
    nuisance = 2.0 * rng.normal(size=(len(ids), D_NUIS))
    return np.column_stack([signal, nuisance]) @ rotation.T


def arc_logits(cos, labels, margin):
    """The chapter's ArcMarginHead: add the margin to the target angle, cos(theta + m), with the stability fix near pi - m."""
    sin = np.sqrt(1 - cos ** 2)
    shifted = np.where(cos > np.cos(np.pi - margin), cos * np.cos(margin) - sin * np.sin(margin),
                       cos - np.sin(np.pi - margin) * margin)
    out = cos.copy()
    out[np.arange(len(labels)), labels] = shifted[np.arange(len(labels)), labels]
    return SCALE * out


def loss_and_grads(params, X, labels, margin, head):
    """Softmax cross-entropy loss and its gradients. head 'ce': plain linear head. head 'arc': normalised features, prototypes and margin."""
    W, P = params
    n = len(X)
    E = X @ W
    onehot = np.eye(len(P))[labels]
    if head == "ce":
        logits = E @ P.T
    else:
        norm_e = np.linalg.norm(E, axis=1, keepdims=True)
        norm_p = np.linalg.norm(P, axis=1, keepdims=True)
        U, V = E / norm_e, P / norm_p
        cos = np.clip(U @ V.T, -1 + 1e-7, 1 - 1e-7)
        logits = arc_logits(cos, labels, margin)
    z = logits - logits.max(1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    loss = -np.mean(np.log(prob[np.arange(n), labels]))
    dlogits = (prob - onehot) / n
    if head == "ce":
        return loss, [X.T @ (dlogits @ P), dlogits.T @ E]
    sin = np.sqrt(1 - cos ** 2)
    stable = cos > np.cos(np.pi - margin)
    slope = np.where(stable, np.cos(margin) + cos * np.sin(margin) / sin, 1.0)
    dcos = SCALE * dlogits * np.where(onehot == 1, slope, 1.0)
    dU, dV = dcos @ V, dcos.T @ U
    dE = (dU - (dU * U).sum(1, keepdims=True) * U) / norm_e      # back through the length normalisation
    dP = (dV - (dV * V).sum(1, keepdims=True) * V) / norm_p
    return loss, [X.T @ dE, dP]


def train(X, labels, head, margin, seed):
    """Full-batch Adam on a linear 32 -> 8 embedding and the class prototypes. Returns the embedding matrix."""
    rng = np.random.default_rng(seed)
    params = [rng.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], D_SIG)), rng.normal(0, 1, (TRAIN_IDS, D_SIG))]
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    for t in range(1, EPOCHS + 1):
        _, grads = loss_and_grads(params, X, labels, margin, head)
        for i, g in enumerate(grads):
            m1[i] = 0.9 * m1[i] + 0.1 * g
            m2[i] = 0.999 * m2[i] + 0.001 * g * g
            params[i] -= 0.02 * (m1[i] / (1 - 0.9 ** t)) / (np.sqrt(m2[i] / (1 - 0.999 ** t)) + 1e-8)
    return params[0]


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def map_at_5(query, query_ids, gallery, gallery_ids):
    """MAP@5 with one true identity per query. Ranked identities are listed once each, as in the chapter's retrieval rule."""
    similarity = unit(query) @ unit(gallery).T
    scores = []
    for sim, truth in zip(similarity, query_ids):
        ranked = []
        for j in np.argsort(-sim):
            if gallery_ids[j] not in ranked:
                ranked.append(gallery_ids[j])
            if len(ranked) == 5:
                break
        scores.append(1.0 / (ranked.index(truth) + 1) if truth in ranked else 0.0)
    return float(np.mean(scores))


def separation(query, query_ids, gallery, gallery_ids):
    """Mean cosine between same-identity pairs and between different-identity pairs."""
    sim = unit(query) @ unit(gallery).T
    same = query_ids[:, None] == gallery_ids[None, :]
    return float(sim[same].mean()), float(sim[~same].mean())


def run(margin):
    rng = np.random.default_rng(SEED)
    rows = {"ce": [], "arc": []}
    for draw in range(DRAWS):
        rotation, _ = np.linalg.qr(rng.normal(size=(D_SIG + D_NUIS, D_SIG + D_NUIS)))
        train_centers, test_centers = rng.normal(size=(TRAIN_IDS, D_SIG)) * 2, rng.normal(size=(TEST_IDS, D_SIG)) * 2
        y = np.repeat(np.arange(TRAIN_IDS), TRAIN_IMAGES)
        X = images(rotation, train_centers, y, rng)
        ids_seen, ids_unseen = np.repeat(np.arange(TRAIN_IDS), 3), np.repeat(np.arange(TEST_IDS), 3)
        sets = {"seen": (images(rotation, train_centers, ids_seen, rng), images(rotation, train_centers, ids_seen, rng), ids_seen),
                "unseen": (images(rotation, test_centers, ids_unseen, rng), images(rotation, test_centers, ids_unseen, rng), ids_unseen)}
        for name, head, m in (("ce", "ce", 0.0), ("arc", "arc", float(margin))):
            W = train(X, y, head, m, seed=draw)
            row = {}
            for split, (gallery, query, ids) in sets.items():
                row["map_" + split] = map_at_5(query @ W, ids, gallery @ W, ids)
            same, diff = separation(sets["unseen"][1] @ W, ids_unseen, sets["unseen"][0] @ W, ids_unseen)
            row.update({"same_cos": same, "diff_cos": diff})
            rows[name].append(row)
    mean = {n: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for n, rs in rows.items()}
    gain = np.array([a["map_unseen"] - c["map_unseen"] for a, c in zip(rows["arc"], rows["ce"])])
    return clean({
        "margin": margin, "ce": mean["ce"], "arc": mean["arc"],
        "gain_unseen": gain.mean(), "gain_se": gain.std(ddof=1) / np.sqrt(DRAWS), "win_rate": float((gain > 0).mean()),
        "per_draw": {"ce": [r["map_unseen"] for r in rows["ce"]], "arc": [r["map_unseen"] for r in rows["arc"]]},
        "draws": DRAWS, "train_identities": TRAIN_IDS, "unseen_identities": TEST_IDS,
    })
# notebook-end


SPEC = {
    "chapter": 45,
    "chapter_title": "Metric Learning and ArcFace",
    "subtitle": "Separate the training head, the descriptor and the ranking rule, and judge the margin on identities the model never saw.",
    "summary": ("A classifier over training identities cannot name a new one, but its descriptor can still retrieve it. One "
                "demonstration trains an ArcFace-style head and a plain cross-entropy head and measures retrieval of identities "
                "that were never in training, by angular margin."),
    "title": "ArcFace margin against a cross-entropy descriptor, scored on unseen identities",
    "question": "How much does the angular margin improve retrieval of unseen identities over a cross-entropy descriptor, and where does a larger margin stop helping?",
    "why": ("The margin m is one of the two head settings the chapter calls starting candidates rather than a universal winner. "
            "Sixteen paired trainings, each judged on query identities absent from training, show how the gain moves with m and "
            "when training becomes unreliable."),
    "method": ("Constructed image features: 40 training identities with 6 images each, 32 features of which 8 carry identity "
               "and 24 are larger nuisance noise (pose and light), mixed by a random rotation. A linear embedding to 8 dimensions "
               "is trained with Adam in two ways from the same start: a plain linear cross-entropy head (the reference descriptor), "
               "and the chapter's ArcMarginHead with scale s = 16 and the control's margin m, whose gradients are written out in "
               "numpy. Retrieval is scored as MAP@5 on 40 further identities (3 gallery and 3 query images each), with every "
               "ranked identity listed once, as in the chapter's retrieval rule, averaged over 16 draws."),
    "control": {"key": "margin", "label": "Angular margin m (radians)",
                "values": [0, 0.5, 0.8, 1.2], "default": 0.5,
                "value_labels": ["0: normalised softmax, no margin", "0.5: the chapter's starting value", "0.8", "1.2: a very large margin"]},
    "source_section": "Two Head Settings: Scale and Margin",
    "symbols": ("theta_y is the angle between an embedding and the prototype of its true identity, m the additive angular "
                "margin in radians and s the feature scale. MAP@5 averages, over query images, the reciprocal rank of the true "
                "identity among the five best distinct gallery identities (0 if absent)."),
    "explanation": ("The margin makes the training target harder, so the embedding has to put each identity's images at a "
                    "smaller angle to its prototype than a plain cross-entropy descriptor does. That compactness carries over to "
                    "identities that were never trained on, because retrieval compares angles and never uses the head. Normalising "
                    "with no margin (m = 0) already gives part of the gain over the plain head; the margin adds the rest. A margin "
                    "that is too large asks for more room between 40 prototypes in 8 dimensions than exists, and training becomes unreliable."),
    "application": ("Build a reference descriptor with plain cross-entropy, add the margin head as a separate comparison, score both "
                    "with query identities held out of training, and treat the margin as a setting to tune, not a constant to copy."),
    "assumptions": ("Constructed features rather than images, a linear embedding standing in for a backbone, and s = 16 instead "
                    "of the chapter's example 64 because there are only 40 classes. The cross-entropy reference uses an unnormalised "
                    "linear head and is scored on the same normalised embedding. Seen and unseen identities score almost the same "
                    "here, because a linear embedding does not memorise identities; a deep backbone can, so keep reporting them separately."),
    "prediction": "Which margin gives the best retrieval of unseen identities, and what happens at the largest margin of 1.2?",
    "prediction_options": ["The largest margin wins: more margin, better retrieval",
                           "A moderate margin (0.5 to 0.8) wins, and 1.2 gives most of the gain back",
                           "No margin matters: every setting retrieves about equally well"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "MAP@5 is 0.640 for cross-entropy, 0.778 at m = 0.5 and 0.781 at m = 0.8, but only 0.686 at m = 1.2, and training fails in some draws.",
        "incorrect": "MAP@5 is 0.640 for cross-entropy, 0.778 at m = 0.5 and 0.781 at m = 0.8, but only 0.686 at m = 1.2, where training fails in some draws.",
    },
    "check": "Margins 0.5 and 0.8 retrieve almost equally well. Which would you carry into a real competition, and why not simply the largest margin?",
    "answer": ("Take the smaller of the two, because the gain is flat between them (0.778 against 0.781) while the failure region "
               "begins somewhere above: at m = 1.2 the average gain falls to 0.047 with a standard error of 0.032, and the worst "
               "draw scores 0.452. A margin is tuned on development identities and then frozen, and the chapter lists larger margins as "
               "a source of unstable optimisation."),
    "provenance": "Constructed example: seeded synthetic identity features and a numpy ArcFace head, measured by the chapter activity.",
    "apply": [
        "Keep a cross-entropy descriptor as the reference and compare the margin head against it with the same identities, split and budget.",
        "Hold out whole identities for assessment, and list each identity once in the top five, as in the chapter's retrieval rule.",
        "Tune the margin on development identities over a few seeds, watching the worst seed as well as the mean, then freeze it.",
        "At inference use only the normalised descriptor: the margin is a training-time change and the target is unknown.",
    ],
    "honesty": "Constructed data and a linear embedding; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"z_y = s\cos(\theta_y + m), \qquad z_c = s\cos\theta_c \;\; (c \ne y)",
              "alt": "The target logit z y equals s times the cosine of theta y plus m; every other logit z c equals s times the cosine of theta c",
              "basis": "The additive angular margin of ArcFace Implementation; the manuscript gives it in code (ArcMarginHead), not as a display equation."}]
NCOLS = 2
HEIGHT = 4.4


def draw(axes, result, parameter):
    ax, bx = axes
    ce, arc = result["per_draw"]["ce"], result["per_draw"]["arc"]
    for a, b in zip(ce, arc):
        ax.plot([0, 1], [a, b], color=COLORS["light"], lw=1, zorder=1)
    ax.scatter([0] * len(ce), ce, s=22, color=COLORS["grey"], zorder=2, label="One draw")
    ax.scatter([1] * len(arc), arc, s=22, color=COLORS["grey"], zorder=2)
    for x, v, c in ((0, result["ce"]["map_unseen"], COLORS["ink"]), (1, result["arc"]["map_unseen"], COLORS["teal"])):
        ax.scatter([x], [v], s=140, color=c, marker="D", zorder=3, edgecolor="white", linewidth=1)
        ax.text(x + (-0.12 if x == 0 else 0.12), v, fmt(v), ha="right" if x == 0 else "left", va="center", fontsize=10)
    ax.set_xticks([0, 1], ["Cross-entropy\ndescriptor", f"ArcFace head\nm = {parameter}"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("MAP@5 on 40 unseen identities")
    ax.set_xlabel("Training head (diamond = mean over draws)")
    ax.legend(loc="lower center", frameon=False, fontsize=10)

    names = ["Seen identities\n(new images)", "Unseen identities"]
    xs = np.arange(2)
    ce_map = [result["ce"]["map_seen"], result["ce"]["map_unseen"]]
    arc_map = [result["arc"]["map_seen"], result["arc"]["map_unseen"]]
    bx.bar(xs - 0.2, ce_map, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Cross-entropy")
    bx.bar(xs + 0.2, arc_map, width=0.4, color=COLORS["teal"], edgecolor=COLORS["ink"], lw=0.6, label=f"ArcFace, m = {parameter}")
    for x, a, b in zip(xs, ce_map, arc_map):
        bx.text(x - 0.2, a + 0.015, fmt(a), ha="center", va="bottom", fontsize=10)
        bx.text(x + 0.2, b + 0.015, fmt(b), ha="center", va="bottom", fontsize=10)
    bx.set_xticks(xs, names)
    bx.set_ylim(0, 1.0)
    bx.set_ylabel("MAP@5")
    bx.set_xlabel("Query identities")
    bx.legend(loc="upper right", frameon=False, fontsize=10, ncols=1)


def explain(result, parameter):
    ce, arc = result["ce"], result["arc"]
    gain, se, win = result["gain_unseen"], result["gain_se"], result["win_rate"]
    worst = min(result["per_draw"]["arc"])
    if parameter == 0 and gain > 2 * se:
        verdict = ("With m = 0 there is no margin, so this gain comes only from normalising features and prototypes and "
                   "scaling by s; compare it with the gain at m = 0.5 to see what the margin itself adds.")
    elif gain > 2 * se:
        verdict = "The margin head is clearly better than the reference."
    else:
        verdict = (f"The gain is not clearly different from zero (under two standard errors, {fmt(2 * se)}), and the "
                   f"worst draw scores {fmt(worst)}, so this margin is not a safe choice.")
    interpretation = (
        f"With margin {parameter}, retrieval of unseen identities reaches MAP@5 {fmt(arc['map_unseen'])} against {fmt(ce['map_unseen'])} "
        f"for the cross-entropy descriptor: {fmt(arc['map_unseen'])} - {fmt(ce['map_unseen'])} = {fmt(round(arc['map_unseen'], 3) - round(ce['map_unseen'], 3))} "
        f"(standard error {fmt(se)}, better in {round(100 * win)}% of the {result['draws']} draws). {verdict} "
        f"On identities seen in training the margin head scores {fmt(arc['map_seen'])}, close to its unseen score. "
        f"The reference scores {fmt(ce['map_seen'])} on seen identities.")
    steps = [f"Unseen-identity MAP@5, margin head minus reference: {fmt(arc['map_unseen'])} - {fmt(ce['map_unseen'])} = {signed(round(arc['map_unseen'], 3) - round(ce['map_unseen'], 3))}.",
             f"Seen-identity MAP@5, margin head minus reference: {fmt(arc['map_seen'])} - {fmt(ce['map_seen'])} = {signed(arc['map_seen'] - ce['map_seen'])}.",
             f"Paired standard error of the unseen gain: {fmt(se)}; the margin head wins in {round(100 * win)}% of draws.",
             f"Worst single draw of the margin head: {fmt(worst)} (reference worst: {fmt(min(result['per_draw']['ce']))})."]
    metrics = {"Reference MAP@5, unseen": fmt(ce["map_unseen"]), "ArcFace MAP@5, unseen": fmt(arc["map_unseen"]),
               "Paired gain (standard error)": f"{signed(gain)} ({fmt(se)})", "ArcFace wins in draws": f"{round(100 * win)}%",
               "ArcFace worst draw": fmt(worst), "ArcFace MAP@5, seen": fmt(arc["map_seen"])}
    alt = (f"Two panels. Left, a paired dot plot of unseen-identity MAP@5 for each of {result['draws']} draws, cross-entropy against "
           f"ArcFace with margin {parameter}, with diamonds at the means {fmt(ce['map_unseen'])} and {fmt(arc['map_unseen'])}. "
           f"Right, MAP@5 for the two heads on seen and on unseen identities.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    # The hand-derived gradients must match finite differences, for both heads.
    rng = np.random.default_rng(3)
    Xg = rng.normal(size=(12, 6))
    yg = rng.integers(0, 4, 12)
    for head, m in (("ce", 0.0), ("arc", 0.5)):
        params = [rng.normal(size=(6, 3)), rng.normal(size=(4, 3))]
        _, grads = loss_and_grads(params, Xg, yg, m, head)
        for k in (0, 1):
            num = np.zeros_like(params[k])
            for idx in np.ndindex(*params[k].shape):
                hi = [p.copy() for p in params]
                lo = [p.copy() for p in params]
                hi[k][idx] += 1e-6
                lo[k][idx] -= 1e-6
                num[idx] = (loss_and_grads(hi, Xg, yg, m, head)[0] - loss_and_grads(lo, Xg, yg, m, head)[0]) / 2e-6
            assert np.allclose(num, grads[k], atol=1e-6), f"gradient mismatch for the {head} head"
    # The margin logit is s * cos(theta + m) when the angle is in the stable range.
    cos = np.array([[0.6, 0.2, -0.1]])
    assert np.isclose(arc_logits(cos, np.array([0]), 0.5)[0, 0], SCALE * np.cos(np.arccos(0.6) + 0.5)), "target logit is s cos(theta + m)"
    assert np.isclose(arc_logits(cos, np.array([0]), 0.5)[0, 1], SCALE * 0.2), "non-target logits are unchanged"
    g = {m: res["gain_unseen"] for m, res in results.items()}
    assert g[0] > 2 * results[0]["gain_se"], "normalising alone gives part of the gain (explanation text)"
    assert g[0] < 0.5 * g[0.5] and g[0.8] > g[0.5] - 0.02, "gain rises with the margin up to 0.5 to 0.8"
    for m in (0.5, 0.8):
        assert results[m]["gain_unseen"] > 4 * results[m]["gain_se"] and results[m]["win_rate"] == 1.0, f"margin {m} should beat the reference in every draw"
    assert results[1.2]["gain_unseen"] < 2 * results[1.2]["gain_se"], "the largest margin should not be clearly better than the reference"
    assert results[1.2]["gain_unseen"] < 0.5 * g[0.5], "the largest margin loses most of the gain"
    assert results[1.2]["win_rate"] < 0.8 and min(results[1.2]["per_draw"]["arc"]) < min(results[1.2]["per_draw"]["ce"]), "training fails in some draws at 1.2"
    for m, res in results.items():
        assert abs(res["arc"]["map_seen"] - res["arc"]["map_unseen"]) < 0.04, "seen and unseen should score about the same (assumptions text)"
    r5, r8, r12 = results[0.5], results[0.8], results[1.2]
    assert fmt(r5["ce"]["map_unseen"]) == "0.640" and fmt(r5["arc"]["map_unseen"]) == "0.778"
    assert fmt(r8["arc"]["map_unseen"]) == "0.781" and fmt(r12["arc"]["map_unseen"]) == "0.686"
    assert fmt(r12["gain_unseen"]) == "0.047" and fmt(r12["gain_se"]) == "0.032"
    assert fmt(min(r12["per_draw"]["arc"])) == "0.452"
    assert abs(r5["arc"]["map_unseen"] - r8["arc"]["map_unseen"]) < 0.01, "flat between 0.5 and 0.8"
