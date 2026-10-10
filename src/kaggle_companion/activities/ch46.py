"""Chapter 46: Segmentation Architectures. Pooled and per-image Dice, the threshold each one picks, and the empty-image share."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from scipy import ndimage

from kaggle_companion.activities._common import clean

SEED = 46
DRAWS = 10                 # independent image sets; every estimate is a mean over draws
SIZE = 32                  # images are 32 x 32 pixels
N_DEV, N_HELD = 150, 300   # development images (choose the rule) and held-out images (score it)
THRESHOLDS = np.linspace(0.2, 0.9, 15)
AREAS = (0, 5, 10, 20)     # minimum object sizes in pixels tried by the post-processing rule


def generate(n, empty_share, rng):
    """Masks with one disc each (radius 2 to 8 px; a share of the images is empty) and a stand-in model's probability map.
    The map is the blurred mask plus spatially smooth noise, so it misses small objects and speckles over empty images."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    masks = np.zeros((n, SIZE, SIZE), bool)
    for i in np.where(rng.random(n) >= empty_share)[0]:
        radius = rng.choice([2, 3, 4, 6, 8], p=[0.25, 0.25, 0.2, 0.15, 0.15])
        cy, cx = rng.integers(radius + 1, SIZE - radius - 1, 2)
        masks[i] = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    blur = ndimage.gaussian_filter(masks.astype(float), (0, 1.2, 1.2))
    field = ndimage.gaussian_filter(rng.normal(size=masks.shape), (0, 1.5, 1.5))
    field /= field.std()
    return masks, 1 / (1 + np.exp(-(9 * (blur - 0.35) + 1.44 * field - 1.0)))


def pooled_dice(pred, target):
    """Intersection and support summed over the whole set, then one ratio."""
    return (2 * (pred & target).sum() + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def image_dice(pred, target):
    """One ratio per image, averaged later. An empty target with an empty prediction scores 1 (the rule stated in the text)."""
    inter = (pred & target).sum((1, 2))
    total = pred.sum((1, 2)) + target.sum((1, 2))
    return np.where(total == 0, 1.0, 2 * inter / np.maximum(total, 1))


def remove_small(pred, area):
    """Delete connected components smaller than `area` pixels."""
    if area == 0:
        return pred
    out = pred.copy()
    for i in np.where(pred.any((1, 2)))[0]:
        labels, count = ndimage.label(pred[i])
        for k, size in enumerate(ndimage.sum(pred[i], labels, range(1, count + 1)), 1):
            if size < area:
                out[i][labels == k] = False
    return out


def run(empty_share):
    rng = np.random.default_rng(SEED)
    keys = ("pooled_rule", "image_rule", "area_rule")
    rows = {k: {"pooled": [], "image": []} for k in keys}
    chosen = {"pooled_rule": [], "image_rule": [], "area_rule": [], "area": []}
    curve_pooled, curve_image = [], []
    for _ in range(DRAWS):
        dev_mask, dev_prob = generate(N_DEV, empty_share, rng)
        held_mask, held_prob = generate(N_HELD, empty_share, rng)
        # Rule 1: the threshold that maximises pooled Dice on the development images.
        t_pooled = THRESHOLDS[np.argmax([pooled_dice(dev_prob > t, dev_mask) for t in THRESHOLDS])]
        # Rule 2: the threshold that maximises mean per-image Dice (empty rule included).
        t_image = THRESHOLDS[np.argmax([image_dice(dev_prob > t, dev_mask).mean() for t in THRESHOLDS])]
        # Rule 3: threshold and minimum object size that maximise mean per-image Dice.
        best = max(((image_dice(remove_small(dev_prob > t, a), dev_mask).mean(), t, a) for a in AREAS for t in THRESHOLDS[::2]),
                   key=lambda s: s[0])
        preds = {"pooled_rule": held_prob > t_pooled, "image_rule": held_prob > t_image,
                 "area_rule": remove_small(held_prob > best[1], best[2])}
        for k, pred in preds.items():
            rows[k]["pooled"].append(pooled_dice(pred, held_mask))
            rows[k]["image"].append(image_dice(pred, held_mask).mean())
        chosen["pooled_rule"].append(t_pooled)
        chosen["image_rule"].append(t_image)
        chosen["area_rule"].append(best[1])
        chosen["area"].append(best[2])
        curve_pooled.append([pooled_dice(held_prob > t, held_mask) for t in THRESHOLDS])
        curve_image.append([image_dice(held_prob > t, held_mask).mean() for t in THRESHOLDS])
    gain = np.array(rows["area_rule"]["image"]) - np.array(rows["pooled_rule"]["image"])
    return clean({
        "empty_share": empty_share,
        "rules": {k: {"pooled": np.mean(v["pooled"]), "image": np.mean(v["image"])} for k, v in rows.items()},
        "threshold": {k: np.mean(chosen[k]) for k in keys}, "min_area": np.mean(chosen["area"]),
        "gain_image": gain.mean(), "gain_se": gain.std(ddof=1) / np.sqrt(DRAWS), "win_rate": float((gain > 0).mean()),
        "thresholds": THRESHOLDS, "curve_pooled": np.mean(curve_pooled, 0), "curve_image": np.mean(curve_image, 0),
        "draws": DRAWS, "dev_images": N_DEV, "held_images": N_HELD,
    })
# notebook-end


SPEC = {
    "chapter": 46,
    "chapter_title": "Segmentation Architectures",
    "subtitle": "Fix the Dice reduction, the empty-mask rule and the threshold before comparing models, and keep each choice on development images.",
    "summary": ("Pooled Dice and mean per-image Dice can rank the same masks very differently. One demonstration tunes a "
                "threshold for each, adds a minimum-size rule, and scores all three on held-out images as the share of empty "
                "images grows."),
    "title": "Three mask rules scored by pooled and per-image Dice, by the share of empty images",
    "question": "Which threshold and post-processing rule is best depends on how Dice is averaged: how large is that effect, and when does it appear?",
    "why": ("The chapter asks to specify the Dice reduction and the empty-target behaviour before reporting a score. In a "
            "competition with many empty images that choice decides which threshold wins, so a model compared under the wrong "
            "reduction is ranked under the wrong rule."),
    "method": ("Constructed 32 x 32 images with one disc each (radius 2 to 8 pixels), where the control sets the share of "
               "images with no disc. A stand-in for a trained network supplies the probability map: the blurred mask plus "
               "spatially smooth noise, so small objects are weak and empty images carry speckle. Three rules are chosen on 150 "
               "development images: the threshold that maximises pooled Dice, the threshold that maximises mean per-image Dice "
               "(an empty prediction on an empty image scores 1), and a threshold plus a minimum object size (0, 5, 10 or 20 "
               "pixels, via connected components) for per-image Dice. Each is scored on 300 held-out images with both "
               "reductions, averaged over 10 draws."),
    "control": {"key": "empty_share", "label": "Share of images with an empty mask",
                "values": [0, 0.25, 0.5, 0.75], "default": 0.5,
                "value_labels": ["0: every image has an object", "0.25", "0.5: half are empty", "0.75: most are empty"]},
    "source_section": "Segmentation Losses",
    "symbols": ("P and Y are the predicted and true masks of one image, and Dice = 2|P and Y| / (|P| + |Y|). Pooled Dice "
                "sums |P and Y|, |P| and |Y| over all images before dividing; mean per-image Dice divides within each image "
                "and then averages, scoring 1 when both masks are empty."),
    "explanation": ("Pooled Dice is dominated by the pixels of large objects, so the best threshold under it tolerates speckle "
                    "in empty images. Per-image Dice gives every empty image a score of 0 or 1, so the same speckle costs a whole "
                    "image each time. The two reductions pick different thresholds, and the per-image threshold climbs as the share "
                    "of empty images grows. A minimum-size rule removes small speckle without raising the threshold that real objects need."),
    "application": ("Write down the official reduction and empty-mask rule first, choose the threshold and any minimum-size "
                    "rule on development images with that exact metric, and report the pooled number only as a second view."),
    "assumptions": ("Constructed images and a generated probability map stand in for a network. The speckle in this generator is "
                    "made of small blobs, which is why a minimum-size rule works so well here; real false positives can be large "
                    "and need a different remedy. One disc per image, a single class, and a threshold grid of 15 values."),
    "prediction": "With half the images empty, what mean per-image Dice does the pooled-Dice threshold reach on held-out images?",
    "prediction_options": ["About the same as its pooled Dice, near 0.89", "Far lower than its pooled Dice, near 0.72",
                           "Higher than its pooled Dice, because empty images score 1"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": "Its pooled Dice is 0.894 but its mean per-image Dice is only 0.723: speckle on the empty images costs a whole image each.",
        "incorrect": "Its pooled Dice is 0.894 but its mean per-image Dice is only 0.723: each speckled empty image scores 0, whatever its few pixels do to the pooled ratio.",
    },
    "check": "Tuning the threshold directly for per-image Dice fixes that, yet pooled Dice falls. What repairs both, and what does it depend on?",
    "answer": ("The per-image threshold rises to 0.825 to suppress speckle, which also shrinks real masks, so pooled Dice falls from 0.894 to "
               "0.855. A minimum-size rule keeps the threshold near 0.59 and deletes components under about 7 pixels, reaching 0.872 "
               "per-image and 0.903 pooled. That works here because the speckle blobs are small; it is a property of this generator."),
    "provenance": "Constructed example: seeded synthetic masks and a generated probability map, measured by the chapter activity.",
    "apply": [
        "Write the official Dice reduction and the empty-mask rule into the validation code before comparing any model.",
        "Choose the threshold, and a minimum object size if the host scores per image, on development images, then freeze both.",
        "Report the share of empty images and the score on empty and non-empty images separately.",
        "Check the minimum-size rule against false positives that are large, not only speckle.",
    ],
    "honesty": "Constructed data and a generated probability map; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"\mathrm{Dice}_{\mathrm{pooled}}=\frac{2\sum_i |P_i\cap Y_i|}{\sum_i |P_i|+\sum_i |Y_i|},\qquad \mathrm{Dice}_{\mathrm{image}}=\frac1N\sum_i\frac{2|P_i\cap Y_i|}{|P_i|+|Y_i|}",
              "alt": "Pooled Dice sums intersections and supports over all images before dividing; per-image Dice divides within each image and averages the N ratios",
              "basis": "The two reductions the chapter contrasts in Segmentation Losses (mean_binary_image_dice in code); the manuscript gives no display equation, so these are the activity's labels."}]
NCOLS = 2
HEIGHT = 4.4
RULE_NAMES = {"pooled_rule": "Tuned for\npooled Dice", "image_rule": "Tuned for\nper-image Dice", "area_rule": "Per-image Dice\n+ minimum size"}


def draw(axes, result, parameter):
    ax, bx = axes
    xs = np.arange(3)
    keys = list(RULE_NAMES)
    image = [result["rules"][k]["image"] for k in keys]
    pooled = [result["rules"][k]["pooled"] for k in keys]
    ax.bar(xs - 0.2, image, width=0.4, color=COLORS["terracotta"], edgecolor=COLORS["ink"], lw=0.6, label="Mean per-image Dice")
    ax.bar(xs + 0.2, pooled, width=0.4, color=COLORS["light"], edgecolor=COLORS["ink"], lw=0.6, label="Pooled Dice")
    for x, a, b in zip(xs, image, pooled):
        ax.text(x - 0.2, a + 0.008, fmt(a), ha="center", va="bottom", fontsize=10)
        ax.text(x + 0.2, b + 0.008, fmt(b), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs, [RULE_NAMES[k] for k in keys])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Dice on 300 held-out images")
    ax.set_xlabel("Rule chosen on development images")
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncols=2)

    t = result["thresholds"]
    bx.plot(t, result["curve_pooled"], color=COLORS["navy"], lw=2, label="Pooled Dice")
    bx.plot(t, result["curve_image"], color=COLORS["terracotta"], lw=2, label="Mean per-image Dice")
    for key, color in (("pooled_rule", COLORS["navy"]), ("image_rule", COLORS["terracotta"])):
        bx.axvline(result["threshold"][key], color=color, ls=(0, (4, 3)), lw=1.2)
    bx.set_xlabel("Probability threshold (dashed: mean threshold each rule chose)")
    bx.set_ylabel("Dice on 300 held-out images")
    low = min(min(result["curve_image"]), min(result["curve_pooled"]))
    bx.set_ylim(min(0.4, np.floor(10 * low) / 10), 1.0)   # show the whole curve, never clip it
    bx.legend(loc="lower right", frameon=False, fontsize=10)


def explain(result, parameter):
    rules, th = result["rules"], result["threshold"]
    p, i, a = rules["pooled_rule"], rules["image_rule"], rules["area_rule"]
    gain = result["gain_image"]
    interpretation = (
        f"With {round(100 * parameter)}% of images empty, the threshold tuned for pooled Dice ({fmt(th['pooled_rule'])}) scores pooled Dice "
        f"{fmt(p['pooled'])} but mean per-image Dice {fmt(p['image'])}. Tuning for per-image Dice moves the threshold to {fmt(th['image_rule'])} "
        f"and gives {fmt(i['image'])} per image and {fmt(i['pooled'])} pooled. Adding a minimum object size (mean {fmt(result['min_area'], 1)} pixels) "
        f"reaches {fmt(a['image'])} per image: {fmt(a['image'])} - {fmt(p['image'])} = {fmt(round(a['image'], 3) - round(p['image'], 3))} over the pooled-Dice rule "
        f"(standard error {fmt(result['gain_se'])}, better in {round(100 * result['win_rate'])}% of draws)."
        + (" With no empty images the three rules score close to each other per image, so the choice hardly matters; "
              "per-image Dice still sits below pooled Dice because a poorly found small disc weighs as much as a large one." if parameter == 0 else ""))
    steps = [f"Pooled-Dice rule, per-image minus pooled: {fmt(p['image'])} - {fmt(p['pooled'])} = {signed(p['image'] - p['pooled'])}.",
             f"Per-image rule, pooled Dice: {fmt(i['pooled'])} - {fmt(p['pooled'])} = {signed(i['pooled'] - p['pooled'])} against the pooled rule.",
             f"Minimum-size rule over the pooled-Dice rule, per image: {fmt(a['image'])} - {fmt(p['image'])} = {signed(round(a['image'], 3) - round(p['image'], 3))}.",
             f"Thresholds chosen: pooled {fmt(th['pooled_rule'])}, per-image {fmt(th['image_rule'])}, with minimum size {fmt(th['area_rule'])}."]
    metrics = {"Pooled rule, per-image Dice": fmt(p["image"]), "Pooled rule, pooled Dice": fmt(p["pooled"]),
               "Per-image rule, per-image Dice": fmt(i["image"]), "Minimum-size rule, per-image Dice": fmt(a["image"]),
               "Minimum-size rule, pooled Dice": fmt(a["pooled"]), "Gain over pooled rule (SE)": f"{signed(gain)} ({fmt(result['gain_se'])})"}
    alt = (f"Two panels at {round(100 * parameter)}% empty images. Left, held-out per-image and pooled Dice for three rules. "
           f"Right, both Dice values against the probability threshold, with dashed lines at the thresholds the pooled and per-image "
           f"rules chose, {fmt(th['pooled_rule'])} and {fmt(th['image_rule'])}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    # Dice reductions on a tiny hand example: one 4-pixel object found fully, one empty image with 2 false pixels.
    target = np.zeros((2, 4, 4), bool)
    target[0, :2, :2] = True
    pred = target.copy()
    pred[1, 0, :2] = True
    assert np.isclose(pooled_dice(pred, target), 2 * 4 / (6 + 4)) and np.isclose(image_dice(pred, target).mean(), (1 + 0) / 2)
    assert np.allclose(image_dice(np.zeros_like(target), np.zeros_like(target)), 1.0), "empty vs empty scores 1"
    keep = remove_small(np.array([[[1, 1, 0], [0, 0, 0], [0, 0, 1]]], bool), 2)
    assert keep.sum() == 2, "components under the minimum size are deleted"
    zero, quarter, half, three = results[0], results[0.25], results[0.5], results[0.75]
    assert zero["gain_image"] < 0.03, "with no empty images the rules agree"
    zr = zero["rules"]
    assert max(v["image"] for v in zr.values()) - min(v["image"] for v in zr.values()) < 0.03, "rules close per image at 0"
    assert all(v["image"] < v["pooled"] for v in zr.values()), "per-image below pooled at 0"
    for r in (quarter, half, three):
        rules = r["rules"]
        assert rules["pooled_rule"]["pooled"] - rules["pooled_rule"]["image"] > 0.1, "pooled-rule per-image Dice far below its pooled Dice"
        assert r["gain_image"] > 4 * r["gain_se"] and r["win_rate"] == 1.0, "minimum-size rule clearly wins per image"
        assert rules["area_rule"]["pooled"] > rules["image_rule"]["pooled"], "minimum-size rule keeps pooled Dice higher than the per-image threshold"
    thr = [results[v]["threshold"]["image_rule"] for v in (0, 0.25, 0.5, 0.75)]
    assert thr[0] < thr[1] < thr[2] < thr[3], "the per-image threshold climbs with the empty share"
    assert half["threshold"]["image_rule"] - half["threshold"]["pooled_rule"] > 0.1, "the two reductions choose different thresholds"
    assert half["rules"]["image_rule"]["pooled"] < half["rules"]["pooled_rule"]["pooled"] - 0.02, "per-image threshold costs pooled Dice"
    for r in results.values():
        assert r["rules"]["area_rule"]["image"] >= r["rules"]["image_rule"]["image"], "adding the area rule never hurts per-image Dice"
    p, i, a = half["rules"]["pooled_rule"], half["rules"]["image_rule"], half["rules"]["area_rule"]
    assert fmt(p["pooled"]) == "0.894" and fmt(p["image"]) == "0.723"
    assert fmt(half["threshold"]["image_rule"]) == "0.825" and fmt(i["pooled"]) == "0.855"
    assert fmt(half["threshold"]["area_rule"]) == "0.590" and fmt(half["min_area"], 0) == "7"
    assert fmt(a["image"]) == "0.872" and fmt(a["pooled"]) == "0.903"
