"""Chapter 49: Text Feature Engineering. Word and character TF-IDF when only the assessment text has typos."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from kaggle_companion.activities._common import clean

SEED = 49
DRAWS = 8                  # independent corpora; every estimate is a mean over draws
N_DOCS, N_CLASSES, TITLE_WORDS = 800, 4, 7
LETTERS = np.array(list("abcdefghijklmnopqrstuvwxyz"))


def invent_words(rng, n):
    """Pronounceable-enough made-up words of 5 to 9 letters (no real language, so nothing is memorised from outside)."""
    return ["".join(rng.choice(LETTERS, rng.integers(5, 10))) for _ in range(n)]


def misspell(word, rng):
    """One typo: substitute a letter, drop a letter or swap two neighbours."""
    i, kind = rng.integers(len(word)), rng.integers(3)
    if kind == 0:
        return word[:i] + str(rng.choice(LETTERS)) + word[i + 1:]
    if kind == 1 and len(word) > 3:
        return word[:i] + word[i + 1:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:] if i < len(word) - 1 else word


def titles(n, vocabulary, rng, typo_rate=0.0):
    """Product-style titles: 30% of words come from the class's own list, the rest from a shared Zipf-weighted background."""
    own, background = vocabulary
    weights = 1 / np.arange(1, len(background) + 1)
    weights /= weights.sum()
    labels = rng.integers(0, N_CLASSES, n)
    docs = []
    for c in labels:
        words = [own[c][rng.integers(len(own[c]))] if rng.random() < 0.3 else background[rng.choice(len(background), p=weights)]
                 for _ in range(TITLE_WORDS)]
        docs.append(" ".join(misspell(w, rng) if rng.random() < typo_rate else w for w in words))
    return docs, labels


def coverage(train_docs, test_docs):
    """Step Zero: the share of assessment tokens (counted with repeats, so weighted by frequency) seen in training."""
    seen = set(" ".join(train_docs).split())
    tokens = " ".join(test_docs).split()
    return float(np.mean([t in seen for t in tokens]))


def run(typo_rate):
    rng = np.random.default_rng(SEED)
    names = ("word", "char", "both")
    rows = {k: {"log_loss": [], "accuracy": []} for k in names}
    cover = []
    for _ in range(DRAWS):
        vocabulary = ([invent_words(rng, 40) for _ in range(N_CLASSES)], invent_words(rng, 200))
        train, y_train = titles(N_DOCS, vocabulary, rng)                        # clean training text
        test, y_test = titles(N_DOCS, vocabulary, rng, typo_rate)               # typos appear only here
        cover.append(coverage(train, test))
        word = TfidfVectorizer(analyzer="word", sublinear_tf=True)             # vocabularies are fitted on training text only
        char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
        Wtr, Ctr, Wte, Cte = word.fit_transform(train), char.fit_transform(train), word.transform(test), char.transform(test)
        for name, A, B in (("word", Wtr, Wte), ("char", Ctr, Cte), ("both", hstack([Wtr, Ctr]).tocsr(), hstack([Wte, Cte]).tocsr())):
            p = LogisticRegression(C=10, max_iter=500).fit(A, y_train).predict_proba(B)
            rows[name]["log_loss"].append(log_loss(y_test, p, labels=range(N_CLASSES)))
            rows[name]["accuracy"].append((p.argmax(1) == y_test).mean())
    advantage = np.array(rows["word"]["log_loss"]) - np.array(rows["char"]["log_loss"])      # positive: characters are better
    return clean({
        "typo_rate": typo_rate, "coverage": np.mean(cover),
        **{k: {"log_loss": np.mean(v["log_loss"]), "accuracy": np.mean(v["accuracy"])} for k, v in rows.items()},
        "char_advantage": advantage.mean(), "char_advantage_se": advantage.std(ddof=1) / np.sqrt(DRAWS),
        "char_win_rate": float((advantage > 0).mean()),
        "per_draw": {k: v["log_loss"] for k, v in rows.items()},
        "documents": N_DOCS, "draws": DRAWS,
    })
# notebook-end


SPEC = {
    "chapter": 49,
    "chapter_title": "Text Feature Engineering",
    "subtitle": "Check term overlap first, keep a sparse baseline, and compare word, character and combined representations on the same rows.",
    "summary": ("A vocabulary fitted on clean training text meets typos in the assessment text. One demonstration measures "
                "the frequency-weighted term coverage and compares word, character and combined TF-IDF models as the typo rate rises."),
    "title": "Word, character and combined TF-IDF when only the assessment text has typos",
    "question": "At what typo rate, and at what term coverage, does a character TF-IDF model overtake a word model?",
    "why": ("Step Zero of the chapter is to measure coverage before choosing a representation, with the warning that no "
            "universal overlap threshold decides it. This demonstration finds where the switch falls in one controlled "
            "setting and shows that it depends on the metric."),
    "method": ("Constructed product-style titles of 7 invented words: four classes, each with 40 class-specific words (30% of "
               "the words in its titles) and a shared Zipf-weighted background of 200 words. Models are trained on 800 clean titles and "
               "scored on 800 more in which each word is misspelled (substitution, deletion or swap) with the control's probability. "
               "Three logistic regressions use TF-IDF with sublinear tf, fitted on training text only: words, characters (analyzer "
               "'char', n-grams 3 to 5, the chapter's baseline) and the two matrices side by side. Coverage is the share of assessment "
               "tokens, repeats included, that appear in the training text. All numbers are means over 8 corpora."),
    "control": {"key": "typo_rate", "label": "Share of assessment words that are misspelled",
                "values": [0, 0.1, 0.2, 0.4], "default": 0.2,
                "value_labels": ["0: clean text", "0.1", "0.2", "0.4: heavy noise"]},
    "source_section": "A Small Sparse Comparator",
    "symbols": ("Coverage is the frequency-weighted share of assessment tokens found in the training vocabulary. Log loss is "
                "the mean negative log probability of the true class (lower is better), accuracy the share of correct top predictions."),
    "explanation": ("A word model can only use words it saw in training, so every misspelled word becomes a blank. Character n-grams "
                    "keep matching the unchanged fragments, but spanning word boundaries they also add noisy features, which is "
                    "why they lose on clean text. Where the two lines cross is a property of the data and the metric, not a coverage "
                    "constant. Placing both representations side by side gives the lowest log loss up to moderate noise here, "
                    "but under heavy typos it only ties the character model, and with any typos it is less accurate than "
                    "characters alone, so the scored metric decides."),
    "application": ("Measure frequency-weighted coverage of the assessment text first, then compare word, character and combined sparse "
                    "models on identical rows with the scored metric, before spending time on embeddings."),
    "assumptions": ("Constructed text from invented words, with typos in the assessment text only; real noise has structure (keyboard "
                    "neighbours, language), and training text with the same typos would favour the word model. The character "
                    "analyzer spans word boundaries as in the chapter's baseline; a word-bounded analyzer would lose less on clean "
                    "text. Models are regularised logistic regressions with C = 10."),
    "prediction": "With 20% of assessment words misspelled (about 81% coverage), which model has the lower log loss?",
    "prediction_options": ["The word model, clearly", "The character model, clearly", "About the same: this is near the switch"],
    "prediction_answer": 2,
    "prediction_feedback": {
        "correct": "Word log loss is 0.434 and character 0.425, a small difference near the switch; both lose clearly to the combined model at 0.389.",
        "incorrect": "Word log loss is 0.434 and character 0.425, a small difference near the switch, and both lose clearly to the combined model at 0.389.",
    },
    "check": "On clean text the word model wins clearly. Why is that not a reason to drop character features, and what does a coverage rule miss?",
    "answer": ("Which model wins changes with the noise: the word model leads by 0.051 log loss on clean text, but at 40% typos the character "
               "model leads by 0.083 and is right 0.887 of the time against 0.811. At 10% typos the models already disagree by metric, with "
               "the word model ahead on log loss (0.389 against 0.413) and the character model ahead on accuracy (0.896 against 0.883). "
               "A coverage cutoff would not reproduce that: coverage is 0.905 at 10% typos and the right choice depends on the scored metric."),
    "provenance": "Constructed example: seeded synthetic titles from invented words and sparse logistic regressions, measured by the chapter activity.",
    "apply": [
        "Compute frequency-weighted coverage of the assessment text against the training vocabulary before choosing a representation.",
        "Compare word, character and combined TF-IDF with a regularised linear model on identical rows, fitted on training text only.",
        "Judge the comparison by the scored metric: log loss and accuracy can pick different models near the switch.",
        "Keep the sparse baseline when adding embeddings, and re-check the comparison on text noised the way the test set is.",
    ],
    "honesty": "Constructed text and typos injected only at assessment time; the sizes of these effects are properties of this generator, not a competition result.",
}

EQUATIONS = [{"tex": r"\mathrm{coverage}=\frac{\#\{\text{assessment tokens found in the training vocabulary}\}}{\#\{\text{assessment tokens}\}}",
              "alt": "coverage equals the count of assessment tokens found in the training vocabulary divided by the count of assessment tokens",
              "basis": "The frequency-weighted overlap of Step Zero, which the manuscript describes in prose; this is the activity's own definition."}]
NCOLS = 2
HEIGHT = 4.4
MODELS = [("word", "Word"), ("char", "Character\n3-5"), ("both", "Word +\ncharacter")]


def draw(axes, result, parameter):
    ax, bx = axes
    colors = [COLORS["navy"], COLORS["terracotta"], COLORS["teal"]]
    ax.bar(range(3), [result[k]["log_loss"] for k, _ in MODELS], width=0.55, color=colors, edgecolor=COLORS["ink"], lw=0.6)
    for x, (k, _) in enumerate(MODELS):
        dots = result["per_draw"][k]
        ax.scatter([x + (i - (len(dots) - 1) / 2) * 0.06 for i in range(len(dots))], dots, s=14, color="white", edgecolor=COLORS["ink"],
                   linewidth=0.8, zorder=3, label="One corpus" if x == 0 else None)
        ax.text(x, max(result[k]["log_loss"], max(dots)) + 0.012, fmt(result[k]["log_loss"]), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(3), [n for _, n in MODELS])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Log loss on assessment titles")
    ax.set_xlabel(f"Model; test-token coverage {fmt(result['coverage'])}")
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    for x, ((k, _), c) in enumerate(zip(MODELS, colors)):
        v = result[k]["accuracy"]
        bx.vlines(x, 0.7, v, color=c, lw=3)
        bx.scatter([x], [v], s=120, color=c, zorder=3, edgecolor="white", linewidth=1)
        bx.text(x + 0.12, v, fmt(v), va="center", fontsize=10)
    bx.set_xticks(range(3), [n for _, n in MODELS])
    bx.set_xlim(-0.5, 2.6)
    bx.set_ylim(0.7, 1.0)
    bx.set_ylabel("Accuracy (axis starts at 0.7)")
    bx.set_xlabel(f"Model, {round(100 * parameter)}% of words misspelled")


def explain(result, parameter):
    w, c, b = result["word"], result["char"], result["both"]
    adv, se = result["char_advantage"], result["char_advantage_se"]
    if adv > 3 * se:
        verdict = "The character model is clearly better on log loss."
    elif adv < -3 * se:
        verdict = "The word model is clearly better on log loss."
    elif abs(adv) > 2 * se:
        verdict = (f"The {'character' if adv > 0 else 'word'} model is ahead on log loss by a small margin, between two and "
                   "three standard errors: this is near the switch.")
    else:
        verdict = "The two are within two standard errors of each other on log loss."
    shown = round(w["log_loss"], 3) - round(c["log_loss"], 3)      # difference of the numbers as displayed, so the sum checks by hand
    hand = f"{fmt(w['log_loss'])} - {fmt(c['log_loss'])} = {fmt(shown)}" if shown >= 0 else f"{fmt(c['log_loss'])} - {fmt(w['log_loss'])} = {fmt(-shown)} in favour of words"
    interpretation = (
        f"With {round(100 * parameter)}% of assessment words misspelled, {fmt(result['coverage'])} of assessment tokens are in the training vocabulary. "
        f"Word log loss is {fmt(w['log_loss'])} and character log loss {fmt(c['log_loss'])}: {hand} (standard error {fmt(se)}). {verdict} "
        f"Accuracy is {fmt(w['accuracy'])} for words and {fmt(c['accuracy'])} for characters; the combined model has log loss "
        f"{fmt(b['log_loss'])} and accuracy {fmt(b['accuracy'])}.")
    steps = [f"Coverage of assessment tokens: {fmt(result['coverage'])}.",
             f"Log loss, word minus character: {fmt(w['log_loss'])} - {fmt(c['log_loss'])} = {signed(shown)} (positive means characters are better).",
             f"Accuracy, character minus word: {fmt(c['accuracy'])} - {fmt(w['accuracy'])} = {signed(c['accuracy'] - w['accuracy'])}.",
             f"Combined model against the better single model: {fmt(b['log_loss'])} - {fmt(min(w['log_loss'], c['log_loss']))} = {signed(b['log_loss'] - min(w['log_loss'], c['log_loss']))} log loss."]
    metrics = {"Token coverage": fmt(result["coverage"]), "Word log loss": fmt(w["log_loss"]), "Character log loss": fmt(c["log_loss"]),
               "Combined log loss": fmt(b["log_loss"]), "Word accuracy": fmt(w["accuracy"]), "Character accuracy": fmt(c["accuracy"])}
    alt = (f"Two panels at a {round(100 * parameter)}% typo rate. Left, log loss of word, character and combined TF-IDF models "
           f"({fmt(w['log_loss'])}, {fmt(c['log_loss'])}, {fmt(b['log_loss'])}) with a dot for each of {result['draws']} corpora. "
           f"Right, the accuracy of the same three models.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    zero, tenth, fifth, forty = results[0], results[0.1], results[0.2], results[0.4]
    cov = [results[v]["coverage"] for v in (0, 0.1, 0.2, 0.4)]
    assert cov == sorted(cov, reverse=True) and cov[0] > 0.99, "coverage falls as typos rise"
    assert zero["char_advantage"] < -0.03 and zero["char_win_rate"] == 0.0, "on clean text the word model leads on log loss"
    assert tenth["char_advantage"] < -0.01, "at 10% typos the word model still leads on log loss"
    assert tenth["char"]["accuracy"] > tenth["word"]["accuracy"], "at 10% typos the character model leads on accuracy"
    assert abs(fifth["char_advantage"]) < 3 * fifth["char_advantage_se"] and abs(fifth["char_advantage"]) < 0.02, "near the switch at 20%"
    assert forty["char_advantage"] > 0.05 and forty["char_win_rate"] == 1.0, "heavy noise: characters are clearly better"
    assert forty["char"]["accuracy"] - forty["word"]["accuracy"] > 0.05
    for v in (0, 0.1, 0.2):
        res = results[v]
        assert res["both"]["log_loss"] < min(res["word"]["log_loss"], res["char"]["log_loss"]), f"combined has the lowest log loss at {v}"
    assert abs(forty["both"]["log_loss"] - forty["char"]["log_loss"]) < 0.01, "heavy typos: combined only ties the character model"
    for v in (0.1, 0.2, 0.4):
        assert results[v]["both"]["accuracy"] < results[v]["char"]["accuracy"], f"with typos combined is less accurate than characters at {v}"
    assert fifth["both"]["log_loss"] < min(fifth["word"]["log_loss"], fifth["char"]["log_loss"]) - 0.02
    assert fmt(fifth["word"]["log_loss"]) == "0.434" and fmt(fifth["char"]["log_loss"]) == "0.425" and fmt(fifth["both"]["log_loss"]) == "0.389"
    assert fmt(-zero["char_advantage"]) == "0.051" and fmt(forty["char_advantage"]) == "0.083"
    assert fmt(forty["char"]["accuracy"]) == "0.887" and fmt(forty["word"]["accuracy"]) == "0.811"
    assert fmt(tenth["word"]["log_loss"]) == "0.389" and fmt(tenth["char"]["log_loss"]) == "0.413"
    assert fmt(tenth["char"]["accuracy"]) == "0.896" and fmt(tenth["word"]["accuracy"]) == "0.883" and fmt(tenth["coverage"]) == "0.905"
