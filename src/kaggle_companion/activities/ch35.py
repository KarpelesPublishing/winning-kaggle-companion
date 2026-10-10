"""Chapter 35: Memory Optimization for Large Datasets. Blanket float32 against a checked downcast, by how large the IDs are."""
from kaggle_companion.activities._common import COLORS, fmt, signed

# notebook-begin
import tracemalloc

import numpy as np
import pandas as pd

from kaggle_companion.activities._common import clean

SEED = 35
ROWS, CUSTOMERS = 200_000, 40_000
REGIONS = [f"region_{i:02d}_north_america_east" for i in range(12)]   # a long, repeated string column
INT32_MAX = np.iinfo(np.int32).max


def generate(first_id, rng):
    """Purchases: a customer ID that starts at `first_id`, an amount whose mean depends on the customer, and a region string."""
    customer = rng.integers(0, CUSTOMERS, ROWS)
    taste = rng.normal(0, 1, CUSTOMERS)
    amount = np.round(60 * np.exp(0.6 * taste[customer] + rng.normal(0, 0.3, ROWS)), 2)
    return pd.DataFrame({"customer_id": (first_id + customer).astype(np.int64), "amount": amount,
                         "region": np.array(REGIONS)[rng.integers(0, 12, ROWS)]})


def blanket_float32(df):
    """The shortcut: every numeric column becomes float32 because the range fits."""
    out = df.copy()
    for name in ("customer_id", "amount"):
        out[name] = df[name].astype(np.float32)
    return out


def checked_downcast(df):
    """The chapter's information check: float32 only if values and distinct counts survive; integers shrink only if the range fits."""
    out = df.copy()
    for name in ("customer_id", "amount"):
        values = df[name].to_numpy()
        if values.dtype.kind == "i":
            if values.min() >= np.iinfo(np.int32).min and values.max() <= INT32_MAX:
                out[name] = values.astype(np.int32)
        else:
            candidate = values.astype(np.float32)
            same_values = np.allclose(values, candidate.astype(np.float64), atol=1e-7, rtol=1e-5)
            if same_values and np.unique(values).size == np.unique(candidate).size:
                out[name] = candidate
    out["region"] = df["region"].astype("category")        # repeated strings stored once
    return out


def customer_feature(df, train, fallback):
    """The feature a competitor would build: mean amount per customer ID on training rows, mapped onto held-out rows."""
    per_id = df[train].groupby("customer_id")["amount"].mean()
    return df.loc[~train, "customer_id"].map(per_id).fillna(fallback).to_numpy(dtype=float)


def run(first_id_power):
    rng = np.random.default_rng(SEED)
    df = generate(2 ** first_id_power, rng)
    train = np.arange(ROWS) % 2 == 0                        # even rows build the feature, odd rows are scored against it
    truth = df.loc[~train, "amount"].to_numpy()
    original_mb = df.memory_usage(deep=True).sum() / 1e6
    schemes = {}
    for name, convert in (("none", lambda d: d.copy()), ("blanket", blanket_float32), ("checked", checked_downcast)):
        tracemalloc.start()                                 # measure what the conversion itself allocates at its peak
        frame = convert(df)
        conversion_peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        feature = customer_feature(frame, train, df.loc[train, "amount"].mean())
        columns = frame.memory_usage(deep=True)
        schemes[name] = {
            "megabytes": columns.sum() / 1e6,
            "by_column": {"Customer ID": columns["customer_id"] / 1e6, "Amount": columns["amount"] / 1e6,
                          "Region": columns["region"] / 1e6},
            # measured peak while converting: the original table stays alive while the conversion allocates
            "conversion_peak": round(0.0 if name == "none" else conversion_peak_mb, 1),
            "while_converting": round(original_mb + (0.0 if name == "none" else conversion_peak_mb), 1),
            "distinct_ids": int(frame["customer_id"].nunique()),
            "feature_r2": float(np.corrcoef(feature, truth)[0, 1] ** 2),
            "max_amount_error_cents": float(100 * np.abs(frame["amount"].astype(np.float64) - df["amount"]).max()),
            "id_dtype": str(frame["customer_id"].dtype),
        }
    return clean({"first_id_power": first_id_power, "first_id": 2 ** first_id_power, "true_ids": int(df["customer_id"].nunique()),
                  "schemes": schemes, "rows": ROWS})
# notebook-end


SPEC = {
    "chapter": 35,
    "chapter_title": "Memory Optimization for Large Datasets",
    "subtitle": "Range is not precision: shrink a table only after checking what the smaller type destroys.",
    "summary": ("A blanket float32 downcast looks like free memory. One demonstration builds a purchase table, shrinks it two ways "
                "and measures both the megabytes saved and what survives: how many customer IDs stay distinct and how well a "
                "per-customer feature still predicts held-out rows."),
    "title": "Blanket float32 against a checked downcast, by how large the IDs are",
    "question": "At what ID size does a blanket float32 downcast start merging customers, and how much memory does it save compared with a checked downcast?",
    "why": ("Memory work that silently changes the data is worse than a crash: the run finishes, the score drops, and nothing "
            "points at the dtype. The chapter's check compares values and distinct counts before accepting a smaller type."),
    "method": ("A constructed table of 200,000 purchases from 40,000 customers: an integer customer ID that starts at the control "
               "value, an amount in dollars and cents, and a long region string. Three versions are compared: unchanged, a blanket "
               "float32 for the ID and amount columns, and a checked downcast (float32 only when the chapter's value and "
               "distinct-count check passes, int32 only when the range fits, repeated strings as categories). Each is scored by "
               "its memory, by how many distinct IDs remain, and by R-squared of a held-out-rows prediction from the mean amount "
               "per customer ID."),
    "control": {"key": "first_id_power", "label": "Smallest customer ID, as a power of two",
                "values": [22, 24, 26, 33], "default": 24,
                "value_labels": ["2^22 (4.2 million)", "2^24 (16.8 million)", "2^26 (67 million)", "2^33 (8.6 billion)"]},
    "source_section": "Downcast With an Information Check",
    "symbols": ("n is the number of values in a column, b the bits stored per value, x_i an original value and x-hat_i the value "
                "stored after conversion; e_max is the largest conversion error."),
    "explanation": ("Float32 keeps about seven significant digits. Below 2^24 every integer is exact; between 2^24 and 2^25 only "
                    "even integers survive, so neighboring IDs merge in pairs, and each further doubling merges twice as many. "
                    "Merged IDs make a per-ID aggregate describe the wrong customers. The same float32 is harmless for dollar "
                    "amounts, whose error stays far below a cent. A checked downcast tests each column for what it would destroy."),
    "application": ("Before accepting any smaller dtype, compare distinct counts and task-unit errors, keep identifiers in an "
                    "integer type wide enough for their range, and measure the table's memory before and after rather than assuming."),
    "assumptions": ("Constructed data on this machine's pandas. Memory is pandas deep usage of the resident table, so it depends on "
                    "the pandas version and its string storage; the converted copy and the original are both alive during conversion, "
                    "which is why conversion itself needs more memory than either table (its peak is measured with Python's tracemalloc "
                    "allocation tracer, which sees pandas and NumPy buffers but not allocator overhead). The checked downcast adds an int32 range "
                    "check and category strings, which are ordinary pandas practice rather than steps the chapter's helper contains."),
    "prediction": ("IDs start at 2^24 (16.8 million) and span 39,756 customers. After a blanket float32 downcast, how many distinct "
                   "customers remain?"),
    "prediction_options": ["All of them: float32 holds numbers this large", "About half of them", "Almost none of them"],
    "prediction_answer": 1,
    "prediction_feedback": {
        "correct": ("19,933 of 39,756 remain. Above 2^24 float32 stores only even integers, so neighbors merge in pairs, and the "
                    "per-customer feature loses most of its power (R-squared 0.600 falls to 0.271)."),
        "incorrect": ("19,933 of 39,756 remain. Float32 holds the range but not the precision: above 2^24 it stores only even "
                      "integers, so neighbors merge in pairs, and the per-customer feature loses most of its power (R-squared 0.600 "
                      "falls to 0.271)."),
    },
    "check": "Why does the checked downcast save about 90% of the memory when the blanket float32 saves less than 10%, and why does it keep 64-bit IDs at 2^33?",
    "answer": ("In this table the long region strings hold most of the bytes (15.4 of 18.6 MB), so shrinking two numeric columns "
               "barely matters (17.0 MB) while storing the strings once as categories reaches 1.8 MB. The numeric columns still "
               "matter for correctness: IDs at 2^33 do not fit in int32 and merge in float32, so the check leaves them 64-bit and "
               "the table is 2.6 MB. A check that sometimes says no is the point."),
    "provenance": "Constructed example: a seeded synthetic purchase table; the memory figures are measured by the chapter activity.",
    "apply": [
        "Compare distinct counts and task-unit error before and after every downcast; never accept a type because the range fits.",
        "Keep IDs, timestamps and money differences in an integer or float64 type unless the check passes.",
        "Find where the bytes are first (here, strings) and measure the table: the numeric downcast is often the smaller saving.",
        "Remember the conversion copy: the original and the new column coexist, so convert in chunks when memory is already tight.",
    ],
    "honesty": "Constructed data on this machine's pandas; the savings are properties of this table, not a general rule.",
}

EQUATIONS = [{"tex": r"\mathrm{bytes} = \frac{n\, b}{8}, \qquad e_{\max} = \max_i \left| x_i - \hat{x}_i \right|",
              "alt": "Array bytes equal the number of values n times the bits per value b divided by eight, and the maximum conversion error is the largest absolute gap between each original value x i and its stored value x hat i",
              "basis": "The chapter's byte estimate (stated in prose in Chapter 35, Estimate the Largest Operation) and the activity's own conversion-error metric; Chapter 35 has no display equation."}]
NCOLS = 2
HEIGHT = 4.4
SCHEMES = [("none", "Unchanged"), ("blanket", "Blanket float32"), ("checked", "Checked downcast")]
PARTS = [("Customer ID", COLORS["navy"]), ("Amount", COLORS["teal"]), ("Region", COLORS["light"])]


def draw(axes, result, parameter):
    left, right = axes
    xs = list(range(len(SCHEMES)))
    bottoms = [0.0] * len(SCHEMES)
    for part, color in PARTS:
        heights = [result["schemes"][k]["by_column"][part] for k, _ in SCHEMES]
        left.bar(xs, heights, bottom=bottoms, color=color, edgecolor=COLORS["ink"], lw=0.6, label=part, width=0.6)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    for x, total in zip(xs, bottoms):
        left.text(x, total + 0.4, f"{total:.1f} MB", ha="center", va="bottom", fontsize=10)
    left.set_xticks(xs, [name for _, name in SCHEMES])
    left.set_ylim(0, max(bottoms) * 1.25)
    left.set_ylabel("Table memory (MB)")
    left.set_xlabel("Downcast scheme")
    left.legend(loc="upper right", frameon=False, fontsize=10)

    r2 = [result["schemes"][k]["feature_r2"] for k, _ in SCHEMES]
    colors = [COLORS["grey"], COLORS["terracotta"], COLORS["teal"]]
    right.bar(xs, r2, color=colors, edgecolor=COLORS["ink"], lw=0.6, width=0.6)
    for x, (k, _), v in zip(xs, SCHEMES, r2):
        right.text(x, v + 0.012, f"{fmt(v)}\n{result['schemes'][k]['distinct_ids']:,} IDs", ha="center", va="bottom", fontsize=10)
    right.set_xticks(xs, [name for _, name in SCHEMES])
    right.set_ylim(0, max(r2) * 1.3)
    right.set_ylabel("R-squared on held-out rows")
    right.set_xlabel(f"Downcast scheme, IDs start at 2^{parameter}")


def explain(result, parameter):
    s = result["schemes"]
    n, b, c = s["none"], s["blanket"], s["checked"]
    lost = result["true_ids"] - b["distinct_ids"]
    drop = n["feature_r2"] - b["feature_r2"]
    saved_b = n["megabytes"] - b["megabytes"]
    saved_c = n["megabytes"] - c["megabytes"]
    if lost == 0:
        harm = "No IDs merge, so the blanket downcast is safe here but still saves little."
    else:
        harm = (f"{lost:,} of {result['true_ids']:,} IDs vanish into a neighbor, and the held-out R-squared falls "
                f"from {fmt(n['feature_r2'])} to {fmt(b['feature_r2'])}.")
    interpretation = (
        f"With IDs starting at 2^{parameter}: the table is {n['megabytes']:.1f} MB unchanged, {b['megabytes']:.1f} MB with blanket "
        f"float32 ({n['megabytes']:.1f} - {b['megabytes']:.1f} = {saved_b:.1f} MB saved) and {c['megabytes']:.1f} MB checked "
        f"({n['megabytes']:.1f} - {c['megabytes']:.1f} = {saved_c:.1f} MB saved, ID stored as {c['id_dtype']}). {harm} "
        f"The checked table keeps all {c['distinct_ids']:,} IDs and R-squared {fmt(c['feature_r2'])}. Amounts change by at most "
        f"{c['max_amount_error_cents']:.3f} cents.")
    steps = [
        f"Memory saved by the blanket downcast: {n['megabytes']:.1f} - {b['megabytes']:.1f} = {saved_b:.1f} MB.",
        f"Memory saved by the checked downcast: {n['megabytes']:.1f} - {c['megabytes']:.1f} = {saved_c:.1f} MB.",
        f"Customers merged by float32: {result['true_ids']} - {b['distinct_ids']} = {lost}.",
        f"R-squared lost to the blanket downcast: {fmt(n['feature_r2'])} - {fmt(b['feature_r2'])} = {fmt(drop)}.",
        f"Measured peak while converting to the checked table: {n['megabytes']:.1f} + {c['conversion_peak']:.1f} = {c['while_converting']:.1f} MB (the original table plus the conversion's own peak allocation).",
    ]
    metrics = {"Unchanged table": f"{n['megabytes']:.1f} MB", "Blanket float32": f"{b['megabytes']:.1f} MB",
               "Checked downcast": f"{c['megabytes']:.1f} MB", "Distinct IDs after blanket": f"{b['distinct_ids']:,} of {result['true_ids']:,}",
               "R-squared, blanket": fmt(b["feature_r2"]), "R-squared, checked": fmt(c["feature_r2"])}
    alt = (f"Left: stacked memory bars for the unchanged, blanket float32 and checked tables when IDs start at 2^{parameter}; the "
           f"checked table is {c['megabytes']:.1f} MB against {n['megabytes']:.1f} MB. Right: held-out R-squared and distinct "
           f"IDs for each; blanket float32 keeps {b['distinct_ids']:,} IDs and R-squared {fmt(b['feature_r2'])}.")
    return {"metrics": metrics, "interpretation": interpretation, "steps": steps, "alt": alt}


def verify(results):
    """Every claim the fixed text makes, checked against the measured results."""
    for p, res in results.items():
        s = res["schemes"]
        assert s["none"]["distinct_ids"] == s["checked"]["distinct_ids"] == res["true_ids"], f"checked keeps every ID at {p}"
        assert abs(s["checked"]["feature_r2"] - s["none"]["feature_r2"]) < 1e-9, f"checked feature unchanged at {p}"
        assert s["checked"]["max_amount_error_cents"] < 0.01, f"float32 amounts stay far below a cent at {p}"
        assert s["checked"]["megabytes"] < 0.15 * s["none"]["megabytes"], f"checked saves about 90% at {p}"
        assert s["blanket"]["megabytes"] > 0.9 * s["none"]["megabytes"], f"blanket saves under 10% at {p}"
        assert s["checked"]["while_converting"] > s["none"]["megabytes"], "conversion needs both copies"
        assert s["checked"]["while_converting"] > s["none"]["megabytes"] + s["checked"]["megabytes"], "peak exceeds both tables"
    assert results[22]["schemes"]["blanket"]["distinct_ids"] == results[22]["true_ids"], "below 2^24 float32 is exact"
    assert results[22]["schemes"]["checked"]["id_dtype"] == "int32" and results[24]["schemes"]["checked"]["id_dtype"] == "int32"
    mid = results[24]["schemes"]
    assert mid["blanket"]["distinct_ids"] == 19933 and results[24]["true_ids"] == 39756, "prediction feedback counts"
    assert fmt(mid["none"]["feature_r2"]) == "0.600" and fmt(mid["blanket"]["feature_r2"]) == "0.271", "prediction feedback R-squared"
    assert 0.4 * 39756 < mid["blanket"]["distinct_ids"] < 0.6 * 39756, "about half"
    assert results[26]["schemes"]["blanket"]["distinct_ids"] < 0.2 * 39756, "later doubling merges more"
    assert results[33]["schemes"]["blanket"]["distinct_ids"] < 0.01 * 39756
    assert results[33]["schemes"]["checked"]["id_dtype"] == "int64", "2^33 does not fit int32"
    base = results[24]["schemes"]
    assert f"{base['none']['by_column']['Region']:.1f}" == "15.4", "region MB"
    assert f"{base['none']['megabytes']:.1f}" == "18.6" and f"{base['blanket']['megabytes']:.1f}" == "17.0"
    assert f"{base['checked']['megabytes']:.1f}" == "1.8" and f"{results[33]['schemes']['checked']['megabytes']:.1f}" == "2.6"
