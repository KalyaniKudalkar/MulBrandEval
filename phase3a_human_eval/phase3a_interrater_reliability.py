"""
phase3a_human_eval/phase3a_interrater_reliability.py

Computes Krippendorff's alpha (ordinal) across external raters on the
30-image Image Compliance inter-rater set, using the current Google Form
export. Re-runnable as more responses come in — just re-export and rerun.

Four brief IDs (MB068, MB070, MB044, MB079) were each generated under two
different model/language combinations, and both versions landed in this
30-image sample. The form's grid title only shows the brief ID, so both
grids share the same title text and the CSV export disambiguates the
second one with a ".1" suffix — these are two DIFFERENT images, not a
repeated image. Since the Apps Script generator builds grid items in the
exact row order of phase3a_interrater_30.csv, and that order matches the
CSV export's column order one-for-one, the unsuffixed column = the first
occurrence of that brief_id in phase3a_interrater_30.csv, and the ".1"
column = the second occurrence.

Run from project root:
    python -m phase3a_human_eval.phase3a_interrater_reliability
"""
import re
import pandas as pd

import numpy as np
np.float_ = np.float64  # compatibility shim: the krippendorff package still
                         # references the pre-2.0 NumPy alias, removed in NumPy 2.0
import krippendorff

INPUT_PATH = "results/phase3a/phase3a_interrater_survey_responses.csv"
ITEM_ORDER_PATH = "results/phase3a/phase3a_interrater_30.csv"
OUTPUT_PATH = "results/phase3a/phase3a_interrater_reliability_results.csv"

ATTENTION_CHECK_ID = "MB125"
ATTENTION_CHECK_PASS_MAX = 2
EXCLUDE_ATTENTION_CHECK_FAILURES = True  # recommended: True, per exposé's stated design intent

DIMENSIONS = ["Overall", "Objects", "Spatial", "Colour", "Typography"]


def parse_column(col: str):
    m = re.match(r"Ratings for (MB\d+) \[(\w+)\](\.\d+)?", col)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3) or ""


def build_item_key_map(item_order_path: str) -> dict:
    """Maps (brief_id, occurrence_index) -> 'brief_id_model_language' item key,
    using row order in phase3a_interrater_30.csv (matches form item order)."""
    order_df = pd.read_csv(item_order_path)
    occurrence_counter = {}
    key_map = {}
    for _, row in order_df.iterrows():
        bid = row["brief_id"]
        occ = occurrence_counter.get(bid, 0)
        key_map[(bid, occ)] = f"{bid}_{row['model']}_{row['language']}"
        occurrence_counter[bid] = occ + 1
    return key_map


def load_long_format(df: pd.DataFrame, key_map: dict) -> pd.DataFrame:
    rating_cols = [c for c in df.columns if c.startswith("Ratings for")]
    records = []
    for rater_idx, row in df.iterrows():
        for col in rating_cols:
            parsed = parse_column(col)
            if parsed is None:
                continue
            img_id, dim, instance = parsed
            occ = 0 if instance == "" else 1
            item_key = img_id if img_id == ATTENTION_CHECK_ID else key_map[(img_id, occ)]
            val = row[col]
            val_num = int(re.match(r"\d+", str(val)).group()) if pd.notna(val) else None
            records.append({"rater": rater_idx, "item_key": item_key, "dimension": dim, "value": val_num})
    return pd.DataFrame(records)


def attention_check_report(long_df: pd.DataFrame):
    ac = long_df[
        (long_df.item_key == ATTENTION_CHECK_ID) & (long_df.dimension == "Overall")
    ].set_index("rater")["value"]
    passed = ac[ac <= ATTENTION_CHECK_PASS_MAX].index.tolist()
    failed = ac[ac > ATTENTION_CHECK_PASS_MAX].index.tolist()
    print(f"Attention check: {len(passed)}/{len(ac)} raters passed "
          f"(Overall <= {ATTENTION_CHECK_PASS_MAX} on {ATTENTION_CHECK_ID})")
    if failed:
        print(f"Failed raters (row index in export, 0-based): {failed}")
    return passed, failed


def compute_alpha(long_df: pd.DataFrame, rater_subset, label: str) -> pd.DataFrame:
    real = long_df[long_df.item_key != ATTENTION_CHECK_ID]
    print(f"\n--- {label} (n_raters={len(rater_subset)}) ---")
    rows = []
    for dim in DIMENSIONS:
        sub = real[(real.dimension == dim) & (real.rater.isin(rater_subset))]
        pivot = sub.pivot(index="rater", columns="item_key", values="value")
        alpha = krippendorff.alpha(
            reliability_data=pivot.to_numpy(dtype=float), level_of_measurement="ordinal"
        )
        print(f"{dim}: alpha={alpha:.4f}  (raters={pivot.shape[0]}, items={pivot.shape[1]})")
        rows.append({"dimension": dim, "alpha": alpha, "n_raters": pivot.shape[0], "n_items": pivot.shape[1]})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT_PATH)
    key_map = build_item_key_map(ITEM_ORDER_PATH)
    long_df = load_long_format(df, key_map)

    n_items = long_df[long_df.item_key != ATTENTION_CHECK_ID]["item_key"].nunique()
    print(f"Distinct real items recovered: {n_items} (expect 30)")

    passed, failed = attention_check_report(long_df)

    all_raters = list(range(len(df)))
    compute_alpha(long_df, all_raters, "All raters (unfiltered)")
    result = compute_alpha(long_df, passed, "Attention-check passers only")

    if EXCLUDE_ATTENTION_CHECK_FAILURES:
        result.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved (attention-check-passers version) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()