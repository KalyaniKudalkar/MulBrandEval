"""
phase3a_human_eval/phase3a_intrarater_reliability.py

Compares the primary rater's Pass 1 scores to their Pass 2 repeat scores
on the same 20 images (rated again after a 48-hour gap), joined via
brief_id/model/language rather than the anonymised rating_id or brief
text — text alone is ambiguous since the same English brief was used to
generate images in multiple languages.

Reports mean absolute difference per dimension and flags any pair
differing by more than MAX_DIFF_FLAG points.

Run from project root:
    python -m phase3a_human_eval.phase3a_intrarater_reliability
"""
import pandas as pd

TRACKER_PATH = "results/phase3a/phase3a_primary_rater_ratings.xlsx"
MAPPING_PASS1_PATH = "results/phase3a/phase3a_120_secret_mapping.csv"
MAPPING_PASS2_PATH = "results/phase3a/phase3a_20_secret_mapping.csv"
OUTPUT_PATH = "results/phase3a/phase3a_intrarater_reliability_results.csv"

MAX_DIFF_FLAG = 2  # flag pairs differing by more than this many points

COLUMN_RENAME = {
    "Rating ID": "rating_id",
    "Overall (1-5)": "overall",
    "Objects (1-5)": "objects",
    "Spatial (1-5)": "spatial",
    "Colour (1-5)": "colour",
    "Typography (1-5)": "typography",
}
DIMENSIONS = ["overall", "objects", "spatial", "colour", "typography"]
KEY_COLS = ["brief_id", "model", "language"]


def load_sheet(path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df.rename(columns=COLUMN_RENAME)[["rating_id"] + DIMENSIONS]


def main():
    pass1_scores = load_sheet(TRACKER_PATH, "Pass 1 - 120 Images")
    pass2_scores = load_sheet(TRACKER_PATH, "Pass 2 - Intrarater 20")

    map1 = pd.read_csv(MAPPING_PASS1_PATH)
    map2 = pd.read_csv(MAPPING_PASS2_PATH)

    pass1 = pass1_scores.merge(map1, on="rating_id", how="left")
    pass2 = pass2_scores.merge(map2, on="rating_id", how="left")

    merged = pass2.merge(
        pass1, on=KEY_COLS, how="left", suffixes=("_pass2", "_pass1")
    )
    assert merged["rating_id_pass1"].isna().sum() == 0, "Unmatched Pass 2 -> Pass 1 join"
    assert len(merged) == 20

    for dim in DIMENSIONS:
        merged[f"{dim}_diff"] = (merged[f"{dim}_pass2"] - merged[f"{dim}_pass1"]).abs()

    summary_rows = []
    for dim in DIMENSIONS:
        diffs = merged[f"{dim}_diff"]
        summary_rows.append({
            "dimension": dim,
            "mean_abs_diff": diffs.mean(),
            "max_abs_diff": diffs.max(),
            "n_pairs": len(diffs),
            "n_flagged": int((diffs > MAX_DIFF_FLAG).sum()),
        })
    summary = pd.DataFrame(summary_rows)

    print(summary.to_string(index=False))

    flagged = merged[(merged[[f"{d}_diff" for d in DIMENSIONS]] > MAX_DIFF_FLAG).any(axis=1)]
    if len(flagged) > 0:
        print(f"\n{len(flagged)} pairs with a difference > {MAX_DIFF_FLAG} on any dimension:")
        cols = ["rating_id_pass2", "rating_id_pass1"] + [f"{d}_diff" for d in DIMENSIONS]
        print(flagged[cols].to_string(index=False))
    else:
        print(f"\nNo pairs exceeded {MAX_DIFF_FLAG}-point difference on any dimension.")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()