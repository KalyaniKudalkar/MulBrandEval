"""
Join Pass 1 primary-rater scores to Config A, Config B, and Config C
pipeline output for the same 120 images.

Config A's clip_score is min-max normalized to 0-1 using the min/max from
the FULL 900-image multilingual pool, not just the 120-image sample, so
the normalization doesn't shift depending on which images were sampled
for human rating.

Run from project root:
    python -m phase3a_human_eval.phase3a_join_primary_rater
"""
import pandas as pd

TRACKER_PATH = "results/phase3a/phase3a_primary_rater_ratings.xlsx"
MAPPING_PASS1_PATH = "results/phase3a/phase3a_120_secret_mapping.csv"
CONFIG_A_FULL_PATH = "results/phase2b/configuration_a_clip_baseline_multilingual.csv"
CONFIG_B_PATH = "results/phase2b/multilingual_ablation_table.csv"
CONFIG_C_PATH = "results/phase3a/phase3a_120_sample.csv"
OUTPUT_PATH = "results/phase3a/phase3a_120_primary_rater_joined.csv"

KEY_COLS = ["brief_id", "model", "language"]

COLUMN_RENAME = {
    "Rating ID": "rating_id",
    "Overall (1-5)": "overall",
    "Objects (1-5)": "objects",
    "Spatial (1-5)": "spatial",
    "Colour (1-5)": "colour",
    "Typography (1-5)": "typography",
}


def load_pass1_scores(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Pass 1 - 120 Images")
    df = df.rename(columns=COLUMN_RENAME)
    df["pass"] = 1
    return df[["rating_id", "pass", "overall", "objects", "spatial", "colour", "typography"]]


def load_config_a_normalized(path: str) -> pd.DataFrame:
    """
    Loads the full 900-image Config A output and adds a min-max normalized
    0-1 column, scaled using the full 900-image pool's min/max.
    """
    df = pd.read_csv(path)
    lo, hi = df["clip_score"].min(), df["clip_score"].max()
    df["clip_score_norm"] = (df["clip_score"] - lo) / (hi - lo)
    print(f"Config A normalization: min={lo:.4f}, max={hi:.4f} (across all {len(df)} images)")
    return df[KEY_COLS + ["clip_score", "clip_score_norm"]]


def main():
    pass1 = load_pass1_scores(TRACKER_PATH)
    mapping = pd.read_csv(MAPPING_PASS1_PATH)

    base = pass1.merge(mapping, on="rating_id", how="left")
    assert base["brief_id"].isna().sum() == 0, "Unmatched rating_id -> brief_id mapping"

    config_a = load_config_a_normalized(CONFIG_A_FULL_PATH)
    config_c = pd.read_csv(CONFIG_C_PATH)
    config_b_full = pd.read_csv(CONFIG_B_PATH)
    config_b = config_b_full.merge(base[KEY_COLS], on=KEY_COLS, how="inner")

    merged = base.merge(config_c, on=KEY_COLS, how="left", suffixes=("", "_configC"))
    merged = merged.merge(
        config_b, on=KEY_COLS, how="left", suffixes=("_configC", "_configB")
    )
    merged = merged.merge(config_a, on=KEY_COLS, how="left")
    merged = merged.rename(columns={
        "clip_score": "clip_score_configA",
        "clip_score_norm": "clip_score_norm_configA",
    })

    assert merged["compliance_score_configC"].isna().sum() == 0, "Missing Config C match"
    assert merged["compliance_score_configB"].isna().sum() == 0, "Missing Config B match"
    assert merged["clip_score_configA"].isna().sum() == 0, "Missing Config A match — check the 120 sample only used DE/FR/ES/sd15/flux"

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Joined {len(merged)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()