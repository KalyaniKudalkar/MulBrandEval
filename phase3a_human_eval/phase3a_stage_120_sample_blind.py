"""
MulBrandEval — Phase 3A: Stage the 120-image primary-rater sample (BLINDED)
=============================================================================
Copies all 120 images from phase3a_120_sample.csv into one flat folder,
renamed to anonymized rating IDs (R1-001.png ... R1-120.png) so the model
(SD v1.5 vs FLUX) is NOT visible from the filename while you rate. This
addresses the bias risk Prof. Chandna flagged: knowing which model produced
an image could unconsciously influence your Likert scores.

The real brief_id/model/language mapping is saved SEPARATELY to
results/phase3a/phase3a_120_secret_mapping.csv - do NOT open this file
until after you have finished rating (both Pass 1 and Pass 2), since it
would re-reveal model identity.

Also stages the fixed 20-image intra-rater subset (R2-001.png ... R2-020.png)
using the SAME logic, with its own secret mapping file.

Run from project root:
    python phase3a_human_eval/phase3a_stage_120_sample_blind.py
"""

import shutil
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results/phase3a")
SAMPLE_120_CSV = RESULTS_DIR / "phase3a_120_sample.csv"
SAMPLE_20_CSV = RESULTS_DIR / "phase3a_intrarater_20.csv"

STAGING_DIR_120 = Path("data/phase3a_primary_rater_staging_pass1")
STAGING_DIR_20 = Path("data/phase3a_primary_rater_staging_pass2")
STAGING_DIR_120.mkdir(parents=True, exist_ok=True)
STAGING_DIR_20.mkdir(parents=True, exist_ok=True)

SHUFFLE_SEED_PASS1 = 42
SHUFFLE_SEED_PASS2 = 99


def stage_blind(csv_path, staging_dir, prefix, seed):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle order
    df["rating_id"] = [f"{prefix}-{str(i+1).zfill(3)}" for i in range(len(df))]

    copied, missing = [], []
    for _, row in df.iterrows():
        src = Path(str(row["output_path"]).replace("\\", "/"))
        dest = staging_dir / f"{row['rating_id']}.png"
        if not src.exists():
            print(f"MISSING: {src}")
            missing.append(row["rating_id"])
            continue
        shutil.copy2(src, dest)
        copied.append(row["rating_id"])

    print(f"\n{prefix}: copied {len(copied)} of {len(df)} files to {staging_dir}/")
    if missing:
        print(f"MISSING ({len(missing)}): {missing}")

    return df


def main():
    df120 = stage_blind(SAMPLE_120_CSV, STAGING_DIR_120, "R1", SHUFFLE_SEED_PASS1)
    df20 = stage_blind(SAMPLE_20_CSV, STAGING_DIR_20, "R2", SHUFFLE_SEED_PASS2)

    # Secret mapping files - keep these CLOSED while rating
    mapping120 = df120[["rating_id", "brief_id", "model", "language", "output_path"]]
    mapping20 = df20[["rating_id", "brief_id", "model", "language", "output_path"]]

    mapping120.to_csv(RESULTS_DIR / "phase3a_120_secret_mapping.csv", index=False)
    mapping20.to_csv(RESULTS_DIR / "phase3a_20_secret_mapping.csv", index=False)

    print(f"\nSecret mappings saved to {RESULTS_DIR}/ - do NOT open these until "
          f"after both rating passes are complete.")
    print("\nNext: open Phase3A_Primary_Rater_Tracker_Blind.xlsx and rate using "
          "the embedded thumbnails - no need to open the staging folders directly.")


if __name__ == "__main__":
    main()