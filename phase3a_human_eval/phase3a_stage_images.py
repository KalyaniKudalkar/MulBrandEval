"""
MulBrandEval — Phase 3A Image Staging
=======================================
Copies the 30 inter-rater images + 1 attention-check image from their
nested per-model/per-language folders into one flat folder, ready to
upload to Google Drive as a batch.

Run from project root (after phase3a_sampling.py has produced the CSVs):
    python phase3a_stage_images.py
"""

import shutil
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results/phase3a")
STAGING_DIR = Path("data/phase3a_interrater_staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)


def stage_file(row):
    src = Path(row["output_path"].replace("\\", "/"))  # normalize Windows path
    if not src.exists():
        print(f"MISSING: {src}")
        return None
    # Collision-safe name: brief_id + model + language are unique together,
    # but the same brief_id/seed filename recurs across model/language folders.
    dest_name = f"{row['brief_id']}_{row['model']}_{row['language']}.png"
    dest = STAGING_DIR / dest_name
    shutil.copy2(src, dest)
    return dest_name


def main():
    interrater = pd.read_csv(RESULTS_DIR / "phase3a_interrater_30.csv")
    attention = pd.read_csv(RESULTS_DIR / "phase3a_attention_check.csv")

    all_rows = pd.concat([interrater, attention], ignore_index=True)

    copied, missing = [], []
    for _, row in all_rows.iterrows():
        result = stage_file(row)
        (copied if result else missing).append(row.get("brief_id", "?"))

    print(f"\nCopied {len(copied)} of {len(all_rows)} files to {STAGING_DIR}/")
    if missing:
        print(f"MISSING ({len(missing)}): {missing}")
        print("Check that output_path values match your actual folder structure.")
    else:
        print("All files found and copied successfully.")
    print(f"\nNext: upload the entire '{STAGING_DIR}' folder to Google Drive.")


if __name__ == "__main__":
    main()