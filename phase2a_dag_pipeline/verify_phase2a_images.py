"""
Verify Phase 2A image generation output on disk.

Run from MulBrandEval/ root:
    python verify_phase2a_images.py

Checks:
  1. Exact image count per model/language folder (expect 150 each, 300 total)
  2. Which brief_ids are present vs. expected (from mbb_briefs_en.csv)
  3. Any missing or unexpected (extra) files
  4. Duplicate/stale rows in generation_log.csv (informational only)
"""

import pandas as pd
from pathlib import Path

BRIEFS_FILE = Path("data/prompts/mbb/mbb_briefs_en.csv")
OUTPUT_ROOT = Path("data/generated_images/phase2a")
LOG_FILE    = Path("results/phase2a/generation_log.csv")
LANGUAGE    = "en"
SEED        = 42
MODELS      = ["sd15", "flux"]

def main():
    briefs = pd.read_csv(BRIEFS_FILE)
    expected_ids = set(briefs["brief_id"])
    print(f"Expected brief_ids: {len(expected_ids)}")

    grand_total = 0
    for model_key in MODELS:
        folder = OUTPUT_ROOT / model_key / LANGUAGE
        files = sorted(folder.glob(f"*_seed{SEED}.png"))
        found_ids = {f.stem.replace(f"_seed{SEED}", "") for f in files}

        missing = expected_ids - found_ids
        extra   = found_ids - expected_ids
        dupes_on_disk = len(files) - len(found_ids)  # would be >0 if case-diff etc.

        print(f"\n--- {model_key} / {LANGUAGE} ---")
        print(f"Files found      : {len(files)}")
        print(f"Unique brief_ids : {len(found_ids)}")
        print(f"Missing ({len(missing)}): {sorted(missing) if missing else 'none'}")
        print(f"Extra/unexpected ({len(extra)}): {sorted(extra) if extra else 'none'}")
        if dupes_on_disk:
            print(f"WARNING: {dupes_on_disk} filename collisions on disk")

        grand_total += len(files)

    print(f"\n=== TOTAL IMAGES ON DISK: {grand_total} (expected 300) ===")

    # --- Optional: inspect the log for stale/duplicate rows ---
    if LOG_FILE.exists():
        log_df = pd.read_csv(LOG_FILE)
        print(f"\n--- generation_log.csv sanity check ---")
        print(f"Total rows in log        : {len(log_df)}")
        print(f"SUCCESS rows             : {(log_df['status'] == 'SUCCESS').sum()}")
        print(f"FAIL rows                : {(log_df['status'] == 'FAIL').sum()}")

        # For each (model, prompt_id, seed), does it have at least one SUCCESS,
        # even if it also has earlier FAIL rows from a prior attempt?
        key_cols = ["model", "prompt_id", "seed"]
        has_success = (
            log_df[log_df["status"] == "SUCCESS"]
            .drop_duplicates(subset=key_cols)
        )
        fail_only = (
            log_df[log_df["status"] == "FAIL"]
            .merge(has_success[key_cols], on=key_cols, how="left", indicator=True)
        )
        truly_unresolved = fail_only[fail_only["_merge"] == "left_only"]
        print(f"FAIL rows with a later SUCCESS (stale, harmless): "
              f"{len(fail_only) - len(truly_unresolved)}")
        print(f"FAIL rows with NO matching SUCCESS anywhere (real problem): "
              f"{len(truly_unresolved)}")
        if len(truly_unresolved):
            print(truly_unresolved[key_cols + ["error"]].to_string(index=False))

if __name__ == "__main__":
    main()