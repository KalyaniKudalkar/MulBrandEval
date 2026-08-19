"""
verify_phase2b_images.py

Ground-truth verification for Phase 2B image generation, adapted from
verify_phase2a_images.py for the multilingual (DE/FR/ES) case.

Checks two independent things:
  1. Filesystem — exact file count per (model, language) folder, confirming
     150/150 briefs present for each of the 6 (model, language) combinations
     (900 total). Missing brief_ids are listed explicitly, not just counted.
  2. generation_log.csv — distinguishes STALE FAIL rows (a later SUCCESS
     exists for the same model/language/prompt_id/seed — e.g. the NSFW
     retries on sd15/de/MB006 and sd15/es/MB077) from TRULY UNRESOLVED FAIL
     rows (no matching SUCCESS anywhere — a real, current problem).

Trusts the filesystem as ground truth, not the terminal's running summary
line, which accumulates stale retry counts across every run of
phase2b_generate.py (same append-mode logging behaviour documented for
Phase 2A).

Run from MulBrandEval/ root (same conda env as generation):
    python verify_phase2b_images.py
"""

import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

BRIEFS_FILES = {
    "de": "data/prompts/mbb/mbb_briefs_de.csv",
    "fr": "data/prompts/mbb/mbb_briefs_fr.csv",
    "es": "data/prompts/mbb/mbb_briefs_es.csv",
}
LANGUAGES  = ["de", "fr", "es"]   # Arabic intentionally excluded from full run
MODELS     = ["sd15", "flux"]
IMAGE_ROOT = Path("data/generated_images/phase2b")
LOG_FILE   = Path("results/phase2b/generation_log.csv")


def check_filesystem():
    print("=" * 70)
    print("FILESYSTEM CHECK — image files on disk")
    print("=" * 70)

    total_expected = 0
    total_found = 0
    any_missing = False

    for lang in LANGUAGES:
        briefs = pd.read_csv(BRIEFS_FILES[lang])
        assert len(briefs) == 150, (
            f"Expected 150 briefs, found {len(briefs)} in {BRIEFS_FILES[lang]}"
        )
        brief_ids = briefs["brief_id"].tolist()

        for model in MODELS:
            folder = IMAGE_ROOT / model / lang
            expected = set(f"{bid}_seed42.png" for bid in brief_ids)
            found = set(f.name for f in folder.glob("*.png")) if folder.exists() else set()

            missing = sorted(expected - found)
            extra   = sorted(found - expected)

            total_expected += len(expected)
            total_found    += len(found & expected)

            status = "OK" if not missing else "MISSING"
            print(f"  {model:5s} | {lang} | {len(found & expected):3d}/150  [{status}]")
            if missing:
                any_missing = True
                print(f"           missing: {[m.replace('_seed42.png', '') for m in missing]}")
            if extra:
                print(f"           unexpected extra files (not in brief list): {extra}")

    print("-" * 70)
    print(f"  TOTAL: {total_found}/{total_expected}")
    print("=" * 70)
    return not any_missing and total_found == total_expected


def check_log():
    print("\n" + "=" * 70)
    print("GENERATION LOG CHECK — stale vs. real failures")
    print("=" * 70)

    if not LOG_FILE.exists():
        print(f"  {LOG_FILE} not found — skipping log cross-check.")
        return

    log_df = pd.read_csv(LOG_FILE)
    # Restrict to DE/FR/ES rows only — AR pilot rows are out of scope here.
    log_df = log_df[log_df["language"].isin(LANGUAGES)]

    key_cols = ["model", "language", "prompt_id", "seed"]

    has_success = (
        log_df[log_df["status"] == "SUCCESS"]
        .drop_duplicates(subset=key_cols)
    )
    fail_only = (
        log_df[log_df["status"] == "FAIL"]
        .merge(has_success[key_cols], on=key_cols, how="left", indicator=True)
    )
    truly_unresolved = fail_only[fail_only["_merge"] == "left_only"]

    print(f"  Total FAIL rows (DE/FR/ES) in log: {len(fail_only)}")
    print(f"  FAIL rows with a later SUCCESS (stale, harmless): "
          f"{len(fail_only) - len(truly_unresolved)}")
    print(f"  FAIL rows with NO matching SUCCESS anywhere (real problem): "
          f"{len(truly_unresolved)}")
    if len(truly_unresolved):
        print("\n  ** REAL, UNRESOLVED FAILURES — investigate before evaluation: **")
        print(truly_unresolved[key_cols + ["error"]].to_string(index=False))
    print("=" * 70)

    return len(truly_unresolved) == 0


def main():
    fs_ok = check_filesystem()
    log_ok = check_log()

    print("\n" + "=" * 70)
    if fs_ok and log_ok:
        print("VERDICT: 900/900 images present, no unresolved failures. "
              "Safe to proceed to phase2b_evaluate.py.")
    else:
        print("VERDICT: issues found above — resolve before running "
              "phase2b_evaluate.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()