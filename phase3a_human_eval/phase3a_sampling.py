"""
MulBrandEval — Phase 3A Sampling
==================================
Builds the three image sets needed for Phase 3A human evaluation:

  1. phase3a_120_sample.csv      - primary rater's full stratified sample
  2. phase3a_interrater_30.csv   - subset shared with 3-5 external raters
  3. phase3a_intrarater_20.csv   - subset the primary rater re-rates after 48h

DESIGN (see exposé section 4.6.1, adjusted per Arabic-skip precedent —
confirm with Prof. Chandna before treating this as final):
  - Arabic EXCLUDED from the balanced quantitative sample. Its pilot data
    (configuration_c_arabic_pilot.csv) has 0/20 images at or above the
    compliance threshold below, so no compliant/non-compliant balance is
    possible for Arabic. Reallocated to 40 images each for DE/FR/ES
    (120 total unchanged) instead of 30 x 4 languages.
  - Per language: fully balanced 2 (model) x 2 (compliant/non-compliant)
    x 10 replicates = 40 images.
  - COMPLIANCE_THRESHOLD below is NOT a previously-established cutoff —
    it's a plain-language "satisfies at least half the weighted brand
    criteria" default. Confirm this is the threshold you actually want
    before treating results as final.

File location: place this inside the phase3a_human_eval/ package folder,
alongside an __init__.py (same convention as phase2b_multilingual/).

Run from project root:
    python -m phase3a_human_eval.phase3a_sampling
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---- Configuration -------------------------------------------------------
COMPLIANCE_THRESHOLD = 0.5   # "compliant" if compliance_score >= this value
SEED = 42                     # matches the seed used throughout the project
N_PER_LANGUAGE = 40            # per language, split evenly across model x compliance
N_PER_CELL = N_PER_LANGUAGE // 4   # 10 per (model, compliance) cell
LANGUAGES = ["de", "fr", "es"]     # Arabic excluded — see docstring
MODELS = ["sd15", "flux"]

# Paths relative to project root (script is run via -m from there, so these
# resolve correctly regardless of where the .py file itself sits on disk)
RESULTS_PHASE2B_DIR = Path("results/phase2b")
CONFIG_C_PATH = RESULTS_PHASE2B_DIR / "configuration_c_flat_pipeline_multilingual.csv"
ARABIC_PILOT_PATH = RESULTS_PHASE2B_DIR / "configuration_c_arabic_pilot.csv"
GENERATION_LOG_PATH = RESULTS_PHASE2B_DIR / "generation_log.csv"

OUTPUT_DIR = Path("results/phase3a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config_c_with_paths():
    """Load Config C multilingual scores and attach real image file paths."""
    df = pd.read_csv(CONFIG_C_PATH)
    df["compliant"] = df["compliance_score"] >= COMPLIANCE_THRESHOLD

    gen = pd.read_csv(GENERATION_LOG_PATH)
    gen_ok = (
        gen[gen["status"] == "SUCCESS"][["model", "language", "prompt_id", "output_path"]]
        .drop_duplicates()
        .rename(columns={"prompt_id": "brief_id"})
    )
    df = df.merge(gen_ok, on=["brief_id", "model", "language"], how="left")

    n_missing = df["output_path"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} rows have no matching generated image file")
    return df


def build_120_sample(df):
    """Stratified sample: N_PER_CELL images per (language, model, compliant)."""
    rows = []
    for lang in LANGUAGES:
        for model in MODELS:
            for comp_flag in [True, False]:
                pool = df[(df.language == lang) & (df.model == model) & (df.compliant == comp_flag)]
                take = min(N_PER_CELL, len(pool))
                if take < N_PER_CELL:
                    print(f"WARNING: only {len(pool)} available for "
                          f"{lang}/{model}/compliant={comp_flag}, needed {N_PER_CELL}")
                rows.append(pool.sample(n=take, random_state=SEED))
    return pd.concat(rows, ignore_index=True)


def build_interrater_subset(sample120):
    """30 images: 10 per language, drawn from the 120-image sample."""
    rows = []
    for lang in LANGUAGES:
        pool = sample120[sample120.language == lang]
        rows.append(pool.sample(n=10, random_state=SEED))
    return pd.concat(rows, ignore_index=True)


def build_intrarater_subset(sample120):
    """20 images: independent random draw from the 120-image sample."""
    return sample120.sample(n=20, random_state=SEED + 1)


def get_attention_check_image():
    """Reuse the worst-scoring Arabic pilot image — an unambiguous failure,
    not one of the 120 carefully-balanced images."""
    ar = pd.read_csv(ARABIC_PILOT_PATH)
    worst = ar.loc[ar["compliance_score"].idxmin()]

    gen = pd.read_csv(GENERATION_LOG_PATH)
    gen_ok = gen[gen["status"] == "SUCCESS"][["model", "language", "prompt_id", "output_path"]] \
        .drop_duplicates().rename(columns={"prompt_id": "brief_id"})
    match = gen_ok[
        (gen_ok.brief_id == worst["brief_id"]) &
        (gen_ok.model == worst["model"]) &
        (gen_ok.language == "ar")
    ]
    return {
        "brief_id": worst["brief_id"],
        "model": worst["model"],
        "language": "ar",
        "compliance_score": worst["compliance_score"],
        "output_path": match["output_path"].iloc[0] if len(match) else None,
    }


def main():
    df = load_config_c_with_paths()

    sample120 = build_120_sample(df)
    interrater_30 = build_interrater_subset(sample120)
    intrarater_20 = build_intrarater_subset(sample120)
    attention_check = get_attention_check_image()

    sample120.to_csv(OUTPUT_DIR / "phase3a_120_sample.csv", index=False)
    interrater_30.to_csv(OUTPUT_DIR / "phase3a_interrater_30.csv", index=False)
    intrarater_20.to_csv(OUTPUT_DIR / "phase3a_intrarater_20.csv", index=False)
    pd.DataFrame([attention_check]).to_csv(OUTPUT_DIR / "phase3a_attention_check.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}/:")
    print(f"  phase3a_120_sample.csv       ({len(sample120)} images)")
    print(f"  phase3a_interrater_30.csv    ({len(interrater_30)} images)")
    print(f"  phase3a_intrarater_20.csv    ({len(intrarater_20)} images)")
    print(f"  phase3a_attention_check.csv  (1 image: {attention_check['brief_id']}/"
          f"{attention_check['model']}/ar)")
    print("\nBalance check (120-sample):")
    print(sample120.groupby(["language", "model", "compliant"]).size())


if __name__ == "__main__":
    main()