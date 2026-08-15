"""
Phase 2B COMETKiwi Scoring
Runs Unbabel/wmt22-cometkiwi-da (reference-free MT quality estimation) on
the four Phase 2B translated MBB brief files, adding a `cometkiwi_score`
column to each — same protocol as phase1_cometkiwi_scoring.py, adapted for
the mbb column names (prompt_text_en / prompt_text instead of
caption_en / caption_translated).

Run from project root, in the mulbrandeval-comet env (NOT mulbrandeval —
same torchmetrics conflict as Phase 1, see memory / phase2b_translate.py
docstring):

    conda activate mulbrandeval-comet
    python phase2b_multilingual/phase2b_cometkiwi.py

Prerequisite: phase2b_translate.py must have finished writing all four
mbb_briefs_{lang}.csv files (150 rows each) before this runs.

Requires: unbabel-comet, HF_TOKEN in .env (same as Phase 1).
"""

import os
import pandas as pd
from dotenv import load_dotenv
from comet import download_model, load_from_checkpoint

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not found in .env file. Add HF_TOKEN=your_token_here to .env "
        "in the project root before running this script."
    )

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
OUTPUT_DIR = "results/phase2b_cometkiwi"

# Files to score: (path, language_code)
FILES = [
    ("data/prompts/mbb/mbb_briefs_de.csv", "de"),
    ("data/prompts/mbb/mbb_briefs_fr.csv", "fr"),
    ("data/prompts/mbb/mbb_briefs_es.csv", "es"),
    ("data/prompts/mbb/mbb_briefs_ar.csv", "ar"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading/loading model: {MODEL_NAME} ...")
    # load_dotenv() above already placed HF_TOKEN into the process environment.
    # huggingface_hub automatically checks the HF_TOKEN env var for auth,
    # so no need to pass it manually into download_model().
    model_path = download_model(MODEL_NAME)
    model = load_from_checkpoint(model_path)
    print("Model loaded.\n")

    summary_rows = []

    for filename, lang in FILES:
        print(f"Scoring {filename} ...")

        if not os.path.exists(filename):
            print(f"  SKIPPED — file not found. Has phase2b_translate.py "
                  f"finished for '{lang}' yet?")
            continue

        df = pd.read_csv(filename)

        # mbb files use prompt_text_en / prompt_text, not caption_en /
        # caption_translated — the rest of the pipeline (state.py,
        # node5_brand_quality.py) is already locked to prompt_text_en, so
        # phase2b_translate.py wrote columns to match rather than the coco
        # naming convention.
        if "prompt_text_en" not in df.columns or "prompt_text" not in df.columns:
            print(f"  SKIPPED — expected columns (prompt_text_en, prompt_text) "
                  f"not found in {filename}")
            continue

        if len(df) != 150:
            print(f"  WARNING — expected 150 rows, found {len(df)} in {filename}. "
                  f"Scoring anyway, but check phase2b_translate.py completed cleanly.")

        data = [
            {"src": row["prompt_text_en"], "mt": row["prompt_text"]}
            for _, row in df.iterrows()
        ]

        model_output = model.predict(data, batch_size=8, gpus=0)
        scores = model_output["scores"]

        df["cometkiwi_score"] = scores

        # Overwrite in place — nothing has been committed to Git yet for
        # Phase 2B, unlike Phase 1's retrofit where a separate *_cometkiwi.csv
        # was kept to avoid disturbing already-committed files.
        df.to_csv(filename, index=False)
        print(f"  Saved: {filename}  (mean COMETKiwi = {sum(scores)/len(scores):.4f})")

        summary_rows.append({
            "phase": "phase2b",
            "language": lang,
            "n_rows": len(df),
            "mean_cometkiwi": sum(scores) / len(scores),
            "min_cometkiwi": min(scores),
            "max_cometkiwi": max(scores),
            "mean_cc_score": df["cc_score"].mean() if "cc_score" in df.columns else None,
        })

    print("\n=== Summary: COMETKiwi vs existing cycle-consistency, by language ===")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_path = os.path.join(OUTPUT_DIR, "phase2b_cometkiwi_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # ── Flag any low scores for manual review ──────────────────────────────
    # Phase 1 established Arabic scores consistently lower on COMETKiwi
    # (0.77-0.78) vs European languages (0.83-0.85) even when cycle-
    # consistency alone looked fine — worth re-checking that pattern holds
    # for the mbb briefs specifically, not just the coco/phase1b prompts.
    if not summary_df.empty:
        print("\n=== Phase 1 comparison context ===")
        print("Phase 1A/1B COMETKiwi means — DE: ~0.83-0.85, FR: ~0.83-0.85, "
              "ES: ~0.83-0.85, AR: ~0.77-0.78")
        print("Compare against the means above to check whether the Arabic "
              "gap pattern is consistent for mbb-specific prompts.")


if __name__ == "__main__":
    main()