"""
Phase 1 COMETKiwi Scoring
Runs Unbabel/wmt22-cometkiwi-da (reference-free MT quality estimation) on all
existing Phase 1A and Phase 1B translated prompt files.

Run from project root:
    python -m phase1_cometkiwi_scoring

Requires: unbabel-comet, huggingface-cli login already done (see README).
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
OUTPUT_DIR = "results/phase1_cometkiwi"

# Files to score: (path, language_code, phase_label)
FILES = [
    ("data/prompts/de/coco_prompts_de.csv", "de", "phase1a"),
    ("data/prompts/fr/coco_prompts_fr.csv", "fr", "phase1a"),
    ("data/prompts/es/coco_prompts_es.csv", "es", "phase1a"),
    ("data/prompts/ar/coco_prompts_ar.csv", "ar", "phase1a"),
    ("data/prompts/de/phase1b_prompts_de.csv", "de", "phase1b"),
    ("data/prompts/fr/phase1b_prompts_fr.csv", "fr", "phase1b"),
    ("data/prompts/es/phase1b_prompts_es.csv", "es", "phase1b"),
    ("data/prompts/ar/phase1b_prompts_ar.csv", "ar", "phase1b"),
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

    for filename, lang, phase in FILES:
        print(f"Scoring {filename} ...")
        df = pd.read_csv(filename)

        if "caption_en" not in df.columns or "caption_translated" not in df.columns:
            print(f"  SKIPPED — expected columns not found in {filename}")
            continue

        data = [
            {"src": row["caption_en"], "mt": row["caption_translated"]}
            for _, row in df.iterrows()
        ]

        model_output = model.predict(data, batch_size=8, gpus=0)
        scores = model_output["scores"]

        df["cometkiwi_score"] = scores

        base_name = os.path.basename(filename).replace(".csv", "_cometkiwi.csv")
        out_path = os.path.join(OUTPUT_DIR, base_name)
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  (mean COMETKiwi = {sum(scores)/len(scores):.4f})")

        summary_rows.append({
            "phase": phase,
            "language": lang,
            "n_rows": len(df),
            "mean_cometkiwi": sum(scores) / len(scores),
            "min_cometkiwi": min(scores),
            "max_cometkiwi": max(scores),
            "mean_cc_score": df["cc_score"].mean() if "cc_score" in df.columns else None,
        })

    print("\n=== Summary: COMETKiwi vs existing cycle-consistency, by language/phase ===")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_path = os.path.join(OUTPUT_DIR, "phase1_cometkiwi_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()