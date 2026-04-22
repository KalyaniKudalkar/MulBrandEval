# phase1a_realism_baseline/translate_prompts.py
# Task 2: Translate 50 EN prompts → DE, FR, ES, AR using DeepL
# + Cycle Consistency verification using mCLIP cosine similarity
# Thresholds: PASS >= 0.85 | REVIEW >= 0.75 | FAIL < 0.75

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import deepl
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils.config import DEEPL_AUTH_KEY, MCLIP_MODEL_ID

# ── Config ─────────────────────────────────────────────────────
LANGUAGES = {
    "DE": "DE",
    "FR": "FR",
    "ES": "ES",
    "AR": "AR"
}
CC_PASS   = 0.85
CC_REVIEW = 0.75

INPUT_CSV  = Path("data/prompts/en/coco_prompts_en.csv")
OUTPUT_DIR = Path("data/prompts")
TRANS_DIR  = Path("data/translations")
TRANS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load DeepL client
    print("Initialising DeepL client...")
    translator = deepl.Translator(DEEPL_AUTH_KEY)

    # Verify account
    usage = translator.get_usage()
    print(f"DeepL usage: {usage.character.count:,} / {usage.character.limit:,} characters used\n")

    # Load mCLIP
    print("Loading mCLIP model (first run downloads ~1GB)...")
    model = SentenceTransformer(MCLIP_MODEL_ID)
    print("mCLIP ready.\n")

    # Load English prompts
    df_en = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df_en)} English prompts.\n")

    all_results = []

    for lang_code, deepl_code in LANGUAGES.items():
        print(f"── Translating to {lang_code} ──")
        lang_output_dir = OUTPUT_DIR / lang_code.lower()
        lang_output_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        for _, row in df_en.iterrows():
            original  = row["caption"]
            prompt_id = row["prompt_id"]

            # Forward translation EN → target
            translated = translator.translate_text(
                original,
                source_lang="EN",
                target_lang=deepl_code
            ).text
            time.sleep(0.2)

            # Back translation target → EN
            back_translated = translator.translate_text(
                translated,
                source_lang=deepl_code,
                target_lang="EN-GB"
            ).text
            time.sleep(0.2)

            # mCLIP cycle consistency score
            embeddings = model.encode(
                [original, back_translated],
                normalize_embeddings=True
            )
            cc_score = float(np.dot(embeddings[0], embeddings[1]))

            # Status
            if cc_score >= CC_PASS:
                status = "PASS"
            elif cc_score >= CC_REVIEW:
                status = "REVIEW"
            else:
                status = "FAIL"

            rows.append({
                "prompt_id":          prompt_id,
                "category":           row["category"],
                "image_id":           row["image_id"],
                "caption_en":         original,
                "caption_translated": translated,
                "back_translated":    back_translated,
                "cc_score":           round(cc_score, 4),
                "status":             status,
                "language":           lang_code.lower()
            })

            print(f"  {prompt_id} | CC={cc_score:.3f} | {status}")

        df_lang = pd.DataFrame(rows)

        # Save translated prompts
        trans_csv = lang_output_dir / f"coco_prompts_{lang_code.lower()}.csv"
        df_lang.to_csv(trans_csv, index=False)

        # Save CC report
        cc_csv = TRANS_DIR / f"cc_report_{lang_code.lower()}.csv"
        df_lang.to_csv(cc_csv, index=False)

        pass_count = (df_lang["status"] == "PASS").sum()
        flag_count = (df_lang["status"] != "PASS").sum()
        print(f"\n  Results: {pass_count}/50 PASS | {flag_count} flagged")
        print(f"  Avg CC score: {df_lang['cc_score'].mean():.4f}")
        print(f"  Saved to {trans_csv}\n")

        all_results.append(df_lang)

    # Combined report
    df_all = pd.concat(all_results, ignore_index=True)
    combined_path = TRANS_DIR / "cc_report_all_languages.csv"
    df_all.to_csv(combined_path, index=False)

    # Summary table
    print("── Cycle Consistency Summary ──")
    summary = df_all.groupby("language").agg(
        avg_cc      = ("cc_score", "mean"),
        pass_count  = ("status", lambda x: (x == "PASS").sum()),
        review_count= ("status", lambda x: (x == "REVIEW").sum()),
        fail_count  = ("status", lambda x: (x == "FAIL").sum())
    ).round(4)
    print(summary.to_string())
    print(f"\nFull report saved to {combined_path}")

if __name__ == "__main__":
    main()