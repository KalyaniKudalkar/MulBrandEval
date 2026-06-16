# phase1b_alignment_baseline/translate_prompts.py
# Translate 100 Phase 1B prompts (DrawBench + GenAI-Bench) → DE, FR, ES, AR
# + Cycle Consistency verification using mCLIP cosine similarity
# Thresholds: PASS >= 0.85 | REVIEW >= 0.75 | FAIL < 0.75

import sys
import random
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import deepl
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.config import DEEPL_AUTH_KEY, MCLIP_MODEL_ID

# ── Config ──────────────────────────────────────────────────────────────────────

LANGUAGES = {
    "DE": "DE",
    "FR": "FR",
    "ES": "ES",
    "AR": "AR"
}
CC_PASS   = 0.85
CC_REVIEW = 0.75

INPUT_CSV  = Path("data/prompts/en/phase1b_prompts_en.csv")
OUTPUT_DIR = Path("data/prompts")
TRANS_DIR  = Path("data/translations")
TRANS_DIR.mkdir(parents=True, exist_ok=True)

# ── DeepL call with retry ───────────────────────────────────────────────────────

def translate_with_retry(translator, text, source_lang, target_lang, max_attempts=5):
    """
    Calls DeepL translate_text with exponential backoff on rate limit errors.
    Waits 10s, 20s, 30s... between retries with a small random jitter.
    """
    for attempt in range(max_attempts):
        try:
            result = translator.translate_text(
                text,
                source_lang=source_lang,
                target_lang=target_lang
            ).text
            time.sleep(1.5)   # polite delay after every successful call
            return result
        except Exception as e:
            wait = 10 * (attempt + 1) + random.uniform(1, 3)
            print(f"    [retry {attempt + 1}/{max_attempts}] DeepL error: {e}")
            print(f"    Waiting {wait:.0f}s before retrying...")
            time.sleep(wait)

    raise RuntimeError(
        f"DeepL translation failed after {max_attempts} attempts for text: {text[:60]}"
    )

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("Initialising DeepL client...")
    translator = deepl.Translator(DEEPL_AUTH_KEY)

    usage = translator.get_usage()
    print(f"DeepL usage: {usage.character.count:,} / {usage.character.limit:,} characters used\n")

    print("Loading mCLIP model...")
    model = SentenceTransformer(MCLIP_MODEL_ID)
    print("mCLIP ready.\n")

    df_en = pd.read_csv(INPUT_CSV)
    n = len(df_en)
    print(f"Loaded {n} English prompts from {INPUT_CSV}\n")

    all_results = []

    for lang_code, deepl_code in LANGUAGES.items():
        print(f"── Translating to {lang_code} ──")
        lang_output_dir = OUTPUT_DIR / lang_code.lower()
        lang_output_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        for idx, row in df_en.iterrows():
            original  = row["caption"]
            prompt_id = row["prompt_id"]

            # Forward translation: EN → target language
            translated = translate_with_retry(
                translator,
                text=original,
                source_lang="EN",
                target_lang=deepl_code
            )

            # Back translation: target language → EN
            back_translated = translate_with_retry(
                translator,
                text=translated,
                source_lang=deepl_code,
                target_lang="EN-GB"
            )

            # mCLIP cosine similarity (cycle consistency score)
            embeddings = model.encode(
                [original, back_translated],
                normalize_embeddings=True
            )
            cc_score = float(np.dot(embeddings[0], embeddings[1]))

            # Status assignment
            if cc_score >= CC_PASS:
                status = "PASS"
            elif cc_score >= CC_REVIEW:
                status = "REVIEW"
            else:
                status = "FAIL"

            rows.append({
                "prompt_id":          prompt_id,
                "category":           row["category"],
                "dag_node":           row["dag_node"],
                "source":             row["source"],
                "human_score":        row["human_score"],
                "caption_en":         original,
                "caption_translated": translated,
                "back_translated":    back_translated,
                "cc_score":           round(cc_score, 4),
                "status":             status,
                "language":           lang_code.lower()
            })

            print(f"  [{idx + 1:>3}/{n}] {prompt_id} | CC={cc_score:.3f} | {status}")

        df_lang = pd.DataFrame(rows)

        # Save translated prompts to language folder
        trans_csv = lang_output_dir / f"phase1b_prompts_{lang_code.lower()}.csv"
        df_lang.to_csv(trans_csv, index=False)

        # Save CC report to translations folder
        cc_csv = TRANS_DIR / f"phase1b_cc_report_{lang_code.lower()}.csv"
        df_lang.to_csv(cc_csv, index=False)

        pass_count   = (df_lang["status"] == "PASS").sum()
        review_count = (df_lang["status"] == "REVIEW").sum()
        fail_count   = (df_lang["status"] == "FAIL").sum()

        print(f"\n  Results: {pass_count}/{n} PASS | "
              f"{review_count} REVIEW | {fail_count} FAIL")
        print(f"  Avg CC score: {df_lang['cc_score'].mean():.4f}")
        print(f"  Saved to {trans_csv}\n")

        all_results.append(df_lang)

    # Combined report across all languages
    df_all = pd.concat(all_results, ignore_index=True)
    combined_path = TRANS_DIR / "phase1b_cc_report_all.csv"
    df_all.to_csv(combined_path, index=False)

    # Summary table
    print("── Cycle Consistency Summary ──")
    summary = df_all.groupby("language").agg(
        avg_cc       = ("cc_score", "mean"),
        pass_count   = ("status", lambda x: (x == "PASS").sum()),
        review_count = ("status", lambda x: (x == "REVIEW").sum()),
        fail_count   = ("status", lambda x: (x == "FAIL").sum())
    ).round(4)
    print(summary.to_string())
    print(f"\nFull report saved to {combined_path}")

if __name__ == "__main__":
    main()