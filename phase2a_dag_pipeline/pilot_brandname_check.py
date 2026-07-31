"""
Pilot check: does the brand name survive DeepL translation intact,
for both the translated prompt_text and the required_text field?
Run BEFORE translating all 150 MBB briefs for Phase 2B.
"""

import csv
import json
import os
from pathlib import Path

import deepl
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIEFS_PATH = PROJECT_ROOT / "data" / "prompts" / "mbb" / "mbb_briefs_en.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "phase2a" / "pilot_brandname_check.csv"

PILOT_IDS = ["MB001", "MB003", "MB050", "MB111", "MB144"]
LANGS = {"DE": "DE", "AR": "AR"}  # DeepL target-lang codes

PASS_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.75

# ── Setup ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

DEEPL_API_KEY = os.getenv("DEEPL_AUTH_KEY")
if not DEEPL_API_KEY:
    raise RuntimeError("DEEPL_API_KEY not found in .env")

translator = deepl.Translator(DEEPL_API_KEY)
mclip = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")


def cc_score(original_en: str, back_translated_en: str) -> float:
    """mCLIP cosine similarity between original EN and back-translated EN text."""
    a = torch.tensor(mclip.encode(original_en, convert_to_numpy=True), dtype=torch.float32)
    b = torch.tensor(mclip.encode(back_translated_en, convert_to_numpy=True), dtype=torch.float32)
    a = F.normalize(a.unsqueeze(0), dim=-1)
    b = F.normalize(b.unsqueeze(0), dim=-1)
    return float(F.cosine_similarity(a, b))


def status_from_score(score: float) -> str:
    if score >= PASS_THRESHOLD:
        return "PASS"
    elif score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "FAIL"


def main():
    with open(BRIEFS_PATH, encoding="utf-8") as f:
        rows = {r["brief_id"]: r for r in csv.DictReader(f)}

    results = []

    for bid in PILOT_IDS:
        if bid not in rows:
            print(f"  WARNING: {bid} not found in briefs file, skipping.")
            continue
        row = rows[bid]
        brand = row["brand_name"]
        prompt_en = row["prompt_text"]
        required_text_en = json.loads(row["required_text"])

        print(f"\n=== {bid} | brand='{brand}' ===")
        print(f"EN prompt: {prompt_en}")

        for lang_label, lang_code in LANGS.items():
            # 1. Translate full prompt_text
            translated_prompt = translator.translate_text(
                prompt_en, target_lang=lang_code
            ).text

            # 2. Check brand name survival in translated prompt
            brand_survived_prompt = brand in translated_prompt

            # 3. Translate required_text separately (to see what WOULD happen
            #    if this field were naively passed through DeepL too)
            translated_required_text = translator.translate_text(
                required_text_en[0], target_lang=lang_code
            ).text
            required_text_unchanged = translated_required_text == required_text_en[0]

            # 4. Back-translate the full prompt to EN for cycle-consistency scoring
            back_translated = translator.translate_text(
                translated_prompt, target_lang="EN-US"
            ).text
            score = cc_score(prompt_en, back_translated)
            status = status_from_score(score)

            print(f"\n  [{lang_label}]")
            print(f"    Translated prompt : {translated_prompt}")
            print(f"    Brand '{brand}' survived in translated prompt : {brand_survived_prompt}")
            print(f"    required_text translated naively would give  : '{translated_required_text}'"
                  f" (unchanged: {required_text_unchanged})")
            print(f"    Back-translated EN : {back_translated}")
            print(f"    Cycle-consistency score : {score:.4f} -> {status}")

            results.append({
                "brief_id": bid,
                "brand_name": brand,
                "language": lang_label,
                "translated_prompt": translated_prompt,
                "brand_survived_in_prompt": brand_survived_prompt,
                "required_text_naive_translation": translated_required_text,
                "required_text_would_survive_naive_translation": required_text_unchanged,
                "back_translated_en": back_translated,
                "cc_score": round(score, 4),
                "status": status,
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\n\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()