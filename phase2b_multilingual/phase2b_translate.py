"""
MulBrandEval — Phase 2B Translation Script
===========================================
Translates all 150 MBB briefs' `prompt_text` field into DE / FR / ES / AR
via DeepL, and scores cycle-consistency (back-translate to EN, mCLIP cosine
similarity against the original) — same protocol as Phase 1A/1B and the
`pilot_brandname_check.py` pilot run.

Locked decisions this script enforces (do not change without re-checking
Phase2A_Node4_Typography_MethodologyDecision.docx / Point 2 resolution):
  - `required_text` is NEVER passed to DeepL — copied unchanged from EN.
    (Pilot run proved DeepL breaks it in isolation ~60% of the time.)
  - `required_objects`, `colour_attributes`, `spatial_constraints`,
    `style_descriptors` are NEVER translated — the DAG evaluates these
    structured fields in English regardless of image language (Point 2:
    evaluator locked to English fields by construction).
  - Only `prompt_text` (the literal image-generation prompt) is translated.

Run from project root, in the `mulbrandeval` env (NOT `mulbrandeval-comet`
— COMETKiwi scoring is a separate script, `phase2b_cometkiwi.py`, due to
the torchmetrics version conflict documented in memory):

    python phase2b_multilingual/phase2b_translate.py

Output: data/prompts/mbb/mbb_briefs_{lang}.csv  for lang in de, fr, es, ar
Columns: brief_id, industry, brand_name, prompt_text, prompt_text_en,
         required_objects, colour_attributes, spatial_constraints,
         required_text, style_descriptors, back_translated, cc_score,
         status, language

Resume-safe: if an output file already has N rows translated, this script
picks up from row N+1 rather than re-translating (re-translating costs
DeepL character quota for no reason and cycle-consistency is deterministic
given the same translation, so nothing is gained by re-running rows).
"""

import os
import csv
from pathlib import Path

import deepl
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIEFS_PATH  = PROJECT_ROOT / "data" / "prompts" / "mbb" / "mbb_briefs_en.csv"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "prompts" / "mbb"

# DeepL target-lang codes -> our internal language tag
LANGS = {
    "de": "DE",
    "fr": "FR",
    "es": "ES",
    "ar": "AR",
}

PASS_THRESHOLD   = 0.85
REVIEW_THRESHOLD = 0.75

# Columns copied unchanged from the English brief — never translated
STATIC_COLUMNS = [
    "required_objects",
    "colour_attributes",
    "spatial_constraints",
    "required_text",
    "style_descriptors",
]

OUTPUT_FIELDS = [
    "brief_id", "industry", "brand_name",
    "prompt_text", "prompt_text_en",
    "required_objects", "colour_attributes", "spatial_constraints",
    "required_text", "style_descriptors",
    "back_translated", "cc_score", "status", "language",
]

# ── Setup ─────────────────────────────────────────────────────────────────
DEEPL_API_KEY = os.getenv("DEEPL_AUTH_KEY")
if not DEEPL_API_KEY:
    raise RuntimeError(
        "DEEPL_AUTH_KEY not found in .env — check the key name matches "
        "exactly (this is the name used by pilot_brandname_check.py)."
    )

translator = deepl.Translator(DEEPL_API_KEY)

print("Loading mCLIP for cycle-consistency scoring...")
mclip = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")
print("mCLIP loaded.\n")


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


def load_briefs() -> list[dict]:
    with open(BRIEFS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 150, f"Expected 150 briefs, found {len(rows)} in {BRIEFS_PATH}"
    return rows


def already_done_ids(output_path: Path) -> set:
    """Return brief_ids already present in an existing output file (resume support)."""
    if not output_path.exists():
        return set()
    with open(output_path, encoding="utf-8") as f:
        return {row["brief_id"] for row in csv.DictReader(f)}


def translate_one(row: dict, lang_code: str) -> dict:
    """Translate one brief's prompt_text and score cycle-consistency."""
    prompt_en = row["prompt_text"]

    translated_prompt = translator.translate_text(
        prompt_en, target_lang=lang_code
    ).text

    back_translated = translator.translate_text(
        translated_prompt, target_lang="EN-US"
    ).text

    score  = cc_score(prompt_en, back_translated)
    status = status_from_score(score)

    out = {col: row[col] for col in ["brief_id", "industry", "brand_name"] + STATIC_COLUMNS}
    out["prompt_text"]    = translated_prompt
    out["prompt_text_en"] = prompt_en
    out["back_translated"] = back_translated
    out["cc_score"]        = round(score, 4)
    out["status"]          = status
    return out


def run():
    briefs = load_briefs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_review_or_fail = []

    for lang_tag, lang_code in LANGS.items():
        output_path = OUTPUT_DIR / f"mbb_briefs_{lang_tag}.csv"
        done_ids    = already_done_ids(output_path)

        remaining = [r for r in briefs if r["brief_id"] not in done_ids]
        if not remaining:
            print(f"[{lang_tag.upper()}] All 150 briefs already translated — skipping.")
            continue

        print(f"\n[{lang_tag.upper()}] {len(done_ids)} already done | "
              f"{len(remaining)} to translate")

        file_exists = output_path.exists()
        fh     = open(output_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        if not file_exists:
            writer.writeheader()

        for row in tqdm(remaining, desc=f"{lang_tag.upper()}", unit="brief"):
            try:
                out = translate_one(row, lang_code)
                out["language"] = lang_tag
                writer.writerow(out)
                fh.flush()

                if out["status"] != "PASS":
                    all_review_or_fail.append(
                        (out["brief_id"], lang_tag, out["cc_score"], out["status"])
                    )
            except Exception as e:
                print(f"\n  [ERROR] {row['brief_id']} | {lang_tag} | {e}")
                print("  Stopping this language's run so you can inspect the "
                      "error — already-written rows are saved, safe to resume.")
                fh.close()
                raise

        fh.close()
        print(f"[{lang_tag.upper()}] Done -> {output_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2B TRANSLATION COMPLETE")
    print("=" * 60)
    if all_review_or_fail:
        print(f"\n{len(all_review_or_fail)} REVIEW/FAIL cases need a look:")
        for bid, lang, score, status in all_review_or_fail:
            print(f"  {bid} | {lang} | cc_score={score:.4f} | {status}")
    else:
        print("\nAll translations PASSED cycle-consistency threshold.")
    print(f"\nOutput files: {OUTPUT_DIR}/mbb_briefs_{{de,fr,es,ar}}.csv")
    print("Next: run phase2b_cometkiwi.py in the mulbrandeval-comet env "
          "to add the cometkiwi_score column.")


if __name__ == "__main__":
    run()