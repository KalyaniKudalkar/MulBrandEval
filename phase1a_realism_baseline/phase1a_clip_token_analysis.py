"""
MulBrandEval — Phase 1A CLIP Token Truncation Analysis (COCO prompts)
=======================================================================
Companion to phase2b_clip_token_analysis.py. Runs the same CLIP-token
measurement on Phase 1A's COCO captions to test whether the Arabic
generation failure seen throughout Phase 1A/1B/2B is explained by
truncation (as it is for Phase 2B's longer MBB briefs) or not.

Expectation going in: COCO captions are short, single-sentence image
descriptions -- much shorter than MBB's dense multi-attribute briefs.
This script checks whether they still hit CLIP's 77-token limit in
Arabic. If they do NOT get truncated, and Arabic generation still fails
(which the Phase 1A image folders already show it does), that is
important evidence isolating the root cause: CLIP's text encoder lacking
functional Arabic understanding, independent of prompt length.

No brand_name field exists in COCO prompts (unlike MBB briefs), so this
script only measures token counts / truncation -- no brand-survival check.

Run from MulBrandEval/ root:
  python phase1a_realism_baseline\\phase1a_clip_token_analysis.py

Inputs:
  data/prompts/en/coco_prompts_en.csv   (caption)
  data/prompts/ar/coco_prompts_ar.csv   (caption_translated)

Outputs:
  results/phase1a/coco_token_comparison.csv
  results/phase1a/coco_token_analysis_summary.txt
"""

import pandas as pd
import open_clip
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────

EN_PATH = Path("data/prompts/en/coco_prompts_en.csv")
AR_PATH = Path("data/prompts/ar/coco_prompts_ar.csv")

EN_CAPTION_COL = "caption"              # matches phase1a_generate.py's LANGUAGES dict
AR_CAPTION_COL = "caption_translated"   # matches phase1a_generate.py's LANGUAGES dict

OUTPUT_CSV     = Path("results/phase1a/coco_token_comparison.csv")
OUTPUT_SUMMARY = Path("results/phase1a/coco_token_analysis_summary.txt")

CLIP_MAX_CONTENT_TOKENS = 75
CLIP_TOKENIZER_MODEL    = "ViT-B-32"

# ── Main analysis ────────────────────────────────────────────────────────

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = open_clip.get_tokenizer(CLIP_TOKENIZER_MODEL)

    df_en = pd.read_csv(EN_PATH)
    df_ar = pd.read_csv(AR_PATH)

    assert "prompt_id" in df_en.columns, f"Missing prompt_id in {EN_PATH}"
    assert "prompt_id" in df_ar.columns, f"Missing prompt_id in {AR_PATH}"
    assert EN_CAPTION_COL in df_en.columns, f"Missing {EN_CAPTION_COL} in {EN_PATH}"
    assert AR_CAPTION_COL in df_ar.columns, f"Missing {AR_CAPTION_COL} in {AR_PATH}"

    df_en_slim = df_en[["prompt_id", EN_CAPTION_COL]].rename(
        columns={EN_CAPTION_COL: "caption_en"})
    df_ar_slim = df_ar[["prompt_id", AR_CAPTION_COL]].rename(
        columns={AR_CAPTION_COL: "caption_ar"})

    merged = df_en_slim.merge(df_ar_slim, on="prompt_id")
    n = len(merged)
    print(f"Matched {n} prompt_id rows between EN and AR COCO files "
          f"(EN had {len(df_en)}, AR had {len(df_ar)}).")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    merged["tokens_en"] = merged["caption_en"].apply(count_tokens)
    merged["tokens_ar"] = merged["caption_ar"].apply(count_tokens)
    merged["en_truncated"] = merged["tokens_en"] > CLIP_MAX_CONTENT_TOKENS
    merged["ar_truncated"] = merged["tokens_ar"] > CLIP_MAX_CONTENT_TOKENS
    merged["ar_tokens_dropped"] = (
        merged["tokens_ar"] - CLIP_MAX_CONTENT_TOKENS
    ).clip(lower=0)
    merged["token_ratio_ar_over_en"] = merged["tokens_ar"] / merged["tokens_en"]

    merged.to_csv(OUTPUT_CSV, index=False)

    lines = []
    lines.append("=" * 70)
    lines.append("Phase 1A — CLIP Token Analysis (COCO prompts, Arabic vs English)")
    lines.append(f"{n} matched prompts")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"English tokens -- mean: {merged['tokens_en'].mean():.1f}, "
                  f"median: {merged['tokens_en'].median():.0f}, "
                  f"min: {merged['tokens_en'].min()}, max: {merged['tokens_en'].max()}")
    lines.append(f"Arabic tokens  -- mean: {merged['tokens_ar'].mean():.1f}, "
                  f"median: {merged['tokens_ar'].median():.0f}, "
                  f"min: {merged['tokens_ar'].min()}, max: {merged['tokens_ar'].max()}")
    lines.append(f"Mean token ratio (AR/EN): {merged['token_ratio_ar_over_en'].mean():.2f}x")
    lines.append("")
    lines.append(f"English prompts truncated (>{CLIP_MAX_CONTENT_TOKENS} tokens): "
                  f"{merged['en_truncated'].sum()} / {n}")
    lines.append(f"Arabic prompts truncated (>{CLIP_MAX_CONTENT_TOKENS} tokens):  "
                  f"{merged['ar_truncated'].sum()} / {n} "
                  f"({merged['ar_truncated'].sum() / n * 100:.1f}%)")
    lines.append("")
    lines.append("Interpretation: if ar_truncated is 0 or near-0 here, this shows")
    lines.append("Arabic COCO captions are short enough to avoid truncation entirely,")
    lines.append("which isolates the deeper cause -- CLIP's Arabic text conditioning")
    lines.append("failing regardless of length -- from the compounding truncation")
    lines.append("effect seen separately in Phase 2B's longer MBB briefs.")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print(summary_text)
    OUTPUT_SUMMARY.write_text(summary_text, encoding="utf-8")

    print(f"\nPer-prompt data saved to : {OUTPUT_CSV}")
    print(f"Summary saved to         : {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()