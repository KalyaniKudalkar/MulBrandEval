"""
MulBrandEval — Phase 2B Arabic CLIP Token Truncation Analysis
================================================================
Quantifies how many of the 150 MBB briefs' Arabic prompt_text values
exceed CLIP's 77-token limit (75 content tokens + SOT/EOT), and whether
the brand name (required_text) falls in the truncated region.

Background / why this exists:
  A pilot generation run (10 briefs x 4 languages x 2 models) showed that
  Arabic-language images from both SD v1.5 and FLUX.1-dev bore no visible
  relation to their briefs, while DE/FR/ES images tracked their captions
  correctly. The same pattern was confirmed retroactively in Phase 1A and
  Phase 1B Arabic outputs (COCO and DrawBench/GenAI-Bench prompts), and
  reproduced independently on Replicate's own playground (outside this
  codebase entirely), ruling out a pipeline bug.

  Two contributing causes were identified:
    1. CLIP's text encoder has little functional understanding of Arabic
       script at all -- confirmed by testing a SHORT, fully-intact Arabic
       prompt ("silver smartwatch", well under the token limit), which
       still produced content unrelated to the brief.
    2. CLIP's Arabic tokenization is far less token-efficient than English,
       meaning most Arabic MBB prompts exceed the 77-token limit and get
       truncated -- silently dropping content, most often the brand name
       and style descriptors, which sit near the end of the MBB brief
       template's sentence structure.

  This script quantifies cause #2 across all 150 briefs, for the record
  and for citation in the thesis methodology/limitations section.

Requirements:
  pip install open_clip_torch --break-system-packages

Run from MulBrandEval/ root:
  python phase2b_multilingual\\phase2b_clip_token_analysis.py

Inputs:
  data/prompts/mbb/mbb_briefs_en.csv            (English prompt_text, brand_name)
  data/prompts/mbb/mbb_briefs_ar.csv            (Arabic prompt_text, brand_name)

Outputs:
  results/phase2b/mbb_token_comparison.csv      (per-brief token counts + truncation flags)
  Console summary (also written to results/phase2b/token_analysis_summary.txt)
"""

import pandas as pd
import open_clip
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────

EN_BRIEFS_PATH = Path("data/prompts/mbb/mbb_briefs_en.csv")
AR_BRIEFS_PATH = Path("data/prompts/mbb/mbb_briefs_ar.csv")

OUTPUT_CSV     = Path("results/phase2b/mbb_token_comparison.csv")
OUTPUT_SUMMARY = Path("results/phase2b/token_analysis_summary.txt")

CLIP_MAX_CONTENT_TOKENS = 75  # CLIP's 77-token limit minus SOT/EOT special tokens
CLIP_TOKENIZER_MODEL    = "ViT-B-32"  # Same CLIP tokenizer family used by SD v1.5

# ── Main analysis ────────────────────────────────────────────────────────

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = open_clip.get_tokenizer(CLIP_TOKENIZER_MODEL)

    df_en = pd.read_csv(EN_BRIEFS_PATH)
    df_ar = pd.read_csv(AR_BRIEFS_PATH)

    assert len(df_en) == 150, f"Expected 150 English briefs, found {len(df_en)}"
    assert len(df_ar) == 150, f"Expected 150 Arabic briefs, found {len(df_ar)}"

    df_en_slim = df_en[["brief_id", "prompt_text"]].rename(
        columns={"prompt_text": "prompt_en"})
    df_ar_slim = df_ar[["brief_id", "prompt_text", "brand_name"]].rename(
        columns={"prompt_text": "prompt_ar"})

    merged = df_en_slim.merge(df_ar_slim, on="brief_id")
    assert len(merged) == 150, "Merge lost rows -- check brief_id alignment between EN/AR files"

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    merged["tokens_en"] = merged["prompt_en"].apply(count_tokens)
    merged["tokens_ar"] = merged["prompt_ar"].apply(count_tokens)
    merged["en_truncated"] = merged["tokens_en"] > CLIP_MAX_CONTENT_TOKENS
    merged["ar_truncated"] = merged["tokens_ar"] > CLIP_MAX_CONTENT_TOKENS
    merged["ar_tokens_dropped"] = (
        merged["tokens_ar"] - CLIP_MAX_CONTENT_TOKENS
    ).clip(lower=0)
    merged["token_ratio_ar_over_en"] = merged["tokens_ar"] / merged["tokens_en"]

    # Where does the brand name sit in the raw Arabic string, as a fraction
    # of total character length? Used as a proxy for whether it likely
    # falls within the surviving ~75-token window or the truncated tail.
    def brand_char_fraction(row):
        idx = row["prompt_ar"].find(row["brand_name"])
        if idx == -1:
            return None
        return idx / len(row["prompt_ar"])

    merged["brand_char_fraction"] = merged.apply(brand_char_fraction, axis=1)

    # Approximate survival fraction: fraction of TOKENS (not characters)
    # that survive truncation, per brief.
    merged["ar_survival_fraction"] = (
        CLIP_MAX_CONTENT_TOKENS / merged["tokens_ar"]
    ).clip(upper=1.0)

    # Brand name flagged "likely truncated" if its character position falls
    # past the token-survival fraction (rough but directionally reliable
    # proxy, since char position and token position correlate closely for
    # a single script/language).
    merged["brand_likely_truncated"] = (
        merged["brand_char_fraction"] > merged["ar_survival_fraction"]
    )

    merged.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("Phase 2B — CLIP Token Truncation Analysis (Arabic vs English)")
    lines.append("All 150 MBB briefs")
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
                  f"{merged['en_truncated'].sum()} / 150")
    lines.append(f"Arabic prompts truncated (>{CLIP_MAX_CONTENT_TOKENS} tokens):  "
                  f"{merged['ar_truncated'].sum()} / 150 "
                  f"({merged['ar_truncated'].sum() / 150 * 100:.1f}%)")
    lines.append("")
    truncated = merged[merged["ar_truncated"]]
    if len(truncated) > 0:
        lines.append(f"Among truncated Arabic prompts, tokens dropped -- "
                      f"mean: {truncated['ar_tokens_dropped'].mean():.1f}, "
                      f"max: {truncated['ar_tokens_dropped'].max()}")
    lines.append("")
    lines.append(f"Brand name char-position -- mean fraction through sentence: "
                  f"{merged['brand_char_fraction'].mean():.3f}")
    lines.append(f"Brand name char-position -- median fraction through sentence: "
                  f"{merged['brand_char_fraction'].median():.3f}")
    lines.append(f"Briefs where brand name is likely in the truncated tail: "
                  f"{merged['brand_likely_truncated'].sum()} / 150 "
                  f"({merged['brand_likely_truncated'].sum() / 150 * 100:.1f}%)")
    lines.append("")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print(summary_text)
    OUTPUT_SUMMARY.write_text(summary_text, encoding="utf-8")

    print(f"\nPer-brief data saved to : {OUTPUT_CSV}")
    print(f"Summary saved to        : {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()