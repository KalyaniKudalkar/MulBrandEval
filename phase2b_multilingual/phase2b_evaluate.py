"""
MulBrandEval — Phase 2B Evaluation Script
==========================================
Configuration B — Full DAG (primary system), multilingual (DE/FR/ES + optional AR)

Runs the compiled LangGraph DAG (phase2a_dag_pipeline.pipeline.app) on
generated Phase 2B images and writes one flattened row per image.

DE/FR/ES rows (the thesis's primary CLCG-eligible dataset) go to:
    results/phase2b/multilingual_ablation_table.csv

AR rows go to a SEPARATE file:
    results/phase2b/arabic_noncomparable_diagnostics.csv

This split is deliberate, not cosmetic. Arabic's per-node DAG scores are
NOT valid CLCG measurements — the CLIP text encoder never meaningfully
received the Arabic prompt in the first place (documented root cause:
functional Arabic-script failure, independent of token truncation), so a
CLCG(EN, AR) gap would be measuring encoder noise, not compliance
degradation. Prof. Chandna approved skipping the full 150-brief Arabic run
on these grounds. AR rows must never end up in the same table used to
build the per-node CLCG heatmap.

What AR evaluation IS useful for: quantitative diagnostic evidence inside
the cautionary case-study writeup itself (e.g. "Node 1 object presence
collapsed to X% on the 20 pilot images" is stronger evidence than the
qualitative description alone). Hence the separate output file — usable
as case-study evidence, structurally incapable of contaminating the
heatmap.

--languages controls which language(s) to evaluate, mirroring
phase2b_generate.py's own --languages flag. Defaults to de,fr,es (the
thesis's actual scope). Only 20 AR pilot images exist on disk (10 briefs x
2 models) — image-existence-awareness means passing --languages ar simply
evaluates those 20 and silently skips the other 130 briefs with no image
yet, no special-casing required.

Resume-safe: (brief_id, model, language) TRIPLES already present in the
relevant output CSV are skipped — not the (brief_id, model) pair used in
Phase 2A, since Phase 2B has multiple languages sharing the same
brief_id/model space. Image-existence-aware throughout.

Each brief row already carries both prompt_text (translated) and
prompt_text_en (English original) columns from the Phase 2B translation
step — no override needed here, unlike Phase 2A's English-only script.

Place this script at:  phase2b_multilingual/phase2b_evaluate.py
  (lives alongside phase2b_generate.py and phase2b_translate.py — an
  explicit sys.path insertion below makes the phase2a_dag_pipeline
  import resolve correctly from here, without needing -m invocation
  or moving this file into that package.)

Run from MulBrandEval/ root, same plain-script style as phase2b_generate.py:
  # Thesis scope (default):
  python phase2b_multilingual\\phase2b_evaluate.py

  # Add the Arabic case-study diagnostics (20 pilot images only):
  python phase2b_multilingual\\phase2b_evaluate.py --languages ar

  # Both in one run:
  python phase2b_multilingual\\phase2b_evaluate.py --languages de,fr,es,ar
"""

import sys
import csv
import time
import logging
import argparse
import gc
from pathlib import Path

# ── Make phase2a_dag_pipeline importable from this script's location ──────────
# This file lives in phase2b_multilingual/, one level below the project root,
# while the DAG engine it needs (pipeline.app) lives in phase2a_dag_pipeline/,
# also directly under the project root. Running this as a plain script (not
# `python -m ...`) only puts phase2b_multilingual/ itself on sys.path by
# default, so the import below would fail without this line. Inserting the
# project root explicitly makes phase2a_dag_pipeline resolve correctly,
# matching phase2b_generate.py's plain-script invocation style exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from phase2a_dag_pipeline.pipeline import app

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

BRIEFS_FILES = {
    "de": "data/prompts/mbb/mbb_briefs_de.csv",
    "fr": "data/prompts/mbb/mbb_briefs_fr.csv",
    "es": "data/prompts/mbb/mbb_briefs_es.csv",
    "ar": "data/prompts/mbb/mbb_briefs_ar.csv",
}
ALL_LANGUAGES     = ["de", "fr", "es", "ar"]   # canonical order, matches phase2b_generate.py
DEFAULT_LANGUAGES = ["de", "fr", "es"]          # thesis scope — CLCG-eligible languages
IMAGE_ROOT = Path("data/generated_images/phase2b")
MODELS     = ["sd15", "flux"]

MAIN_OUTPUT_FILE = Path("results/phase2b/multilingual_ablation_table.csv")        # DE/FR/ES only
AR_OUTPUT_FILE    = Path("results/phase2b/arabic_noncomparable_diagnostics.csv")  # AR only, never merged


def output_file_for(language: str) -> Path:
    """Route each language's rows to the correct output file. AR is
    permanently isolated from the CLCG-eligible table — see module docstring."""
    return AR_OUTPUT_FILE if language == "ar" else MAIN_OUTPUT_FILE

OUTPUT_FIELDS = [
    "brief_id", "model", "language",
    "node1_pass", "node1_score", "node1_short_circuit",
    "node2_pass", "node2_score", "node2_skipped",
    "node3_pass", "node3_score", "node3_skipped",
    "node4_pass", "node4_score", "node4_skipped",
    "node4_skipped_reason", "node4_clcg_interpretable",
    "node5_pass", "node5_score",
    "compliance_score", "failure_diagnosis",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Helpers ─────────────────────────────────────────────────────────────────────

def image_path(model: str, language: str, brief_id: str) -> Path:
    return IMAGE_ROOT / model / language / f"{brief_id}_seed42.png"


def load_briefs(language: str) -> pd.DataFrame:
    df = pd.read_csv(BRIEFS_FILES[language])
    n = len(df)
    assert n == 150, (
        f"Expected 150 briefs, found {n} in {BRIEFS_FILES[language]} — check the file."
    )
    return df


def already_evaluated(output_file: Path) -> set:
    """Return the set of (brief_id, model, language) TRIPLES already in the given output CSV."""
    if not output_file.exists():
        return set()
    df = pd.read_csv(output_file)
    if df.empty:
        return set()
    return set(zip(df["brief_id"], df["model"], df["language"]))


def flatten_result(brief_id: str, model: str, language: str, result: dict) -> dict:
    """Flatten the DAG's nested compliance_report into one CSV row."""
    r = result["compliance_report"]
    return {
        "brief_id":                 brief_id,
        "model":                    model,
        "language":                 language,
        "node1_pass":               r["node1"]["pass"],
        "node1_score":              r["node1"]["score"],
        "node1_short_circuit":      r["node1"]["short_circuit"],
        "node2_pass":               r["node2"]["pass"],
        "node2_score":              r["node2"]["score"],
        "node2_skipped":            r["node2"]["skipped"],
        "node3_pass":               r["node3"]["pass"],
        "node3_score":              r["node3"]["score"],
        "node3_skipped":            r["node3"]["skipped"],
        "node4_pass":               r["node4"]["pass"],
        "node4_score":              r["node4"]["score"],
        "node4_skipped":            r["node4"]["skipped"],
        "node4_skipped_reason":     r["node4"]["skipped_reason"],
        "node4_clcg_interpretable": r["node4"]["clcg_interpretable"],
        "node5_pass":               r["node5"]["pass"],
        "node5_score":              r["node5"]["score"],
        "compliance_score":         result["compliance_score"],
        "failure_diagnosis":        result["failure_diagnosis"],
    }

def parse_languages(arg_value: str) -> list:
    """Parse and validate a comma-separated --languages argument, same
    contract as phase2b_generate.py's parser (canonical order, dedup)."""
    requested = [x.strip().lower() for x in arg_value.split(",") if x.strip()]
    if not requested:
        raise argparse.ArgumentTypeError(
            "--languages received an empty value. "
            f"Valid options are: {', '.join(ALL_LANGUAGES)} (comma-separated, any subset)."
        )
    invalid = [x for x in requested if x not in ALL_LANGUAGES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown language code(s): {invalid}. "
            f"Valid options are: {', '.join(ALL_LANGUAGES)}."
        )
    return [l for l in ALL_LANGUAGES if l in requested]

# ── Main loop ──────────────────────────────────────────────────────────────────

def run(languages: list):
    MAIN_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AR_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Separate "already done" bookkeeping per output file — AR's dedup set
    # must never be conflated with the main DE/FR/ES set, or vice versa.
    done_main = already_evaluated(MAIN_OUTPUT_FILE)
    done_ar   = already_evaluated(AR_OUTPUT_FILE)

    jobs = []
    skipped_no_image = 0

    for lang in languages:
        briefs = load_briefs(lang)
        done = done_ar if lang == "ar" else done_main
        for _, row in briefs.iterrows():
            brief_id = row["brief_id"]
            for model in MODELS:
                if (brief_id, model, lang) in done:
                    continue
                img = image_path(model, lang, brief_id)
                if not img.exists():
                    skipped_no_image += 1
                    continue
                jobs.append((brief_id, model, lang, row.to_dict(), img))

    log.info(f"Phase 2B evaluation (Configuration B, languages={','.join(languages)}) — "
             f"{len(jobs)} image(s) ready | "
             f"{len(done_main) + len(done_ar)} already evaluated | "
             f"{skipped_no_image} not yet generated")

    # One writer per output file, opened lazily only if that file's
    # language(s) are actually part of this run.
    writers = {}
    handles = {}
    for target in {MAIN_OUTPUT_FILE, AR_OUTPUT_FILE}:
        if target == AR_OUTPUT_FILE and "ar" not in languages:
            continue
        if target == MAIN_OUTPUT_FILE and not any(l != "ar" for l in languages):
            continue
        write_header = not target.exists()
        fh = open(target, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        handles[target] = fh
        writers[target] = writer

    success = 0
    failures = 0

    for i, (brief_id, model, lang, brief_row, img) in enumerate(jobs, 1):
        brief = dict(brief_row)
        # prompt_text and prompt_text_en both already present from Phase 2B
        # translation step — no override needed (unlike Phase 2A's English-only case).
        brief["image_path"] = str(img)
        brief["model"]      = model
        brief["language"]   = lang

        target = output_file_for(lang)
        t0 = time.time()
        try:
            result  = app.invoke(brief)
            row_out = flatten_result(brief_id, model, lang, result)
            writers[target].writerow(row_out)
            handles[target].flush()
            dur = round(time.time() - t0, 1)
            success += 1
            log.info(f"[{i}/{len(jobs)}] [OK] {brief_id} | {model} | {lang} | "
                     f"compliance_score={result['compliance_score']} ({dur}s)"
                     + ("  [case-study, not CLCG-eligible]" if lang == "ar" else ""))
        except Exception as e:
            failures += 1
            log.error(f"[{i}/{len(jobs)}] [FAIL] {brief_id} | {model} | {lang} -> {e}")

        # Periodic garbage collection — mitigates gradual memory growth over
        # long runs (observed: Python process climbing ~1GB over ~35 min on
        # an 8GB machine). Doesn't touch scoring/output, purely reclaims
        # memory Python would otherwise hold onto longer than necessary.
        if i % 20 == 0:
            gc.collect()

    for fh in handles.values():
        fh.close()

    log.info("=" * 60)
    log.info(f"Evaluation run complete — [OK] {success}  [FAIL] {failures}")
    for target in handles:
        log.info(f"Results file: {target}")
    if skipped_no_image:
        log.info(f"{skipped_no_image} image(s) not yet generated — re-run this script "
                 f"once generation finishes to pick them up.")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2B multilingual DAG evaluation")
    parser.add_argument(
        "--languages", type=str, default=",".join(DEFAULT_LANGUAGES),
        help=(
            "Comma-separated language codes to evaluate, any subset of "
            f"{{{', '.join(ALL_LANGUAGES)}}}. Defaults to de,fr,es (thesis scope). "
            "Pass 'ar' to also/only evaluate the 20 existing Arabic pilot images "
            "as case-study diagnostics — written to a separate output file, "
            "never merged into the CLCG-eligible table."
        ),
    )
    args = parser.parse_args()
    run(parse_languages(args.languages))