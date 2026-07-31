"""
MulBrandEval — Phase 2A Evaluation Script
==========================================
Configuration B — Full DAG (primary system)

Runs the compiled LangGraph DAG (phase2a_dag_pipeline.pipeline.app) on every
generated Phase 2A image (150 briefs x 2 models, English baseline = up to
300 images) and writes one flattened row per image to:

    results/phase2a/english_ablation_table.csv

Resume-safe: (brief_id, model) pairs already present in the output CSV are
skipped. Image-existence-aware: only evaluates images that actually exist
on disk, so it's safe to run this alongside an in-progress generation run
and re-run it later to pick up newly generated images.

NOTE — this covers Configuration B only. Configurations A (CLIP-Score-only
monolithic baseline) and C (flat DAG, no short-circuit) are separate,
smaller scripts still to be written — see checklist Step 6.

Place this script at:  phase2a_dag_pipeline/phase2a_evaluate.py
Run from MulBrandEval/ root:  python -m phase2a_dag_pipeline.phase2a_evaluate
"""

import csv
import time
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from phase2a_dag_pipeline.pipeline import app

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

BRIEFS_FILE = "data/prompts/mbb/mbb_briefs_en.csv"   # = mbb_briefs_en_v3_FINAL
IMAGE_ROOT  = Path("data/generated_images/phase2a")
LANGUAGE    = "en"
MODELS      = ["sd15", "flux"]

OUTPUT_FILE = Path("results/phase2a/english_ablation_table.csv")

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

def image_path(model: str, brief_id: str) -> Path:
    return IMAGE_ROOT / model / LANGUAGE / f"{brief_id}_seed42.png"


def load_briefs() -> pd.DataFrame:
    df = pd.read_csv(BRIEFS_FILE)
    n = len(df)
    assert n == 150, f"Expected 150 briefs, found {n} in {BRIEFS_FILE} — check the file."
    return df


def already_evaluated() -> set:
    """Return the set of (brief_id, model) pairs already in the output CSV."""
    if not OUTPUT_FILE.exists():
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    if df.empty:
        return set()
    return set(zip(df["brief_id"], df["model"]))


def flatten_result(brief_id: str, model: str, result: dict) -> dict:
    """Flatten the DAG's nested compliance_report into one CSV row."""
    r = result["compliance_report"]
    return {
        "brief_id":                 brief_id,
        "model":                    model,
        "language":                 LANGUAGE,
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

# ── Main loop ──────────────────────────────────────────────────────────────────

def run():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    briefs = load_briefs()
    done   = already_evaluated()

    # Build job list: only (brief_id, model) pairs not yet evaluated AND
    # whose image actually exists on disk right now.
    jobs = []
    skipped_no_image = 0
    for _, row in briefs.iterrows():
        brief_id = row["brief_id"]
        for model in MODELS:
            if (brief_id, model) in done:
                continue
            img = image_path(model, brief_id)
            if not img.exists():
                skipped_no_image += 1
                continue
            jobs.append((brief_id, model, row.to_dict(), img))

    log.info(f"Phase 2A evaluation (Configuration B) — {len(jobs)} image(s) ready | "
             f"{len(done)} already evaluated | {skipped_no_image} not yet generated")

    write_header = not OUTPUT_FILE.exists()
    fh = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    success = 0
    failures = 0

    for i, (brief_id, model, brief_row, img) in enumerate(jobs, 1):
        brief = dict(brief_row)
        brief["prompt_text_en"] = brief["prompt_text"]  # Phase 2A: English-only, identical
        brief["image_path"]     = str(img)
        brief["model"]          = model
        brief["language"]       = LANGUAGE

        t0 = time.time()
        try:
            result  = app.invoke(brief)
            row_out = flatten_result(brief_id, model, result)
            writer.writerow(row_out)
            fh.flush()
            dur = round(time.time() - t0, 1)
            success += 1
            log.info(f"[{i}/{len(jobs)}] [OK] {brief_id} | {model} | "
                     f"compliance_score={result['compliance_score']} ({dur}s)")
        except Exception as e:
            failures += 1
            log.error(f"[{i}/{len(jobs)}] [FAIL] {brief_id} | {model} -> {e}")

    fh.close()

    log.info("=" * 60)
    log.info(f"Evaluation run complete — [OK] {success}  [FAIL] {failures}")
    log.info(f"Results file: {OUTPUT_FILE}")
    if skipped_no_image:
        log.info(f"{skipped_no_image} image(s) not yet generated — re-run this script "
                 f"once generation finishes to pick them up.")
    log.info("=" * 60)


if __name__ == "__main__":
    run()