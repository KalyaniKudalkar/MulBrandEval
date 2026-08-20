"""
MulBrandEval — Phase 2B Ablation, Configuration C — Arabic Pilot (Diagnostic)
==============================================================================
Configuration C — Flat pipeline (no short-circuit), Arabic pilot (20 images)

Runs the compiled FLAT LangGraph pipeline (phase2a_dag_pipeline.pipeline_flat.app)
on the same 20 Arabic pilot images already scored by Configuration B
(phase2b_evaluate.py --languages ar -> arabic_noncomparable_diagnostics.csv),
forcing every node to execute regardless of Node 1's object-presence outcome.

WHY THIS RUN, SPECIFICALLY:
Configuration B's short-circuit masking means Nodes 2-5 never ran on most of
the Arabic pilot images once Node 1 failed. This run removes that masking so
every node produces a real score even when Node 1 failed — giving a fuller
diagnostic picture for the Arabic case-study section: does the categorical
failure persist through every node once short-circuiting is removed, or does
any node get lucky (e.g. Node 5's PickScore/quality check, which isn't
checking brief-specific content)?

NOT CLCG-ELIGIBLE. Same reasoning as arabic_noncomparable_diagnostics.csv:
the CLIP text encoder never meaningfully received the Arabic prompt, so any
score here is encoder noise, not a validated compliance measurement. This
output is diagnostic-only, for the case-study write-up — it must NEVER be
merged into configuration_c_flat_pipeline_multilingual.csv or any CLCG
heatmap/statistical-testing table. Hence a fully separate output file below.

Scope: only the 10 pilot briefs (MB004, MB024, MB038, MB039, MB061, MB084,
MB094, MB111, MB125, MB128) x 2 models = 20 images, same set as the
Configuration B Arabic pilot. mbb_briefs_ar.csv has all 150 brief rows, but
since only these 10 have generated images on disk, the existing
image-existence-aware skip logic naturally scopes the run to the 20 pilot
images — no hardcoded brief-id list needed here.

Not a new generation run — re-evaluates the SAME 20 images already on disk
under the flat pipeline graph. No new Replicate cost; only GPT-4o mini VQA
calls for whichever nodes a given image hadn't reached under Configuration B.
Estimated: ~20 images through Nodes 1-5, ~10-15 min wall-clock, pennies in
OpenAI cost.

Resume-safe: (brief_id, model, language) triples already present in the
output CSV are skipped. Image-existence-aware throughout.

Place this script at:  phase2b_multilingual/phase2b_configuration_c_evaluate_arabic_pilot.py
  (mirrors phase2b_configuration_c_evaluate.py's placement and import
  pattern exactly — sys.path insertion below makes phase2a_dag_pipeline
  resolve correctly from this folder, no -m invocation needed.)

Run from MulBrandEval/ root, same plain-script style as its siblings:
  python phase2b_multilingual\\phase2b_configuration_c_evaluate_arabic_pilot.py
"""

import sys
import csv
import time
import logging
import gc
from pathlib import Path

# ── Make phase2a_dag_pipeline importable from this script's location ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from phase2a_dag_pipeline.pipeline_flat import app

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

BRIEFS_FILE = "data/prompts/mbb/mbb_briefs_ar.csv"
LANGUAGE    = "ar"
IMAGE_ROOT  = Path("data/generated_images/phase2b")
MODELS      = ["sd15", "flux"]

# Diagnostic-only — deliberately NOT named to match the main Config C file,
# and never written into it. Keeps this permanently isolated from the
# CLCG-eligible tables, same convention as arabic_noncomparable_diagnostics.csv.
OUTPUT_FILE = Path("results/phase2b/configuration_c_arabic_pilot.csv")

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
    assert n == 150, (
        f"Expected 150 briefs, found {n} in {BRIEFS_FILE} — check the file."
    )
    return df


def already_evaluated() -> set:
    """Return the set of (brief_id, model, language) triples already in the output CSV."""
    if not OUTPUT_FILE.exists():
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    if df.empty:
        return set()
    return set(zip(df["brief_id"], df["model"], df["language"]))


def flatten_result(brief_id: str, model: str, result: dict) -> dict:
    """Flatten the flat pipeline's nested compliance_report into one CSV row."""
    r = result["compliance_report"]
    return {
        "brief_id":                 brief_id,
        "model":                    model,
        "language":                 LANGUAGE,
        "node1_pass":               r["node1"]["pass"],
        "node1_score":              r["node1"]["score"],
        "node1_short_circuit":      r["node1"]["short_circuit"],  # informational only —
                                                                    # Nodes 2-5 still ran
                                                                    # regardless of this value
        "node2_pass":               r["node2"]["pass"],
        "node2_score":              r["node2"]["score"],
        "node2_skipped":            r["node2"]["skipped"],   # always False here
        "node3_pass":               r["node3"]["pass"],
        "node3_score":              r["node3"]["score"],
        "node3_skipped":            r["node3"]["skipped"],   # always False here
        "node4_pass":               r["node4"]["pass"],
        "node4_score":              r["node4"]["score"],
        "node4_skipped":            r["node4"]["skipped"],   # still True for sd15
                                                                # (architectural, not short-circuit)
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

    done = already_evaluated()
    briefs = load_briefs()

    jobs = []
    skipped_no_image = 0
    for _, row in briefs.iterrows():
        brief_id = row["brief_id"]
        for model in MODELS:
            if (brief_id, model, LANGUAGE) in done:
                continue
            img = image_path(model, brief_id)
            if not img.exists():
                # Expected for 140/150 briefs — only the 10-brief pilot set
                # has generated Arabic images on disk. Not an error.
                skipped_no_image += 1
                continue
            jobs.append((brief_id, model, row.to_dict(), img))

    log.info(f"Phase 2B Configuration C — Arabic pilot (diagnostic-only) — "
             f"{len(jobs)} image(s) ready | {len(done)} already evaluated")
    if len(jobs) + len(done) != 20:
        log.warning(f"Expected 20 total pilot images (10 briefs x 2 models), "
                     f"found {len(jobs) + len(done)}. Check image directory "
                     f"if this looks wrong.")

    if not jobs:
        log.info("Nothing to do.")
        return

    write_header = not OUTPUT_FILE.exists()
    fh = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    success = 0
    failures = 0

    for i, (brief_id, model, brief_row, img) in enumerate(jobs, 1):
        brief = dict(brief_row)
        brief["image_path"] = str(img)
        brief["model"]      = model
        brief["language"]   = LANGUAGE

        t0 = time.time()
        try:
            result  = app.invoke(brief)
            row_out = flatten_result(brief_id, model, result)
            writer.writerow(row_out)
            fh.flush()
            dur = round(time.time() - t0, 1)
            success += 1
            log.info(f"[{i}/{len(jobs)}] [OK] {brief_id} | {model} | ar | "
                     f"compliance_score={result['compliance_score']} ({dur}s) "
                     f"[case-study, not CLCG-eligible]")
        except Exception as e:
            failures += 1
            log.error(f"[{i}/{len(jobs)}] [FAIL] {brief_id} | {model} -> {e}")

        if i % 20 == 0:
            gc.collect()

    fh.close()

    log.info("=" * 60)
    log.info(f"Configuration C Arabic pilot complete — [OK] {success}  [FAIL] {failures}")
    log.info(f"Results file: {OUTPUT_FILE}  (diagnostic-only — do not merge into "
             f"configuration_c_flat_pipeline_multilingual.csv)")
    log.info("=" * 60)


if __name__ == "__main__":
    run()