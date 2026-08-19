"""
MulBrandEval — Phase 2B Ablation, Configuration C (Multilingual)
==================================================================
Configuration C — Flat pipeline (no short-circuit), DE/FR/ES

Runs the compiled FLAT LangGraph pipeline (phase2a_dag_pipeline.pipeline_flat.app)
on the same Phase 2B images already scored by Configuration B
(phase2b_evaluate.py), forcing every node to execute regardless of Node 1's
object-presence outcome. Writes one flattened row per image to:

    results/phase2b/configuration_c_flat_pipeline_multilingual.csv

WHY THIS RUN, SPECIFICALLY:
Configuration B's short-circuit rate turned out far higher for DE/FR/ES than
the exposé anticipated (FLUX short-circuit ~15% in English -> 48-61% in
DE/FR/ES; SD v1.5 ~69% -> 85-87%). Because Nodes 2-5 never execute on a
short-circuited image, Configuration B's data cannot test the exposé's own
stated RQ2 prediction that Node 2 (Spatial) and Node 3 (Attribute) would
show the largest CLCG values while Node 1 stayed "relatively stable" -- the
opposite of what was observed. Configuration C removes that masking effect:
every node runs on every image, so Nodes 2-5 get a real, non-confounded
per-language score even when Node 1 would have failed.

Same 900 images as Configuration B (150 briefs x 2 models x 3 languages,
DE/FR/ES only -- AR excluded from the multilingual scope by design, same as
phase2b_evaluate.py; see arabic_noncomparable_diagnostics.csv for the
separate 20-image AR diagnostic run). Not a new generation run -- this
re-evaluates the SAME images already on disk under a different pipeline
graph, so no new Replicate/API image-generation cost, only the GPT-4o mini
VQA calls for whichever nodes a given image hadn't reached under Configuration B.

Resume-safe: (brief_id, model, language) triples already present in the
output CSV are skipped. Image-existence-aware throughout.

Place this script at:  phase2b_multilingual/phase2b_configuration_c_evaluate.py
  (mirrors phase2b_evaluate.py's placement and import pattern exactly --
  sys.path insertion below makes phase2a_dag_pipeline resolve correctly
  from this folder, no -m invocation needed.)

Run from MulBrandEval/ root, same plain-script style as its siblings:
  # Thesis scope (default):
  python phase2b_multilingual\\phase2b_configuration_c_evaluate.py

  # Any subset, e.g. just one language to check first:
  python phase2b_multilingual\\phase2b_configuration_c_evaluate.py --languages de
"""

import sys
import csv
import time
import logging
import argparse
import gc
from pathlib import Path

# ── Make phase2a_dag_pipeline importable from this script's location ──────────
# Same rationale as phase2b_evaluate.py: this file lives in
# phase2b_multilingual/, one level below the project root, while
# pipeline_flat.app lives in phase2a_dag_pipeline/, also directly under the
# project root. Inserting the project root onto sys.path lets this run as a
# plain script, matching phase2b_generate.py / phase2b_evaluate.py exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from phase2a_dag_pipeline.pipeline_flat import app

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

BRIEFS_FILES = {
    "de": "data/prompts/mbb/mbb_briefs_de.csv",
    "fr": "data/prompts/mbb/mbb_briefs_fr.csv",
    "es": "data/prompts/mbb/mbb_briefs_es.csv",
}
ALL_LANGUAGES     = ["de", "fr", "es"]   # AR out of scope for Configuration C —
                                          # see module docstring; no ar entry in
                                          # BRIEFS_FILES, so --languages ar would
                                          # correctly fail validation, not silently
                                          # do nothing.
DEFAULT_LANGUAGES = ["de", "fr", "es"]
IMAGE_ROOT = Path("data/generated_images/phase2b")
MODELS     = ["sd15", "flux"]

OUTPUT_FILE = Path("results/phase2b/configuration_c_flat_pipeline_multilingual.csv")

# Same schema as multilingual_ablation_table.csv (Configuration B) so the
# two files can be joined/compared directly on (brief_id, model, language).
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


def already_evaluated() -> set:
    """Return the set of (brief_id, model, language) triples already in the output CSV."""
    if not OUTPUT_FILE.exists():
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    if df.empty:
        return set()
    return set(zip(df["brief_id"], df["model"], df["language"]))


def flatten_result(brief_id: str, model: str, language: str, result: dict) -> dict:
    """Flatten the flat pipeline's nested compliance_report into one CSV row."""
    r = result["compliance_report"]
    return {
        "brief_id":                 brief_id,
        "model":                    model,
        "language":                 language,
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


def parse_languages(arg_value: str) -> list:
    """Parse and validate a comma-separated --languages argument, same
    contract as phase2b_evaluate.py's parser."""
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
            f"Valid options are: {', '.join(ALL_LANGUAGES)}. "
            f"(Arabic is intentionally out of scope for Configuration C — "
            f"see module docstring.)"
        )
    return [l for l in ALL_LANGUAGES if l in requested]

# ── Main loop ──────────────────────────────────────────────────────────────────

def run(languages: list):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    done = already_evaluated()

    jobs = []
    skipped_no_image = 0
    for lang in languages:
        briefs = load_briefs(lang)
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

    log.info(f"Phase 2B Configuration C (flat, languages={','.join(languages)}) — "
             f"{len(jobs)} image(s) ready | {len(done)} already evaluated | "
             f"{skipped_no_image} not yet generated")

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

    for i, (brief_id, model, lang, brief_row, img) in enumerate(jobs, 1):
        brief = dict(brief_row)
        # prompt_text and prompt_text_en both already present from Phase 2B
        # translation step — no override needed, same as phase2b_evaluate.py.
        brief["image_path"] = str(img)
        brief["model"]      = model
        brief["language"]   = lang

        t0 = time.time()
        try:
            result  = app.invoke(brief)
            row_out = flatten_result(brief_id, model, lang, result)
            writer.writerow(row_out)
            fh.flush()
            dur = round(time.time() - t0, 1)
            success += 1
            log.info(f"[{i}/{len(jobs)}] [OK] {brief_id} | {model} | {lang} | "
                     f"compliance_score={result['compliance_score']} ({dur}s)")
        except Exception as e:
            failures += 1
            log.error(f"[{i}/{len(jobs)}] [FAIL] {brief_id} | {model} | {lang} -> {e}")

        # Periodic garbage collection — same mitigation as phase2b_evaluate.py
        # for long-run memory growth.
        if i % 20 == 0:
            gc.collect()

    fh.close()

    log.info("=" * 60)
    log.info(f"Configuration C evaluation complete — [OK] {success}  [FAIL] {failures}")
    log.info(f"Results file: {OUTPUT_FILE}")
    if skipped_no_image:
        log.info(f"{skipped_no_image} image(s) not yet generated — re-run this script "
                 f"once generation finishes to pick them up.")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2B Configuration C — flat pipeline, multilingual")
    parser.add_argument(
        "--languages", type=str, default=",".join(DEFAULT_LANGUAGES),
        help=(
            "Comma-separated language codes to evaluate, any subset of "
            f"{{{', '.join(ALL_LANGUAGES)}}}. Defaults to de,fr,es. "
            "Arabic is intentionally not a valid option for Configuration C."
        ),
    )
    args = parser.parse_args()
    run(parse_languages(args.languages))