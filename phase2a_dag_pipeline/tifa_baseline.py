"""
MulBrandEval — Phase 2A External Baseline: TIFA-style VQA
============================================================
NOT part of the Configuration A/B/C ablation family (that trio is an
internal study of your own DAG's dependency structure — see
english_ablation_table.csv). This is a separate, external baseline
comparison against prior work, per exposé Section 5.3:

  "MulBrandEval is benchmarked against three prior evaluation approaches:
   (1) CLIP-Score baseline [= Configuration A]; (2) CMMD baseline
   [not feasible without a reference-image dataset — documented, not built];
   and (3) TIFA-style VQA faithfulness — decomposed question-answering
   protocol (Hu et al., 2023), implemented independently as a reference
   point for the DAG pipeline's improvement."

TIFA logic (Hu et al., 2023): decompose the prompt into discrete yes/no
questions (one per checkable element), ask a VQA model each question
independently, and average the answers into a single per-image
faithfulness score. Critically, TIFA has NO dependency structure —
every question is always asked, regardless of whether an upstream
question (e.g. "is the object present?") already failed. That is the
whole point of comparing it against Configuration B (the short-circuiting
DAG): TIFA will sometimes answer a colour or text question about an
object that isn't even in the image.

Element coverage (broader than the narrowest reading of the original
TIFA paper, matching what your own Nodes 1/3/4 check):
  - required_objects    -> "Is a {object} clearly visible in this image?"
  - colour_attributes    -> "Is the {object} {colour} in this image?"
  - required_text        -> "Does the image contain the text '{text}'?"

Question phrasing and VQA call parameters (model, detail=low, temperature=0,
max_tokens=5, yes/no parsing) are copied verbatim from node1_object_presence.py
and node3_colour_attribute.py so the TIFA baseline uses the exact same
underlying VQA capability as your DAG nodes — the only thing that differs
is the absence of dependency/short-circuit logic, isolating that one
variable for the comparison.

Per-image score = (number of "yes" answers) / (total questions asked).

Output: results/phase2a/tifa_baseline.csv
Columns: brief_id, model, language, n_questions, n_yes, tifa_score,
         per_object_json, per_colour_json, per_text_json

Resume-safe: (brief_id, model) pairs already in the output CSV are skipped.

Run from MulBrandEval/ root:  python -m phase2a_dag_pipeline.tifa_baseline
"""

import base64
import csv
import json
import logging
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load all API keys from .env at MulBrandEval/ root (same pattern as
# phase2a_generate.py) — this was missing before and caused the
# OPENAI_API_KEY not found error even with a correct .env file.
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

BRIEFS_FILE = "data/prompts/mbb/mbb_briefs_en.csv"   # = mbb_briefs_en_v3_FINAL
IMAGE_ROOT  = Path("data/generated_images/phase2a")
LANGUAGE    = "en"
MODELS      = ["sd15", "flux"]
SEED        = 42   # Phase 2A is single-seed by design (see phase2a_generate.py)

OUTPUT_FILE = Path("results/phase2a/tifa_baseline.csv")
OUTPUT_FIELDS = [
    "brief_id", "model", "language",
    "n_questions", "n_yes", "tifa_score",
    "per_object_json", "per_colour_json", "per_text_json",
]

VQA_MODEL = "gpt-4o-mini"   # same as node1_object_presence.py / node3_colour_attribute.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── OpenAI client singleton ───────────────────────────────────────────────────

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _CLIENT


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ask_yes_no(image_b64: str, question: str) -> bool:
    """
    Single decomposed VQA question, always asked regardless of any other
    question's answer — no short-circuit, no conditional skipping.
    Same call parameters as node1_object_presence.py / node3_colour_attribute.py.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=VQA_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/png;base64,{image_b64}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "text",
                        "text": f"{question} Answer only 'yes' or 'no'.",
                    },
                ],
            }
        ],
        max_tokens=5,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


# ── Element parsing (mirrors node0_prompt_parser.py's JSON field parsing) ──────

def _parse_json_field(value, field_name: str):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse field '{field_name}': {e}\nValue: {value!r}")


# ── Image path / brief loading ─────────────────────────────────────────────────

def image_path(model: str, brief_id: str, seed: int = SEED) -> Path:
    return IMAGE_ROOT / model / LANGUAGE / f"{brief_id}_seed{seed}.png"


def load_briefs() -> pd.DataFrame:
    df = pd.read_csv(BRIEFS_FILE)
    n = len(df)
    assert n == 150, f"Expected 150 briefs, found {n} in {BRIEFS_FILE} — check the file."
    return df


def already_evaluated() -> set:
    if not OUTPUT_FILE.exists():
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    if df.empty:
        return set()
    return set(zip(df["brief_id"], df["model"]))


# ── Per-image TIFA evaluation ──────────────────────────────────────────────────

def evaluate_image(brief_row: pd.Series, img_path: Path) -> dict:
    """
    Ask every decomposed question for this brief against this image.
    No short-circuiting: object, colour, and text questions are ALL
    asked regardless of any answer — including asking a colour/text
    question about an object that a prior question said is absent.
    """
    required_objects  = _parse_json_field(brief_row["required_objects"],  "required_objects")
    colour_attributes = _parse_json_field(brief_row["colour_attributes"], "colour_attributes")
    required_text     = _parse_json_field(brief_row["required_text"],     "required_text")

    image_b64 = _encode_image(img_path)

    per_object = {}
    for obj in required_objects:
        per_object[obj] = _ask_yes_no(
            image_b64, f"Is a {obj} clearly visible in this image?"
        )

    per_colour = {}
    for obj, colour in colour_attributes.items():
        per_colour[f"{obj}={colour}"] = _ask_yes_no(
            image_b64, f"Is the {obj} {colour} in this image?"
        )

    per_text = {}
    for text in required_text:
        per_text[text] = _ask_yes_no(
            image_b64, f"Does the image contain the text '{text}'?"
        )

    all_answers = list(per_object.values()) + list(per_colour.values()) + list(per_text.values())
    n_questions = len(all_answers)
    n_yes       = sum(all_answers)
    tifa_score  = round(n_yes / n_questions, 4) if n_questions else 0.0

    return {
        "n_questions":     n_questions,
        "n_yes":           n_yes,
        "tifa_score":      tifa_score,
        "per_object_json": json.dumps(per_object),
        "per_colour_json": json.dumps(per_colour),
        "per_text_json":   json.dumps(per_text),
    }


# ── Main loop ───────────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found.\n"
            "Check that your .env file contains it and that python-dotenv "
            "has been loaded (see phase2a_generate.py for the load_dotenv() pattern)."
        )

    briefs_df = load_briefs()
    done = already_evaluated()

    jobs = []
    for _, brief in briefs_df.iterrows():
        for model in MODELS:
            key = (brief["brief_id"], model)
            if key in done:
                continue
            img_path = image_path(model, brief["brief_id"])
            if not img_path.exists():
                continue
            jobs.append((brief, model, img_path))

    log.info(f"TIFA baseline — {len(jobs)} image(s) to evaluate | "
             f"{len(done)} already evaluated")

    if not jobs:
        log.info("Nothing to do.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_FILE.exists()
    fh = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    n_ok, n_fail = 0, 0
    for i, (brief, model, img_path) in enumerate(jobs, 1):
        brief_id = brief["brief_id"]
        try:
            result = evaluate_image(brief, img_path)
            writer.writerow({
                "brief_id": brief_id,
                "model":    model,
                "language": LANGUAGE,
                **result,
            })
            fh.flush()
            n_ok += 1
            log.info(f"[{i}/{len(jobs)}] [OK] {brief_id} | {model} | "
                      f"tifa_score={result['tifa_score']:.4f} "
                      f"({result['n_yes']}/{result['n_questions']})")
        except Exception as e:
            n_fail += 1
            log.error(f"[{i}/{len(jobs)}] [FAIL] {brief_id} | {model} | error={e}")

    fh.close()
    log.info("=" * 60)
    log.info(f"TIFA baseline complete — [OK] {n_ok}  [FAIL] {n_fail}")
    log.info(f"Results file: {OUTPUT_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()