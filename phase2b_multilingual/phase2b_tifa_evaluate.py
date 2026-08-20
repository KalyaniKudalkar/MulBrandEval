"""
MulBrandEval — Phase 2B TIFA Baseline, Multilingual (DE/FR/ES + optional AR)
==============================================================================
TIFA-style VQA faithfulness baseline (Hu et al., 2023-inspired — see
tifa_baseline.py's module docstring for the full independent-implementation
justification, exposé §5.3) extended to Phase 2B's multilingual scope.

Same TIFA logic as the Phase 2A English baseline: decompose each brief into
discrete yes/no questions (required_objects, colour_attributes,
required_text) and ask a VQA model every question independently — NO
dependency structure, no short-circuiting. This is what makes TIFA a clean
convergent-validity check against Configuration B/C: if TIFA (zero
dependency logic) still shows the same DE/FR/ES CLCG pattern as your DAG,
that's independent evidence the gap is a real content-generation effect,
not an artifact of your own pipeline's node-ordering or short-circuit design.

DE/FR/ES rows (the thesis's primary CLCG-eligible dataset) go to:
    results/phase2b/tifa_multilingual.csv

AR rows go to a SEPARATE file, same isolation convention as
phase2b_evaluate.py / arabic_noncomparable_diagnostics.csv:
    results/phase2b/tifa_arabic_pilot.csv

This split is deliberate, not cosmetic. TIFA questions on Arabic-prompted
images ask a VQA model about images the CLIP text encoder never
meaningfully understood the prompt for in the first place — the resulting
tifa_score is diagnostic evidence for the case-study section (does the
failure persist even with zero dependency structure?), not a valid
CLCG measurement. AR rows must never end up in the same table used for
RQ2/RQ3/RQ4 statistical testing.

--languages controls which language(s) to run, mirroring phase2b_evaluate.py
exactly. Defaults to de,fr,es (thesis scope, 900 images, run separately from
the AR pilot). Pass --languages ar to run just the 20 Arabic pilot images
(10 briefs x 2 models) as case-study diagnostics. Image-existence-aware
throughout, so passing ar simply picks up whichever pilot images already
exist on disk.

Resume-safe: (brief_id, model, language) TRIPLES already present in the
relevant output CSV are skipped.

Place this script at:  phase2b_multilingual/phase2b_tifa_evaluate.py

Run from MulBrandEval/ root, as TWO SEPARATE RUNS (not combined in one
call) — this keeps the long DE/FR/ES run and the quick AR pilot run
independent and separately resumable/restartable:

  # Thesis scope — DE/FR/ES, 900 images, ~2-4 hours:
  python phase2b_multilingual\\phase2b_tifa_evaluate.py

  # Arabic pilot — 20 images, diagnostic only, ~2-3 minutes:
  python phase2b_multilingual\\phase2b_tifa_evaluate.py --languages ar
"""

import base64
import csv
import json
import logging
import os
import argparse
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

BRIEFS_FILES = {
    "de": "data/prompts/mbb/mbb_briefs_de.csv",
    "fr": "data/prompts/mbb/mbb_briefs_fr.csv",
    "es": "data/prompts/mbb/mbb_briefs_es.csv",
    "ar": "data/prompts/mbb/mbb_briefs_ar.csv",
}
ALL_LANGUAGES     = ["de", "fr", "es", "ar"]   # canonical order, matches phase2b_evaluate.py
DEFAULT_LANGUAGES = ["de", "fr", "es"]          # thesis scope — CLCG-eligible languages
IMAGE_ROOT = Path("data/generated_images/phase2b")
MODELS     = ["sd15", "flux"]
SEED       = 42   # matches Phase 2B generation convention

MAIN_OUTPUT_FILE = Path("results/phase2b/tifa_multilingual.csv")     # DE/FR/ES only
AR_OUTPUT_FILE    = Path("results/phase2b/tifa_arabic_pilot.csv")    # AR only, never merged


def output_file_for(language: str) -> Path:
    """Route each language's rows to the correct output file. AR is
    permanently isolated from the CLCG-eligible table — see module docstring."""
    return AR_OUTPUT_FILE if language == "ar" else MAIN_OUTPUT_FILE


OUTPUT_FIELDS = [
    "brief_id", "model", "language",
    "n_questions", "n_yes", "tifa_score",
    "per_object_json", "per_colour_json", "per_text_json",
]

VQA_MODEL = "gpt-4o-mini"   # same as node1_object_presence.py / node3_colour_attribute.py / tifa_baseline.py

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
    Same call parameters as node1_object_presence.py / node3_colour_attribute.py
    / tifa_baseline.py (Phase 2A English version).
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


# ── Element parsing (mirrors tifa_baseline.py / node0_prompt_parser.py) ────────

def _parse_json_field(value, field_name: str):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse field '{field_name}': {e}\nValue: {value!r}")


# ── Image path / brief loading ─────────────────────────────────────────────────

def image_path(model: str, language: str, brief_id: str, seed: int = SEED) -> Path:
    return IMAGE_ROOT / model / language / f"{brief_id}_seed{seed}.png"


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


def parse_languages(arg_value: str) -> list:
    """Parse and validate a comma-separated --languages argument, same
    contract as phase2b_evaluate.py's parser (canonical order, dedup)."""
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


# ── Per-image TIFA evaluation ──────────────────────────────────────────────────

def evaluate_image(brief_row: pd.Series, img_path: Path) -> dict:
    """
    Ask every decomposed question for this brief against this image.
    No short-circuiting: object, colour, and text questions are ALL
    asked regardless of any answer — including asking a colour/text
    question about an object that a prior question said is absent.
    Identical logic to tifa_baseline.py (Phase 2A English version).
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


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(languages: list):
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not found.\n"
            "Check that your .env file contains it and that python-dotenv "
            "has been loaded (see phase2a_generate.py for the load_dotenv() pattern)."
        )

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
                jobs.append((brief_id, model, lang, row, img))

    log.info(f"Phase 2B TIFA baseline (languages={','.join(languages)}) — "
             f"{len(jobs)} image(s) ready | "
             f"{len(done_main) + len(done_ar)} already evaluated | "
             f"{skipped_no_image} not yet generated")

    if not jobs:
        log.info("Nothing to do.")
        return

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

    n_ok, n_fail = 0, 0
    for i, (brief_id, model, lang, brief_row, img_path) in enumerate(jobs, 1):
        target = output_file_for(lang)
        try:
            result = evaluate_image(brief_row, img_path)
            writers[target].writerow({
                "brief_id": brief_id,
                "model":    model,
                "language": lang,
                **result,
            })
            handles[target].flush()
            n_ok += 1
            log.info(f"[{i}/{len(jobs)}] [OK] {brief_id} | {model} | {lang} | "
                      f"tifa_score={result['tifa_score']:.4f} "
                      f"({result['n_yes']}/{result['n_questions']})"
                      + ("  [case-study, not CLCG-eligible]" if lang == "ar" else ""))
        except Exception as e:
            n_fail += 1
            log.error(f"[{i}/{len(jobs)}] [FAIL] {brief_id} | {model} | {lang} -> {e}")

    for fh in handles.values():
        fh.close()

    log.info("=" * 60)
    log.info(f"TIFA baseline run complete — [OK] {n_ok}  [FAIL] {n_fail}")
    for target in handles:
        log.info(f"Results file: {target}")
    if skipped_no_image:
        log.info(f"{skipped_no_image} image(s) not yet generated — re-run this script "
                 f"once generation finishes to pick them up.")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2B multilingual TIFA baseline")
    parser.add_argument(
        "--languages", type=str, default=",".join(DEFAULT_LANGUAGES),
        help=(
            "Comma-separated language codes to evaluate, any subset of "
            f"{{{', '.join(ALL_LANGUAGES)}}}. Defaults to de,fr,es (thesis scope). "
            "Pass 'ar' to run the 20 Arabic pilot images as case-study "
            "diagnostics — written to a separate output file, never merged "
            "into the CLCG-eligible table. Run de,fr,es and ar as SEPARATE "
            "invocations, not combined."
        ),
    )
    args = parser.parse_args()
    run(parse_languages(args.languages))