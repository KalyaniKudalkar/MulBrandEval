"""
MulBrandEval — Phase 2B Image Generation Script
================================================
Generates images for Phase 2B (multilingual extension):
  150 MBB briefs x up to 4 languages (DE, FR, ES, AR) x 1 seed (42) x 2 models.

  Full capability (default, all 4 languages) = 1,200 images.
  This thesis's actual full-scale run is scoped to DE/FR/ES only (900 images)
  via --languages de,fr,es -- see "Language scoping" below for why.

Two run modes (choose via --mode):
  pilot  ->  10 pre-selected briefs (2 per industry, seeded random,
             see PILOT_BRIEF_IDS below) x [selected languages] x 2 models.
             Use this first to validate the generation pipeline (API
             handling, encoding, NSFW-filter behaviour) before committing
             to a full run.
  full   ->  all 150 briefs x [selected languages] x 2 models.
             Pilot images are NOT regenerated -- they are picked up as
             already-done via the shared resume log/output folder, so the
             pilot is the first slice of the full run, not a separate
             throwaway batch.

Language scoping (--languages):
  This script supports all four Phase 2B languages (DE, FR, ES, AR) by
  default -- nothing about Arabic generation has been removed or disabled.
  `--languages ar` (alone or combined with others) reproduces the Arabic
  pilot exactly and can be used to extend it to a full 150-brief Arabic run.

  This thesis's full-scale invocation is scoped to DE/FR/ES only:
      python phase2b_multilingual\\phase2b_generate.py --mode full --languages de,fr,es

  This is a SCOPING decision, not a capability limitation. The 80-image
  pilot (10 briefs x 4 languages x 2 models) showed Arabic generation
  fails completely and consistently across both models -- confirmed via
  isolated Replicate playground testing and a token-overflow analysis
  showing the CLIP text encoder cannot process Arabic script (root cause,
  not a pipeline bug; consistent with the same pattern in Phase 1A/1B).
  Arabic evidence for the thesis is drawn from the existing pilot images,
  the playground control test, and the token analysis -- not from a full
  150-brief Arabic generation run. See the Phase 2B methodology deviation
  note for full justification.

Models:
  - Stable Diffusion v1.5  ->  stability-ai/stable-diffusion (Replicate)
  - FLUX.1-dev             ->  black-forest-labs/flux-dev    (Replicate)

Prompt source:
  data/prompts/mbb/mbb_briefs_{de,fr,es,ar}.csv
  Uses the `prompt_text` column (the DeepL-translated, native-language
  string) as the literal generation prompt -- this is deliberate: Phase 2B
  tests what SD v1.5 / FLUX produce FROM the translated prompt, which is
  the whole point of the multilingual evaluation. `prompt_text_en` is kept
  in the CSV for Node 5 (PickScore) only and is NOT touched here.
  `required_text`, `required_objects`, `colour_attributes`,
  `spatial_constraints`, `style_descriptors` are structured fields for the
  DAG evaluation nodes later -- they are NOT touched here either.

Output folder structure:
  data/generated_images/phase2b/
    sd15/
      de/  fr/  es/  (ar/ only if --languages includes ar)
    flux/
      de/  fr/  es/  (ar/ only if --languages includes ar)
  Only folders for the languages actually generated in a given run are
  created, so an unscoped DE/FR/ES run does not leave behind an empty
  ar/ folder that could be mistaken for a stalled Arabic run.

Each image is saved as:  {brief_id}_seed{seed}.png
Resume:  Already-saved images are skipped automatically (same check used
         by both pilot and full mode -- this is what makes pilot images
         "reused, not skipped-and-discarded" when full mode runs).
Log:     results/phase2b/generation_log.csv  (one row per image, same
         schema as Phase 2A's log for consistency across phases -- pilot
         and full runs, and any language subset, share this single log
         file).

Place this script at:  phase2b_multilingual/phase2b_generate.py
Run from MulBrandEval/ root:
  # Pilot, all 4 languages (default):
  python phase2b_multilingual\\phase2b_generate.py --mode pilot

  # Full run, thesis scope (DE/FR/ES only):
  python phase2b_multilingual\\phase2b_generate.py --mode full --languages de,fr,es

  # Full run, all 4 languages (e.g. to reproduce/extend the Arabic pilot):
  python phase2b_multilingual\\phase2b_generate.py --mode full --languages de,fr,es,ar

  # Arabic only (reproduces the pilot's Arabic slice, extendable to full 150):
  python phase2b_multilingual\\phase2b_generate.py --mode full --languages ar
"""

import os
import csv
import time
import logging
import argparse
import pandas as pd
import replicate
from pathlib import Path
from PIL import Image
from io import BytesIO
import httpx
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

# Load all API keys from .env at MulBrandEval/ root
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

SEEDS           = [42]          # Same single-seed convention as Phase 2A
INFERENCE_STEPS = 50
CFG_SCALE       = 7.5
IMAGE_SIZE_SD   = 512
IMAGE_SIZE_FLUX = 1024

MODELS = {
    "sd15": "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    "flux": "black-forest-labs/flux-dev",
}

# Canonical list of ALL languages this script supports. This does NOT change
# based on what any given run scopes to via --languages -- it is the fixed
# validation/default reference and the fixed ordering used everywhere.
LANGUAGES = ["de", "fr", "es", "ar"]

BRIEFS_FILES = {
    "de": "data/prompts/mbb/mbb_briefs_de.csv",
    "fr": "data/prompts/mbb/mbb_briefs_fr.csv",
    "es": "data/prompts/mbb/mbb_briefs_es.csv",
    "ar": "data/prompts/mbb/mbb_briefs_ar.csv",
}

# 10 briefs, 2 per industry, seeded random (random.seed(42) over
# mbb_briefs_en_v3_FINAL.csv, sorted by industry then brief_id) — chosen to
# stress-test generation MECHANICS (API handling, encoding, NSFW-filter
# behaviour) across a spread of industries, deliberately NOT weighted by
# translation cc_score/cometkiwi_score status, since that is a separate
# concern already closed out in the translation review commit (8966df8).
PILOT_BRIEF_IDS = [
    "MB004", "MB024",   # tech_product
    "MB038", "MB039",   # food_beverage
    "MB061", "MB084",   # fashion_apparel
    "MB094", "MB111",   # automotive
    "MB125", "MB128",   # lifestyle_wellness
]

# Images land here — phase-namespaced so Phase 1A/1B/2A never collide
OUTPUT_ROOT = Path("data/generated_images/phase2b")

# Single shared log for pilot AND full mode, and for any language subset —
# this is what makes pilot images (and any earlier language-subset run)
# count toward later runs instead of being thrown away.
LOG_FILE   = Path("results/phase2b/generation_log.csv")
LOG_FIELDS = ["timestamp", "model", "language", "prompt_id",
              "seed", "status", "output_path", "error", "duration_s"]

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Language selection ─────────────────────────────────────────────────────────

def parse_languages(arg_value: str) -> list:
    """
    Parse and validate a comma-separated --languages argument.
    Returns the requested subset in canonical DE/FR/ES/AR order, regardless
    of the order the user typed them in, so downstream job ordering stays
    deterministic and consistent with prior single-language-set runs.
    """
    requested = [x.strip().lower() for x in arg_value.split(",") if x.strip()]
    if not requested:
        raise argparse.ArgumentTypeError(
            "--languages received an empty value. "
            f"Valid options are: {', '.join(LANGUAGES)} (comma-separated, any subset)."
        )
    invalid = [x for x in requested if x not in LANGUAGES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown language code(s): {invalid}. "
            f"Valid options are: {', '.join(LANGUAGES)}."
        )
    # Preserve canonical ordering regardless of input order/duplicates
    return [l for l in LANGUAGES if l in requested]

# ── Directory helpers ──────────────────────────────────────────────────────────

def make_dirs(languages: list):
    """
    Create output and results folders for this run's selected languages only.
    A DE/FR/ES-only run will not create an empty ar/ folder -- that would
    look like a stalled/failed Arabic run rather than "never attempted".
    """
    for model_key in MODELS:
        for lang in languages:
            (OUTPUT_ROOT / model_key / lang).mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  # results/phase2b/


def image_path(model_key: str, lang: str, brief_id: str, seed: int) -> Path:
    return OUTPUT_ROOT / model_key / lang / f"{brief_id}_seed{seed}.png"


def already_done(model_key: str, lang: str, brief_id: str, seed: int) -> bool:
    return image_path(model_key, lang, brief_id, seed).exists()

# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_briefs(lang: str) -> pd.DataFrame:
    """
    Load one language's translated MBB brief CSV and normalise to
    brief_id/prompt/industry, mirroring Phase 2A's structure.
    """
    path = BRIEFS_FILES[lang]
    df = pd.read_csv(path)
    assert "brief_id"    in df.columns, f"Missing brief_id column in {path}"
    assert "prompt_text" in df.columns, f"Missing prompt_text column in {path}"
    df = df.rename(columns={"prompt_text": "prompt"})
    n = len(df)
    assert n == 150, f"Expected 150 briefs, found {n} in {path} — check the file before running."
    return df[["brief_id", "prompt", "industry"]].copy()

# ── Image saving (identical to Phase 2A) ────────────────────────────────────────

DOWNLOAD_RETRIES = 4
DOWNLOAD_TIMEOUT = 90
DOWNLOAD_BACKOFF  = [3, 8, 15]

def save_image_from_url(url: str, dest: Path):
    """
    Download image from a Replicate output URL and save as PNG.
    Retries the DOWNLOAD only (not the generation) on timeout/connection
    errors, since by this point the image already exists on Replicate's
    CDN and has already been billed.
    """
    last_error = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            response = httpx.get(url, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(dest, format="PNG")
            return
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_error = e
            if attempt < DOWNLOAD_RETRIES:
                wait = DOWNLOAD_BACKOFF[attempt - 1]
                log.warning(f"[RETRY {attempt}/{DOWNLOAD_RETRIES - 1}] download timed out, "
                            f"retrying in {wait}s -> {url}")
                time.sleep(wait)
    raise last_error


def save_image_from_fileobj(fileobj, dest: Path):
    content = fileobj.read()
    img = Image.open(BytesIO(content)).convert("RGB")
    img.save(dest, format="PNG")


def save_output(output, dest: Path):
    if isinstance(output, str) and output.startswith("http"):
        save_image_from_url(output, dest)
    elif hasattr(output, "read"):
        save_image_from_fileobj(output, dest)
    elif hasattr(output, "url"):
        save_image_from_url(output.url, dest)
    else:
        save_image_from_url(str(output), dest)

# ── Generation functions (identical to Phase 2A) ────────────────────────────────

def generate_sd15(prompt: str, seed: int):
    output = replicate.run(
        MODELS["sd15"],
        input={
            "prompt":              prompt,
            "num_inference_steps": INFERENCE_STEPS,
            "guidance_scale":      CFG_SCALE,
            "seed":                seed,
            "width":               IMAGE_SIZE_SD,
            "height":              IMAGE_SIZE_SD,
            "num_outputs":         1,
        },
    )
    return output[0] if isinstance(output, list) else output


def generate_flux(prompt: str, seed: int):
    output = replicate.run(
        MODELS["flux"],
        input={
            "prompt":              prompt,
            "num_inference_steps": INFERENCE_STEPS,
            "guidance":            CFG_SCALE,
            "seed":                seed,
            "width":               IMAGE_SIZE_FLUX,
            "height":              IMAGE_SIZE_FLUX,
            "num_outputs":         1,
            "output_format":       "png",
        },
    )
    return output[0] if isinstance(output, list) else output

# ── Main loop ──────────────────────────────────────────────────────────────────

def run(mode: str, languages: list):
    make_dirs(languages)

    file_handler = logging.FileHandler(LOG_FILE.parent / "generation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    log.addHandler(file_handler)

    lang_label = "/".join(l.upper() for l in languages)

    # ── Load and filter briefs per selected language ────────────────────────
    briefs_by_lang = {}
    for lang in languages:
        df = load_briefs(lang)
        if mode == "pilot":
            df = df[df["brief_id"].isin(PILOT_BRIEF_IDS)].copy()
            assert len(df) == len(PILOT_BRIEF_IDS), (
                f"Expected {len(PILOT_BRIEF_IDS)} pilot briefs in {lang}, found {len(df)} — "
                f"check PILOT_BRIEF_IDS against {BRIEFS_FILES[lang]}."
            )
        briefs_by_lang[lang] = df

    n_briefs_per_lang = len(briefs_by_lang[languages[0]])
    total = len(MODELS) * len(languages) * n_briefs_per_lang * len(SEEDS)

    skipped = sum(
        already_done(mk, lang, row["brief_id"], s)
        for mk in MODELS
        for lang in languages
        for _, row in briefs_by_lang[lang].iterrows()
        for s in SEEDS
    )
    log.info(f"Phase 2B [{mode.upper()}] — languages: {lang_label} — {total} total images | "
             f"{skipped} already done | {total - skipped} to generate")

    log_exists = LOG_FILE.exists()
    log_fh     = open(LOG_FILE, "a", newline="", encoding="utf-8")
    writer     = csv.DictWriter(log_fh, fieldnames=LOG_FIELDS)
    if not log_exists:
        writer.writeheader()

    # ── Build flat job list ──────────────────────────────────────────────────
    # Order: sd15 before flux, languages in canonical DE/FR/ES/AR order
    # restricted to this run's selection, briefs in file order within each
    # language, seed 42 (only seed).
    jobs = [
        (model_key, lang, row, seed)
        for model_key in ["sd15", "flux"]
        for lang in languages
        for _, row in briefs_by_lang[lang].iterrows()
        for seed in SEEDS
    ]

    with tqdm(total=total, initial=skipped, unit="img",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for model_key, lang, row, seed in jobs:
            brief_id = row["brief_id"]
            prompt   = row["prompt"]
            dest     = image_path(model_key, lang, brief_id, seed)

            if dest.exists():
                continue

            pbar.set_description(
                f"{model_key.upper():6s} | {lang} | {brief_id} | seed={seed}"
            )

            t0 = time.time()
            try:
                if model_key == "sd15":
                    output = generate_sd15(prompt, seed)
                else:
                    output = generate_flux(prompt, seed)

                save_output(output, dest)
                duration = round(time.time() - t0, 1)

                writer.writerow({
                    "timestamp":   datetime.utcnow().isoformat(),
                    "model":       model_key,
                    "language":    lang,
                    "prompt_id":   brief_id,
                    "seed":        seed,
                    "status":      "SUCCESS",
                    "output_path": str(dest),
                    "error":       "",
                    "duration_s":  duration,
                })
                log.info(f"[OK] {model_key} | {lang} | {brief_id} | seed={seed} -> saved in {duration}s")

            except replicate.exceptions.ReplicateError as e:
                duration = round(time.time() - t0, 1)
                log.error(f"[FAIL] Replicate error | {model_key} | {lang} | "
                            f"{brief_id} | seed={seed} -> {e}")
                writer.writerow({
                    "timestamp":   datetime.utcnow().isoformat(),
                    "model":       model_key,
                    "language":    lang,
                    "prompt_id":   brief_id,
                    "seed":        seed,
                    "status":      "FAIL",
                    "output_path": "",
                    "error":       str(e),
                    "duration_s":  duration,
                })
                time.sleep(5)

            except Exception as e:
                duration = round(time.time() - t0, 1)
                log.error(f"[FAIL] Unexpected error | {model_key} | {lang} | "
                            f"{brief_id} | seed={seed} -> {e}")
                writer.writerow({
                    "timestamp":   datetime.utcnow().isoformat(),
                    "model":       model_key,
                    "language":    lang,
                    "prompt_id":   brief_id,
                    "seed":        seed,
                    "status":      "FAIL",
                    "output_path": "",
                    "error":       str(e),
                    "duration_s":  duration,
                })
                time.sleep(5)

            finally:
                log_fh.flush()
                pbar.update(1)

    log_fh.close()

    # ── Final summary ──────────────────────────────────────────────────────
    log_df = pd.read_csv(LOG_FILE)
    # Restrict the summary to this phase's rows only (in case the log file
    # is ever shared/appended across runs — safe even though currently
    # Phase 2B has its own dedicated log file).
    success  = (log_df["status"] == "SUCCESS").sum()
    failures = (log_df["status"] == "FAIL").sum()
    log.info("=" * 60)
    log.info(f"Phase 2B [{mode.upper()}] languages={lang_label} complete -- "
             f"[OK] {success} generated  [FAIL] {failures} failed (log totals across all runs so far)")
    if failures > 0:
        log.info("Failed images (re-run this script with the same --mode/--languages to retry them automatically):")
        failed = log_df[log_df["status"] == "FAIL"][
            ["model", "language", "prompt_id", "seed", "error"]]
        log.info("\n" + failed.to_string(index=False))
    log.info("=" * 60)
    log.info(f"Images saved to : {OUTPUT_ROOT}/")
    log.info(f"Log saved to    : {LOG_FILE}")
    if mode == "pilot":
        log.info("Pilot complete. Inspect the images, then run with --mode full "
                 "(same --languages, or a different subset) to generate the "
                 "remaining briefs (pilot images will be skipped, not regenerated).")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2B multilingual image generation")
    parser.add_argument(
        "--mode", choices=["pilot", "full"], required=True,
        help="pilot = 10 briefs x [languages] x 2 models; "
             "full = all 150 briefs x [languages] x 2 models"
    )
    parser.add_argument(
        "--languages", type=str, default=",".join(LANGUAGES),
        help=(
            "Comma-separated language codes to generate, any subset of "
            f"{{{', '.join(LANGUAGES)}}}. Defaults to all four. "
            "This thesis's full-scale run uses --languages de,fr,es "
            "(Arabic generation is scoped out per the documented pilot "
            "finding, not removed from the script -- pass --languages ar "
            "or include it in the list to reproduce or extend the Arabic "
            "pilot)."
        ),
    )
    args = parser.parse_args()

    languages = parse_languages(args.languages)

    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise EnvironmentError(
            "REPLICATE_API_TOKEN not found.\n"
            "Check that your .env file contains:  REPLICATE_API_TOKEN=r8_...\n"
            "and that .env is in the MulBrandEval/ root directory.\n"
            "Run this script from the root, e.g.:  "
            "python phase2b_multilingual\\phase2b_generate.py --mode pilot"
        )
    run(args.mode, languages)