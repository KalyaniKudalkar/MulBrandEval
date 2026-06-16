"""
MulBrandEval — Phase 1A Image Generation Script
================================================
Generates all 1,000 images for Phase 1A:
  50 prompts × 5 languages × 2 seeds × 2 models = 1,000 images

Models:
  - Stable Diffusion v1.5  →  stability-ai/stable-diffusion (Replicate)
  - FLUX.1-dev             →  black-forest-labs/flux-dev    (Replicate)

Output folder structure:
  data/generated_images/phase1a/
    sd15/
      en/  de/  fr/  es/  ar/
    flux/
      en/  de/  fr/  es/  ar/

Each image is saved as:  {prompt_id}_seed{seed}.png
Resume:  Already-saved images are skipped automatically.
Log:     results/phase1a/generation_log.csv  (one row per image)

Place this script at:  phase1a_realism_baseline/phase1a_generate.py
Run from MulBrandEval/ root:  python phase1a_realism_baseline/phase1a_generate.py
"""

import os
import csv
import time
import logging
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

SEEDS           = [42, 123]
INFERENCE_STEPS = 50
CFG_SCALE       = 7.5   # guidance_scale for SD v1.5 / guidance for FLUX
IMAGE_SIZE_SD   = 512   # SD v1.5 native resolution
IMAGE_SIZE_FLUX = 1024  # FLUX.1-dev native resolution

MODELS = {
    "sd15": "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    "flux": "black-forest-labs/flux-dev",
}

# prompt file paths and the column that holds the text to send to the model
LANGUAGES = {
    "en": ("data/prompts/en/coco_prompts_en.csv", "caption"),
    "de": ("data/prompts/de/coco_prompts_de.csv", "caption_translated"),
    "fr": ("data/prompts/fr/coco_prompts_fr.csv", "caption_translated"),
    "es": ("data/prompts/es/coco_prompts_es.csv", "caption_translated"),
    "ar": ("data/prompts/ar/coco_prompts_ar.csv", "caption_translated"),
}

# Images land here — phase-namespaced so Phase 1B / 2A / 2B never collide
OUTPUT_ROOT = Path("data/generated_images/phase1a")

# Generation log lands in the results folder alongside future metric tables
LOG_FILE   = Path("results/phase1a/generation_log.csv")
LOG_FIELDS = ["timestamp", "model", "language", "prompt_id",
              "seed", "status", "output_path", "error", "duration_s"]

# ── Logging setup ──────────────────────────────────────────────────────────────
# File handler is added inside run() after make_dirs() guarantees the folder exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Directory helpers ──────────────────────────────────────────────────────────

def make_dirs():
    """Create all output and results folders if they don't already exist."""
    for model_key in MODELS:
        for lang in LANGUAGES:
            (OUTPUT_ROOT / model_key / lang).mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  # results/phase1a/


def image_path(model_key: str, lang: str, prompt_id: str, seed: int) -> Path:
    return OUTPUT_ROOT / model_key / lang / f"{prompt_id}_seed{seed}.png"


def already_done(model_key: str, lang: str, prompt_id: str, seed: int) -> bool:
    return image_path(model_key, lang, prompt_id, seed).exists()

# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_prompts(lang: str) -> pd.DataFrame:
    """
    Load the prompt CSV for a given language and normalise the caption column
    to 'prompt' so the rest of the script is language-agnostic.
    """
    csv_path, caption_col = LANGUAGES[lang]
    df = pd.read_csv(csv_path)
    df = df.rename(columns={caption_col: "prompt"})
    assert "prompt_id" in df.columns, f"Missing prompt_id column in {csv_path}"
    assert "prompt"    in df.columns, f"Missing caption column '{caption_col}' in {csv_path}"
    return df[["prompt_id", "prompt", "category"]].copy()

# ── Image saving ───────────────────────────────────────────────────────────────

def save_image_from_url(url: str, dest: Path):
    """Download image from a Replicate output URL and save as PNG."""
    response = httpx.get(url, timeout=60)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img.save(dest, format="PNG")


def save_image_from_fileobj(fileobj, dest: Path):
    """Save image from a Replicate FileOutput object."""
    content = fileobj.read()
    img = Image.open(BytesIO(content)).convert("RGB")
    img.save(dest, format="PNG")


def save_output(output, dest: Path):
    """
    Handle all three Replicate output types gracefully:
      - URL string   →  download via httpx
      - FileOutput   →  read bytes directly
      - object.url   →  download via httpx
    """
    if isinstance(output, str) and output.startswith("http"):
        save_image_from_url(output, dest)
    elif hasattr(output, "read"):
        save_image_from_fileobj(output, dest)
    elif hasattr(output, "url"):
        save_image_from_url(output.url, dest)
    else:
        save_image_from_url(str(output), dest)

# ── Generation functions ───────────────────────────────────────────────────────

def generate_sd15(prompt: str, seed: int):
    """Run SD v1.5 on Replicate."""
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
    """Run FLUX.1-dev on Replicate."""
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

def run():
    make_dirs()

    # Now that results/phase1a/ exists, attach the file log handler
    file_handler = logging.FileHandler(LOG_FILE.parent / "generation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
    log.addHandler(file_handler)

    # ── Count progress so the tqdm bar starts at the right position ────────────
    total   = len(MODELS) * len(LANGUAGES) * 50 * len(SEEDS)   # always 1,000
    skipped = sum(
        already_done(mk, lang, f"P{i+1:03d}", s)
        for mk in MODELS
        for lang in LANGUAGES
        for i in range(50)
        for s in SEEDS
    )
    log.info(f"Phase 1A — {total} total images | "
             f"{skipped} already done | {total - skipped} to generate")

    # ── Open (or append to) the generation log ─────────────────────────────────
    log_exists = LOG_FILE.exists()
    log_fh     = open(LOG_FILE, "a", newline="", encoding="utf-8")
    writer     = csv.DictWriter(log_fh, fieldnames=LOG_FIELDS)
    if not log_exists:
        writer.writeheader()

    # ── Build flat job list ────────────────────────────────────────────────────
    # Order: sd15 before flux, en first, seed 42 before 123
    jobs = [
        (model_key, lang, row, seed)
        for model_key in ["sd15", "flux"]
        for lang in ["en", "de", "fr", "es", "ar"]
        for _, row in load_prompts(lang).iterrows()
        for seed in SEEDS
    ]

    # ── Generation loop ────────────────────────────────────────────────────────
    with tqdm(total=total, initial=skipped, unit="img",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                         "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for model_key, lang, row, seed in jobs:
            prompt_id = row["prompt_id"]
            prompt    = row["prompt"]
            dest      = image_path(model_key, lang, prompt_id, seed)

            # Resume: skip if this image already exists on disk
            if dest.exists():
                continue

            pbar.set_description(
                f"{model_key.upper():6s} | {lang} | {prompt_id} | seed={seed}"
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
                    "prompt_id":   prompt_id,
                    "seed":        seed,
                    "status":      "SUCCESS",
                    "output_path": str(dest),
                    "error":       "",
                    "duration_s":  duration,
                })
                log.info(f"[OK] {model_key} | {lang} | {prompt_id} | seed={seed} -> saved in {duration}s")

            except replicate.exceptions.ReplicateError as e:
                duration = round(time.time() - t0, 1)
                log.error(f"[FAIL] Replicate error | {model_key} | {lang} | "
                            f"{prompt_id} | seed={seed} -> {e}")
                writer.writerow({
                    "timestamp":   datetime.utcnow().isoformat(),
                    "model":       model_key,
                    "language":    lang,
                    "prompt_id":   prompt_id,
                    "seed":        seed,
                    "status":      "FAIL",
                    "output_path": "",
                    "error":       str(e),
                    "duration_s":  duration,
                })
                time.sleep(5)   # brief pause before next call on API error

            except Exception as e:
                duration = round(time.time() - t0, 1)
                log.error(f"[FAIL] Unexpected error | {model_key} | {lang} | "
                            f"{prompt_id} | seed={seed} -> {e}")
                writer.writerow({
                    "timestamp":   datetime.utcnow().isoformat(),
                    "model":       model_key,
                    "language":    lang,
                    "prompt_id":   prompt_id,
                    "seed":        seed,
                    "status":      "FAIL",
                    "output_path": "",
                    "error":       str(e),
                    "duration_s":  duration,
                })
                time.sleep(5)

            finally:
                log_fh.flush()   # write each row immediately so no data lost on crash
                pbar.update(1)

    log_fh.close()

    # ── Final summary ──────────────────────────────────────────────────────────
    log_df   = pd.read_csv(LOG_FILE)
    success  = (log_df["status"] == "SUCCESS").sum()
    failures = (log_df["status"] == "FAIL").sum()
    log.info("=" * 60)
    log.info(f"Phase 1A complete -- [OK] {success} generated  [FAIL] {failures} failed")
    if failures > 0:
        log.info("Failed images (re-run this script to retry them automatically):")
        failed = log_df[log_df["status"] == "FAIL"][
            ["model", "language", "prompt_id", "seed", "error"]]
        log.info("\n" + failed.to_string(index=False))
    log.info("=" * 60)
    log.info(f"Images saved to : {OUTPUT_ROOT}/")
    log.info(f"Log saved to    : {LOG_FILE}")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise EnvironmentError(
            "REPLICATE_API_TOKEN not found.\n"
            "Check that your .env file contains:  REPLICATE_API_TOKEN=r8_...\n"
            "and that .env is in the MulBrandEval/ root directory.\n"
            "Run this script from the root:  "
            "python phase1a_realism_baseline/phase1a_generate.py"
        )
    run()