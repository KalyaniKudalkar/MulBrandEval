"""
MulBrandEval — Phase 1B Image Generation Script
================================================
Generates all 1,000 images for Phase 1B:
  100 prompts x 5 languages x 1 seed x 2 models = 1,000 images

Seed decision: seed 42 only (seed 123 dropped from Phase 1B onwards).
Confirmed in Post-Meeting Summary (26 May 2026): per-image metrics in Phase 1B
(CLIP-Score, mCLIP, Vendi Score) are deterministic scoring functions applied to
fixed generated images. A second seed adds no statistical value here.

Models:
  - Stable Diffusion v1.5  ->  stability-ai/stable-diffusion (Replicate)
  - FLUX.1-dev             ->  black-forest-labs/flux-dev    (Replicate)

Output folder structure:
  data/generated_images/phase1b/
    sd15/
      en/  de/  fr/  es/  ar/
    flux/
      en/  de/  fr/  es/  ar/

Each image is saved as:  {prompt_id}_seed{seed}.png
  e.g.  DB001_seed42.png   (DrawBench prompt)
        GB025_seed42.png   (GenAI-Bench prompt)

Resume:  Already-saved images are skipped automatically — safe to re-run after
         any crash or interruption without re-generating existing images.
Log:     results/phase1b/generation_log.csv  (one row per image attempt)

Place this script at:  phase1b_alignment_baseline/phase1b_generate.py
Run from MulBrandEval/ root:
  python phase1b_alignment_baseline/phase1b_generate.py
"""

import csv
import logging
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd
import replicate
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Load all API keys from .env at MulBrandEval/ root
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

SEEDS           = [42]          # seed 123 dropped from Phase 1B onwards
INFERENCE_STEPS = 50
CFG_SCALE       = 7.5           # guidance_scale for SD v1.5 / guidance for FLUX
IMAGE_SIZE_SD   = 512           # SD v1.5 native resolution
IMAGE_SIZE_FLUX = 1024          # FLUX.1-dev native resolution

MODELS = {
    "sd15": "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    "flux": "black-forest-labs/flux-dev",
}

# Prompt CSV paths and the column that holds the text to send to the model.
# English uses 'caption'; all translated files use 'caption_translated'.
LANGUAGES = {
    "en": ("data/prompts/en/phase1b_prompts_en.csv", "caption"),
    "de": ("data/prompts/de/phase1b_prompts_de.csv", "caption_translated"),
    "fr": ("data/prompts/fr/phase1b_prompts_fr.csv", "caption_translated"),
    "es": ("data/prompts/es/phase1b_prompts_es.csv", "caption_translated"),
    "ar": ("data/prompts/ar/phase1b_prompts_ar.csv", "caption_translated"),
}

# Phase-namespaced output root — never collides with Phase 1A, 2A, or 2B
OUTPUT_ROOT = Path("data/generated_images/phase1b")

# Log lives alongside future Phase 1B metric tables under results/phase1b/
LOG_FILE   = Path("results/phase1b/generation_log.csv")
LOG_FIELDS = ["timestamp", "model", "language", "prompt_id",
              "seed", "status", "output_path", "error", "duration_s"]

# ── Logging setup ──────────────────────────────────────────────────────────────
# File handler is added inside run() after make_dirs() guarantees the folder exists.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Directory helpers ──────────────────────────────────────────────────────────

def make_dirs():
    """Create all output and results folders if they do not already exist."""
    for model_key in MODELS:
        for lang in LANGUAGES:
            (OUTPUT_ROOT / model_key / lang).mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def image_path(model_key: str, lang: str, prompt_id: str, seed: int) -> Path:
    return OUTPUT_ROOT / model_key / lang / f"{prompt_id}_seed{seed}.png"


def already_done(model_key: str, lang: str, prompt_id: str, seed: int) -> bool:
    return image_path(model_key, lang, prompt_id, seed).exists()

# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_prompts(lang: str) -> pd.DataFrame:
    """
    Load the Phase 1B prompt CSV for a given language and normalise the caption
    column to 'prompt' so the rest of the script is language-agnostic.

    Phase 1B CSVs contain extra columns (dag_node, source, human_score) that are
    not needed at generation time — only prompt_id, prompt, and category are kept.
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
      - URL string   ->  download via httpx
      - FileOutput   ->  read bytes directly  (has .read())
      - object.url   ->  download via httpx
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
    """Run SD v1.5 on Replicate. Parameter name: guidance_scale."""
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
    """Run FLUX.1-dev on Replicate. Parameter name: guidance (not guidance_scale)."""
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

    # Attach file log handler now that results/phase1b/ is guaranteed to exist
    file_handler = logging.FileHandler(
        LOG_FILE.parent / "generation.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    )
    log.addHandler(file_handler)

    # Load all prompt DataFrames once upfront (avoids re-reading CSV per inner loop)
    prompts   = {lang: load_prompts(lang) for lang in LANGUAGES}
    n_prompts = len(prompts["en"])  # 100

    # ── Count progress so the tqdm bar starts at the right position ────────────
    # Prompt IDs are identical across all language CSVs — use English as reference.
    prompt_ids = prompts["en"]["prompt_id"].tolist()

    total   = len(MODELS) * len(LANGUAGES) * n_prompts * len(SEEDS)  # 1,000
    skipped = sum(
        already_done(mk, lang, pid, s)
        for mk  in MODELS
        for lang in LANGUAGES
        for pid  in prompt_ids
        for s    in SEEDS
    )
    log.info(
        f"Phase 1B — {total} total images | "
        f"{skipped} already done | {total - skipped} to generate"
    )

    # ── Open (or append to) the generation log ─────────────────────────────────
    log_exists = LOG_FILE.exists()
    log_fh     = open(LOG_FILE, "a", newline="", encoding="utf-8")
    writer     = csv.DictWriter(log_fh, fieldnames=LOG_FIELDS)
    if not log_exists:
        writer.writeheader()

    generate_fn = {"sd15": generate_sd15, "flux": generate_flux}

    # ── Generation loop ────────────────────────────────────────────────────────
    # Outer: model -> language -> prompt row -> seed
    # SD v1.5 runs first (cheaper) to confirm the pipeline works before FLUX spend.
    with tqdm(total=total - skipped, desc="Phase 1B generation") as pbar:
        for model_key in MODELS:
            for lang in LANGUAGES:
                df = prompts[lang]

                for _, row in df.iterrows():
                    prompt_id = row["prompt_id"]
                    prompt    = row["prompt"]

                    for seed in SEEDS:
                        dest = image_path(model_key, lang, prompt_id, seed)
                        if dest.exists():
                            continue  # resume — skip already-generated images

                        t_start                      = time.time()
                        status, err_msg, out_path    = "FAIL", "", ""

                        try:
                            output = generate_fn[model_key](prompt, seed)

                            if output is None:
                                raise ValueError(
                                    "Replicate returned None — possible NSFW flag"
                                )

                            save_output(output, dest)
                            status   = "SUCCESS"
                            out_path = str(dest)

                        except Exception as e:
                            err_str = str(e)
                            if "nsfw" in err_str.lower() or "safety" in err_str.lower():
                                err_msg = f"NSFW flag: {err_str[:300]}"
                                log.warning(
                                    f"NSFW detected — "
                                    f"{model_key}/{lang}/{prompt_id}/seed={seed}"
                                )
                            else:
                                err_msg = f"Error: {err_str[:300]}"
                                log.error(
                                    f"FAILED — "
                                    f"{model_key}/{lang}/{prompt_id}/seed={seed} "
                                    f"| {err_str[:200]}"
                                )

                        duration = round(time.time() - t_start, 2)
                        writer.writerow({
                            "timestamp":   datetime.utcnow().isoformat(),
                            "model":       model_key,
                            "language":    lang,
                            "prompt_id":   prompt_id,
                            "seed":        seed,
                            "status":      status,
                            "output_path": out_path,
                            "error":       err_msg,
                            "duration_s":  duration,
                        })
                        log_fh.flush()  # write immediately so no rows are lost on crash

                        if status == "SUCCESS":
                            pbar.update(1)
                            log.info(
                                f"OK  {model_key}/{lang}/{prompt_id}/seed={seed}"
                                f"  ({duration}s)"
                            )

    log_fh.close()
    log.info("Phase 1B image generation complete.")
    log.info(f"Generation log : {LOG_FILE}")
    log.info(f"Images saved to: {OUTPUT_ROOT}")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("MulBrandEval — Phase 1B Image Generation")
    log.info(f"Models     : {list(MODELS.keys())}")
    log.info(f"Languages  : {list(LANGUAGES.keys())}")
    log.info(f"Seeds      : {SEEDS}")
    log.info(f"Output root: {OUTPUT_ROOT}")
    log.info(f"Log file   : {LOG_FILE}")
    run()