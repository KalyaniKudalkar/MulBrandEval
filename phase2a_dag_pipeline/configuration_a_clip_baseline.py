"""
MulBrandEval — Phase 2A Ablation, Configuration A
===================================================
Configuration A — CLIP-Score-only monolithic baseline (no DAG)

Computes a single CLIP-Score per image against the full `prompt_text`
(same model and formula as Phase 1: openai/clip-vit-base-patch32,
100 * cosine_similarity(image_embedding, text_embedding), numerically
equivalent to torchmetrics.functional.multimodal.clip_score).

No GPT-4o mini calls, no OCR, no PickScore, no LangGraph — this is
the "what if we'd only used CLIP-Score, like most prior T2I eval work"
baseline that Configuration B (the full DAG) is compared against.

Output: results/phase2a/configuration_a_clip_baseline.csv
Columns: brief_id, model, language, clip_score

Resume-safe: (brief_id, model) pairs already in the output CSV are skipped.

Run from MulBrandEval/ root:  python -m phase2a_dag_pipeline.configuration_a_clip_baseline
"""

import csv
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ── Configuration ──────────────────────────────────────────────────────────────

BRIEFS_FILE = "data/prompts/mbb/mbb_briefs_en.csv"   # = mbb_briefs_en_v3_FINAL
IMAGE_ROOT  = Path("data/generated_images/phase2a")
LANGUAGE    = "en"
MODELS      = ["sd15", "flux"]

# Phase 2A is single-seed by design (seed 42 only — see phase2a_generate.py's
# SEEDS list). Kept as a named constant here, rather than hardcoded inline,
# so this script stays correct if it's ever reused for a multi-seed run.
SEED = 42

OUTPUT_FILE = Path("results/phase2a/configuration_a_clip_baseline.csv")
OUTPUT_FIELDS = ["brief_id", "model", "language", "clip_score"]

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"   # same as Phase 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


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


def get_clip_score(image: Image.Image, text: str, clip_model, clip_processor) -> float:
    """
    CLIP-Score = 100 * cosine_similarity(image_embedding, text_embedding).
    Identical formula to phase1b_metrics.py's get_clip_score_and_embedding,
    minus the embedding return (not needed here — no Vendi Score in Config A).
    """
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,  # CLIP token limit
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)

    img_emb = outputs.image_embeds
    txt_emb = outputs.text_embeds

    img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_norm = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    return 100.0 * (img_norm * txt_norm).sum().item()


def main():
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
            jobs.append((brief["brief_id"], model, brief["prompt_text"], img_path))

    log.info(f"Configuration A — {len(jobs)} image(s) to score | "
             f"{len(done)} already scored")

    if not jobs:
        log.info("Nothing to do.")
        return

    log.info(f"Loading CLIP model: {CLIP_MODEL_NAME} (device={DEVICE})")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_FILE.exists()
    fh = open(OUTPUT_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    n_ok, n_fail = 0, 0
    for i, (brief_id, model, prompt_text, img_path) in enumerate(jobs, 1):
        try:
            image = Image.open(img_path).convert("RGB")
            score = get_clip_score(image, prompt_text, clip_model, clip_processor)
            writer.writerow({
                "brief_id": brief_id,
                "model": model,
                "language": LANGUAGE,
                "clip_score": round(score, 4),
            })
            fh.flush()
            n_ok += 1
            log.info(f"[{i}/{len(jobs)}] [OK] {brief_id} | {model} | "
                      f"clip_score={score:.4f}")
        except Exception as e:
            n_fail += 1
            log.error(f"[{i}/{len(jobs)}] [FAIL] {brief_id} | {model} | error={e}")

    fh.close()
    log.info("=" * 60)
    log.info(f"Configuration A complete — [OK] {n_ok}  [FAIL] {n_fail}")
    log.info(f"Results file: {OUTPUT_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()