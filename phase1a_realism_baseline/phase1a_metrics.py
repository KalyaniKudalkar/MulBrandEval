"""
MulBrandEval — Phase 1A Metric Computation Script
==================================================
Computes FID and CMMD for all model × language conditions.
Produces the Phase 1A CLCG baseline table.

Inputs:
  data/generated_images/phase1a/{model}/{lang}/*.png
  data/reference_images/coco_phase1a/*.jpg   (downloaded here if missing)

Outputs:
  results/phase1a/fid_cmmd_scores.csv        raw scores per model × language
  results/phase1a/clcg_phase1a.csv           CLCG table (difference from EN)
  results/phase1a/phase1a_summary.txt        human-readable summary

Place at:  phase1a_realism_baseline/phase1a_metrics.py
Run from MulBrandEval/ root:
  python phase1a_realism_baseline/phase1a_metrics.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fix Windows OpenMP conflict
import sys
import logging
import requests
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

GENERATED_ROOT = Path("data/generated_images/phase1a")
REFERENCE_DIR  = Path("data/reference_images/coco_phase1a")
RESULTS_DIR    = Path("results/phase1a")
PROMPTS_CSV    = Path("data/prompts/en/coco_prompts_en.csv")
CMMD_MODULE    = Path("metrics/cmmd.py")

MODELS    = ["sd15", "flux"]
LANGUAGES = ["en", "de", "fr", "es", "ar"]
COCO_URL  = "http://images.cocodataset.org/val2017/{image_id:012d}.jpg"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Setup ──────────────────────────────────────────────────────────────────────

def make_dirs():
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Download MS-COCO reference images ──────────────────────────────────

def download_reference_images():
    """
    Download the 50 MS-COCO val2017 images corresponding to the Phase 1A prompts.
    These form the real-image reference distribution for FID and CMMD.
    Skips images already on disk.
    """
    df = pd.read_csv(PROMPTS_CSV)
    image_ids = df["image_id"].tolist()

    already = list(REFERENCE_DIR.glob("*.jpg"))
    if len(already) == len(image_ids):
        log.info(f"Reference images already present ({len(already)} images). Skipping download.")
        return

    log.info(f"Downloading {len(image_ids)} MS-COCO reference images...")
    failed = []
    for img_id in tqdm(image_ids, desc="Downloading COCO images"):
        dest = REFERENCE_DIR / f"{img_id:012d}.jpg"
        if dest.exists():
            continue
        url = COCO_URL.format(image_id=img_id)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
        except Exception as e:
            log.warning(f"Failed to download image {img_id}: {e}")
            failed.append(img_id)

    downloaded = len(list(REFERENCE_DIR.glob("*.jpg")))
    log.info(f"Reference images ready: {downloaded}/{len(image_ids)}")
    if failed:
        log.warning(f"Failed downloads: {failed}")

# ── Step 2: FID computation ────────────────────────────────────────────────────

def compute_fid(real_dir: Path, gen_dir: Path) -> float:
    """
    Compute FID using torchmetrics (Windows-safe, no multiprocessing workers).
    Loads images one-by-one to avoid DataLoader issues on Windows.
    Returns the FID score (lower = better realism).
    """
    try:
        import torch
        import torchvision.transforms as T
        from torchmetrics.image.fid import FrechetInceptionDistance

        transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
        ])

        fid = FrechetInceptionDistance(feature=2048, normalize=True)
        fid.eval()

        # Load real images
        real_paths = [p for p in real_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for p in tqdm(real_paths, desc="FID real images", leave=False):
            img = Image.open(p).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            fid.update(tensor, real=True)

        # Load generated images
        gen_paths = [p for p in gen_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for p in tqdm(gen_paths, desc="FID gen images", leave=False):
            img = Image.open(p).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            fid.update(tensor, real=False)

        score = float(fid.compute())
        return round(score, 4)

    except Exception as e:
        log.error(f"FID computation failed: {e}")
        return float("nan")

# ── Step 3: CMMD computation ───────────────────────────────────────────────────

def get_image_paths(folder: Path) -> list:
    """Return sorted list of all PNG/JPG file path strings in a folder."""
    paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        paths.extend(sorted(folder.glob(ext)))
    return [str(p) for p in sorted(set(paths))]


def compute_cmmd(real_dir: Path, gen_dir: Path) -> float:
    """
    Compute CMMD using metrics/cmmd.py (Jayasumana et al. 2024).
    Passes file path lists directly — matches the compute_cmmd(real_paths, gen_paths)
    signature in the project implementation.
    Falls back to RBF-MMD on pixel features if the module is unavailable.
    """
    if CMMD_MODULE.exists():
        try:
            import importlib.util
            spec   = importlib.util.spec_from_file_location("cmmd", str(CMMD_MODULE))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            real_paths = get_image_paths(real_dir)
            gen_paths  = get_image_paths(gen_dir)

            if not real_paths or not gen_paths:
                log.warning(f"Empty image list: real={len(real_paths)}, gen={len(gen_paths)}")
                return float("nan")

            score = module.compute_cmmd(real_paths, gen_paths)
            return round(float(score), 6)

        except Exception as e:
            log.warning(f"cmmd.py load failed ({e}). Returning nan.")
            return float("nan")

    log.warning("metrics/cmmd.py not found. Returning nan for CMMD.")
    return float("nan")

# ── Step 4: Run all computations ───────────────────────────────────────────────

def run_all_metrics() -> pd.DataFrame:
    rows = []
    total = len(MODELS) * len(LANGUAGES)
    done  = 0

    for model in MODELS:
        for lang in LANGUAGES:
            done += 1
            gen_dir = GENERATED_ROOT / model / lang
            n_imgs  = len(list(gen_dir.glob("*.png")))

            log.info(f"[{done}/{total}] Computing FID + CMMD | "
                     f"model={model} | lang={lang} | n={n_imgs} images")

            fid_score  = compute_fid(REFERENCE_DIR, gen_dir)
            cmmd_score = compute_cmmd(REFERENCE_DIR, gen_dir)

            rows.append({
                "model":    model,
                "language": lang,
                "n_images": n_imgs,
                "fid":      fid_score,
                "cmmd":     cmmd_score,
            })
            log.info(f"    FID={fid_score:.4f}  CMMD={cmmd_score:.6f}")

    return pd.DataFrame(rows)

# ── Step 5: Compute CLCG table ─────────────────────────────────────────────────

def compute_clcg(df: pd.DataFrame) -> pd.DataFrame:
    """
    CLCG(L, m, M) = Score(EN, m, M) - Score(L, m, M)
    Positive CLCG = non-English performs WORSE than English (gap exists).
    For FID and CMMD, lower score = better, so positive CLCG = degradation.
    """
    rows = []
    for model in MODELS:
        en_row  = df[(df["model"] == model) & (df["language"] == "en")].iloc[0]
        en_fid  = en_row["fid"]
        en_cmmd = en_row["cmmd"]

        for lang in ["de", "fr", "es", "ar"]:
            lang_row  = df[(df["model"] == model) & (df["language"] == lang)].iloc[0]
            fid_clcg  = round(lang_row["fid"]  - en_fid,  4)
            cmmd_clcg = round(lang_row["cmmd"] - en_cmmd, 6)
            rows.append({
                "model":          model,
                "language":       lang,
                "fid_en":         en_fid,
                "fid_lang":       lang_row["fid"],
                "clcg_fid":       fid_clcg,
                "cmmd_en":        en_cmmd,
                "cmmd_lang":      lang_row["cmmd"],
                "clcg_cmmd":      cmmd_clcg,
            })
    return pd.DataFrame(rows)

# ── Step 6: Save outputs ───────────────────────────────────────────────────────

def save_outputs(scores_df: pd.DataFrame, clcg_df: pd.DataFrame):
    scores_path = RESULTS_DIR / "fid_cmmd_scores.csv"
    clcg_path   = RESULTS_DIR / "clcg_phase1a.csv"
    summary_path = RESULTS_DIR / "phase1a_summary.txt"

    scores_df.to_csv(scores_path, index=False)
    clcg_df.to_csv(clcg_path, index=False)
    log.info(f"Scores saved to : {scores_path}")
    log.info(f"CLCG saved to   : {clcg_path}")

    # Human-readable summary
    lines = []
    lines.append("MulBrandEval — Phase 1A Results")
    lines.append("=" * 60)
    lines.append("\nRAW SCORES (FID / CMMD) per model × language")
    lines.append("-" * 60)
    lines.append(scores_df.to_string(index=False))
    lines.append("\nCLCG TABLE (non-English vs English baseline)")
    lines.append("Positive = non-English performs WORSE than English")
    lines.append("-" * 60)
    lines.append(clcg_df[["model","language","clcg_fid","clcg_cmmd"]].to_string(index=False))

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Summary saved to: {summary_path}")

    # Print to terminal
    print("\n" + "=" * 60)
    print("PHASE 1A COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    print("\nRaw scores:")
    print(scores_df.to_string(index=False))
    print("\nCLCG table (cross-lingual compliance gap):")
    print(clcg_df[["model","language","clcg_fid","clcg_cmmd"]].to_string(index=False))
    print("=" * 60)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Phase 1A Metric Computation — FID + CMMD")
    log.info(f"Generated images : {GENERATED_ROOT}")
    log.info(f"Reference images : {REFERENCE_DIR}")
    log.info(f"Results          : {RESULTS_DIR}")

    make_dirs()

    # Step 1 — download reference images
    download_reference_images()

    # Step 2 — compute all FID + CMMD scores
    scores_df = run_all_metrics()

    # Step 3 — compute CLCG table
    clcg_df = compute_clcg(scores_df)

    # Step 4 — save everything
    save_outputs(scores_df, clcg_df)

    log.info("Phase 1A metric computation complete.")