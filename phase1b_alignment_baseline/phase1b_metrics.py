"""
MulBrandEval — Phase 1B Metric Computation Script
==================================================
Computes CLIP-Score, mCLIP Score, and Vendi Score for all Phase 1B images.
Produces the Phase 1B CLCG baseline table and Spearman rho table for RQ1.

Metrics:
  CLIP-Score  — openai/clip-vit-base-patch32, English caption always.
                Measures how well the generated image maps back to the English
                concept, regardless of which language was used for generation.
  mCLIP Score — clip-ViT-B-32-multilingual-v1, native-language caption.
                Direct multilingual comparison: same image, same concept,
                multilingual encoder instead of English-only CLIP.
  Vendi Score — Computed per (model x language) batch, NOT per image.
                Uses CLIP image embeddings reused from CLIP-Score computation
                (same feature space, no extra model load or forward pass).

CLCG sign convention:
  CLCG = Score(EN) - Score(Lang)  ->  positive = non-English performs worse.
  Consistent interpretation across all metrics (higher score = better for
  CLIP-Score, mCLIP, Vendi).

Outputs:
  results/phase1b/clip_mclip_vendi_scores.csv   mean scores per model x language
  results/phase1b/per_image_scores.csv          per-image CLIP and mCLIP scores
  results/phase1b/clcg_phase1b.csv              CLCG table
  results/phase1b/spearman_rq1.csv              Spearman rho vs GenAI-Bench human scores
  results/phase1b/phase1b_summary.txt           human-readable summary

Place at:  phase1b_alignment_baseline/phase1b_metrics.py
Run from MulBrandEval/ root:
  python phase1b_alignment_baseline/phase1b_metrics.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fix Windows OpenMP conflict
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr
import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer, util
from vendi_score import vendi
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS    = ["sd15", "flux"]
LANGUAGES = ["en", "de", "fr", "es", "ar"]
SEED      = 42

GENERATED_ROOT = Path("data/generated_images/phase1b")
RESULTS_DIR    = Path("results/phase1b")

# Prompt CSV paths and the column holding the text to score against
PROMPT_FILES = {
    "en": (Path("data/prompts/en/phase1b_prompts_en.csv"), "caption"),
    "de": (Path("data/prompts/de/phase1b_prompts_de.csv"), "caption_translated"),
    "fr": (Path("data/prompts/fr/phase1b_prompts_fr.csv"), "caption_translated"),
    "es": (Path("data/prompts/es/phase1b_prompts_es.csv"), "caption_translated"),
    "ar": (Path("data/prompts/ar/phase1b_prompts_ar.csv"), "caption_translated"),
}

CLIP_MODEL_NAME  = "openai/clip-vit-base-patch32"
MCLIP_MODEL_NAME = "clip-ViT-B-32-multilingual-v1"

# Use GPU if available — falls back to CPU transparently
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Directory and path helpers ─────────────────────────────────────────────────

def make_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def image_path(model: str, lang: str, prompt_id: str) -> Path:
    return GENERATED_ROOT / model / lang / f"{prompt_id}_seed{SEED}.png"

# ── Step 1: Load prompt CSVs ───────────────────────────────────────────────────

def load_all_prompts() -> tuple:
    """
    Returns:
      prompts_en       : DataFrame with all 100 English prompts
                         (prompt_id, caption, category, source, human_score)
      native_captions  : dict {lang -> {prompt_id -> native caption string}}
                         English maps to English captions for consistency.
    """
    prompts_en = pd.read_csv(PROMPT_FILES["en"][0])

    native_captions = {
        "en": dict(zip(prompts_en["prompt_id"], prompts_en["caption"]))
    }
    for lang in ["de", "fr", "es", "ar"]:
        csv_path, col = PROMPT_FILES[lang]
        df = pd.read_csv(csv_path)
        native_captions[lang] = dict(zip(df["prompt_id"], df[col]))

    return prompts_en, native_captions

# ── Step 2: Load models ────────────────────────────────────────────────────────

def load_models() -> tuple:
    """
    Loads CLIP (via transformers) and mCLIP (via sentence-transformers).
    CLIP is loaded directly to access image embeddings for Vendi Score reuse.
    """
    log.info(f"Loading CLIP model : {CLIP_MODEL_NAME}  (device={DEVICE})")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()

    log.info(f"Loading mCLIP model: {MCLIP_MODEL_NAME}")
    mclip_model = SentenceTransformer(MCLIP_MODEL_NAME)

    return clip_model, clip_processor, mclip_model

# ── Step 3: Per-image metric functions ────────────────────────────────────────

def get_clip_score_and_embedding(
    image: Image.Image,
    en_text: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> tuple:
    """
    Computes CLIP-Score and returns the normalized image embedding.

    CLIP-Score = 100 * cosine_similarity(image_embedding, text_embedding).
    This is numerically equivalent to torchmetrics.functional.multimodal.clip_score
    (same model, same formula) while also returning the image embedding
    for reuse in Vendi Score computation — avoiding a second forward pass.

    Args:
      image   : PIL Image (RGB)
      en_text : English caption (used regardless of generation language)

    Returns:
      score : float in [0, 100]
      emb   : np.ndarray of shape (512,), L2-normalised
    """
    inputs = clip_processor(
        text=[en_text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,          # CLIP token limit
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs  = clip_model(**inputs)

    img_emb = outputs.image_embeds                                  # (1, 512)
    txt_emb = outputs.text_embeds                                   # (1, 512)

    img_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_norm = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    score = 100.0 * (img_norm * txt_norm).sum().item()
    emb   = img_norm.squeeze().cpu().numpy()                        # (512,)

    return score, emb


def get_mclip_score(
    img_emb: np.ndarray,
    native_text: str,
    mclip_model: SentenceTransformer,
) -> float:
    """
    Computes mCLIP cosine similarity between image and native-language text.

    clip-ViT-B-32-multilingual-v1 uses the same CLIP ViT-B/32 image encoder
    as openai/clip-vit-base-patch32. The image embedding is therefore reused
    directly from get_clip_score_and_embedding — no PIL Image is passed to
    sentence-transformers, which avoids the PIL subscript error in ST 3.x.

    Only the text is encoded here via the multilingual text encoder.

    Args:
      img_emb     : L2-normalised CLIP image embedding (512,) — pre-computed
      native_text : caption in the generation language

    Returns:
      float in [-1, 1], typically [0, 1] for semantically related pairs
    """
    txt_emb = mclip_model.encode([native_text], normalize_embeddings=True)[0]
    return float(np.dot(img_emb, txt_emb))


def compute_vendi_score(embeddings: list) -> float:
    """
    Computes Vendi Score for a batch of images from their CLIP embeddings.

    Embeddings are already L2-normalised from get_clip_score_and_embedding.
    Gram matrix K[i,j] = dot(emb_i, emb_j) = cosine similarity (unit vectors).
    Vendi Score = exp(H(eigenvalues of K/n)) — effective number of distinct images.
    Higher score = more diverse batch.

    Args:
      embeddings : list of np.ndarray (512,), one per image in the batch

    Returns:
      float Vendi Score, or nan if batch is empty
    """
    if not embeddings:
        return float("nan")

    X = np.array(embeddings)            # (n, 512)
    K = X @ X.T                         # cosine similarity Gram matrix (n, n)
    K = np.clip(K, 0.0, 1.0)           # numerical stability — clamp rounding errors
    return float(vendi.score_K(K))

# ── Step 4: Main computation loop ─────────────────────────────────────────────

def run_all_metrics(
    prompts_en,
    native_captions,
    clip_model,
    clip_processor,
    mclip_model,
) -> tuple:
    """
    Iterates over all (model x language) conditions.
    For each image: computes CLIP-Score + embedding, then mCLIP score.
    After each batch: computes Vendi Score from cached embeddings.

    Returns:
      per_image_df : DataFrame, one row per image
      summary_df   : DataFrame, one row per (model x language)
    """
    per_image_rows = []
    summary_rows   = []
    total_batches  = len(MODELS) * len(LANGUAGES)
    done           = 0

    for model in MODELS:
        for lang in LANGUAGES:
            done    += 1
            n_prompts = len(prompts_en)
            log.info(
                f"[{done}/{total_batches}]  {model}/{lang}  —  {n_prompts} prompts"
            )

            clip_scores    = []
            mclip_scores   = []
            clip_embeddings = []

            for _, row in tqdm(
                prompts_en.iterrows(),
                total=n_prompts,
                desc=f"  {model}/{lang}",
                leave=False,
            ):
                prompt_id  = row["prompt_id"]
                en_caption = row["caption"]
                native_cap = native_captions[lang].get(prompt_id, en_caption)
                source     = row["source"]
                category   = row["category"]

                img_path = image_path(model, lang, prompt_id)

                # ── Handle missing image (NSFW or generation failure) ──────────
                if not img_path.exists():
                    log.warning(f"Missing image — {model}/{lang}/{prompt_id} — recording NaN")
                    per_image_rows.append({
                        "model": model, "language": lang,
                        "prompt_id": prompt_id, "seed": SEED,
                        "source": source, "category": category,
                        "clip_score":  float("nan"),
                        "mclip_score": float("nan"),
                    })
                    continue

                # ── Compute metrics ────────────────────────────────────────────
                try:
                    image = Image.open(img_path).convert("RGB")

                    # CLIP-Score + image embedding (one forward pass, reused for Vendi)
                    cs, emb = get_clip_score_and_embedding(
                        image, en_caption, clip_model, clip_processor
                    )

                    # mCLIP score — reuses CLIP image embedding, only encodes text via mCLIP
                    ms = get_mclip_score(emb, native_cap, mclip_model)

                    clip_scores.append(cs)
                    mclip_scores.append(ms)
                    clip_embeddings.append(emb)

                    per_image_rows.append({
                        "model": model, "language": lang,
                        "prompt_id": prompt_id, "seed": SEED,
                        "source": source, "category": category,
                        "clip_score":  round(cs, 6),
                        "mclip_score": round(ms, 6),
                    })

                except Exception as e:
                    log.error(f"Error on {model}/{lang}/{prompt_id}: {e}")
                    per_image_rows.append({
                        "model": model, "language": lang,
                        "prompt_id": prompt_id, "seed": SEED,
                        "source": source, "category": category,
                        "clip_score":  float("nan"),
                        "mclip_score": float("nan"),
                    })

            # ── Vendi Score for this batch ─────────────────────────────────────
            vs    = compute_vendi_score(clip_embeddings)
            n_ok  = len(clip_embeddings)
            mean_clip  = float(np.mean(clip_scores))  if clip_scores  else float("nan")
            mean_mclip = float(np.mean(mclip_scores)) if mclip_scores else float("nan")

            log.info(
                f"    n={n_ok}  "
                f"mean_clip={mean_clip:.4f}  "
                f"mean_mclip={mean_mclip:.4f}  "
                f"vendi={vs:.4f}"
            )

            summary_rows.append({
                "model":            model,
                "language":         lang,
                "n_images":         n_ok,
                "mean_clip_score":  round(mean_clip,  6),
                "mean_mclip_score": round(mean_mclip, 6),
                "vendi_score":      round(vs,          6),
            })

    return pd.DataFrame(per_image_rows), pd.DataFrame(summary_rows)

# ── Step 5: Compute CLCG table ─────────────────────────────────────────────────

def compute_clcg(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    CLCG(L, m, M) = Score(EN, m, M) - Score(L, m, M)

    Positive CLCG = non-English performs WORSE than English = degradation.
    For CLIP-Score, mCLIP, Vendi (all higher-is-better), subtracting lang
    from EN means positive = lower non-English score = degradation.
    Convention is consistent across all Phase 1B metrics.
    """
    rows = []
    for model in MODELS:
        en = summary_df[
            (summary_df["model"] == model) & (summary_df["language"] == "en")
        ].iloc[0]

        for lang in ["de", "fr", "es", "ar"]:
            lg = summary_df[
                (summary_df["model"] == model) & (summary_df["language"] == lang)
            ].iloc[0]

            rows.append({
                "model":       model,
                "language":    lang,
                "clip_en":     round(en["mean_clip_score"],  4),
                "clip_lang":   round(lg["mean_clip_score"],  4),
                "clcg_clip":   round(en["mean_clip_score"]  - lg["mean_clip_score"],  4),
                "mclip_en":    round(en["mean_mclip_score"], 4),
                "mclip_lang":  round(lg["mean_mclip_score"], 4),
                "clcg_mclip":  round(en["mean_mclip_score"] - lg["mean_mclip_score"], 4),
                "vendi_en":    round(en["vendi_score"],      4),
                "vendi_lang":  round(lg["vendi_score"],      4),
                "clcg_vendi":  round(en["vendi_score"]      - lg["vendi_score"],      4),
            })

    return pd.DataFrame(rows)

# ── Step 6: Compute Spearman rho (RQ1) ────────────────────────────────────────

def compute_spearman(
    per_image_df: pd.DataFrame,
    prompts_en: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes Spearman rho between automated metric scores and GenAI-Bench
    human alignment ratings. Answers RQ1: do automated metrics agree with
    human judgment about which prompts are harder to align?

    Computed for:
      - Both models (sd15, flux)
      - All 5 languages
      - Both metrics (clip_score, mclip_score)
      - GenAI-Bench prompts only (GB001-GB050, those with published human_score)

    The human_score is a 5-point alignment rating from GenAI-Bench
    (averaged across multiple human raters per prompt).
    """
    gb_df     = prompts_en[prompts_en["source"] == "GenAI-Bench"].copy()
    gb_ids    = gb_df["prompt_id"].tolist()
    human_map = dict(zip(gb_df["prompt_id"], gb_df["human_score"]))

    rows = []
    for model in MODELS:
        for lang in LANGUAGES:
            for metric in ["clip_score", "mclip_score"]:

                subset = per_image_df[
                    (per_image_df["model"]    == model) &
                    (per_image_df["language"] == lang)  &
                    (per_image_df["prompt_id"].isin(gb_ids))
                ].set_index("prompt_id")

                # Keep only prompts where both automated and human scores exist
                valid_ids = [
                    pid for pid in gb_ids
                    if pid in subset.index
                    and not pd.isna(subset.loc[pid, metric])
                    and pid in human_map
                    and not pd.isna(human_map[pid])
                ]

                if len(valid_ids) >= 2:
                    auto_scores   = [subset.loc[pid, metric] for pid in valid_ids]
                    human_scores  = [human_map[pid]          for pid in valid_ids]
                    rho, pval     = spearmanr(auto_scores, human_scores)
                else:
                    rho, pval = float("nan"), float("nan")

                rows.append({
                    "model":        model,
                    "language":     lang,
                    "metric":       metric,
                    "n_prompts":    len(valid_ids),
                    "spearman_rho": round(float(rho),  4),
                    "p_value":      round(float(pval), 4),
                })

    return pd.DataFrame(rows)

# ── Step 7: Save all outputs ───────────────────────────────────────────────────

def save_outputs(summary_df, per_image_df, clcg_df, spearman_df):
    scores_path   = RESULTS_DIR / "clip_mclip_vendi_scores.csv"
    detail_path   = RESULTS_DIR / "per_image_scores.csv"
    clcg_path     = RESULTS_DIR / "clcg_phase1b.csv"
    spearman_path = RESULTS_DIR / "spearman_rq1.csv"
    summary_path  = RESULTS_DIR / "phase1b_summary.txt"

    summary_df.to_csv(scores_path,   index=False)
    per_image_df.to_csv(detail_path, index=False)
    clcg_df.to_csv(clcg_path,        index=False)
    spearman_df.to_csv(spearman_path, index=False)

    log.info(f"Saved: {scores_path}")
    log.info(f"Saved: {detail_path}")
    log.info(f"Saved: {clcg_path}")
    log.info(f"Saved: {spearman_path}")

    # ── Human-readable summary ────────────────────────────────────────────────
    lines = [
        "MulBrandEval — Phase 1B Results",
        "=" * 70,
        "",
        "METRIC SUMMARY — mean CLIP-Score / mean mCLIP / Vendi Score",
        "per (model x language)",
        "-" * 70,
        summary_df.to_string(index=False),
        "",
        "CLCG TABLE — Cross-Lingual Compliance Gap",
        "Formula : CLCG = Score(EN) - Score(Lang)",
        "Sign    : Positive = non-English WORSE than English (degradation)",
        "Metrics : CLIP-Score, mCLIP Score, Vendi Score (all higher = better)",
        "-" * 70,
        clcg_df[[
            "model", "language",
            "clcg_clip", "clcg_mclip", "clcg_vendi"
        ]].to_string(index=False),
        "",
        "SPEARMAN rho — Automated Metrics vs GenAI-Bench Human Ratings (RQ1)",
        "GenAI-Bench prompts only (GB001–GB050, n=50 per condition)",
        "-" * 70,
        spearman_df[[
            "model", "language", "metric", "n_prompts",
            "spearman_rho", "p_value"
        ]].to_string(index=False),
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Saved: {summary_path}")

    # ── Print to terminal ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1B COMPLETE — RESULTS SUMMARY")
    print("=" * 70)
    print("\nMetric summary (mean per model x language):")
    print(summary_df.to_string(index=False))
    print("\nCLCG table (positive = degradation vs English):")
    print(clcg_df[[
        "model", "language", "clcg_clip", "clcg_mclip", "clcg_vendi"
    ]].to_string(index=False))
    print("\nSpearman rho vs human ratings (RQ1):")
    print(spearman_df[[
        "model", "language", "metric", "spearman_rho", "p_value"
    ]].to_string(index=False))
    print("=" * 70)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("MulBrandEval — Phase 1B Metric Computation")
    log.info(f"Device          : {DEVICE}")
    log.info(f"Generated images: {GENERATED_ROOT}")
    log.info(f"Results         : {RESULTS_DIR}")

    make_dirs()

    log.info("Step 1 — Loading prompt CSVs")
    prompts_en, native_captions = load_all_prompts()
    log.info(f"  {len(prompts_en)} prompts | "
             f"{len(prompts_en[prompts_en['source']=='GenAI-Bench'])} GenAI-Bench "
             f"(with human scores for Spearman)")

    log.info("Step 2 — Loading CLIP and mCLIP models")
    clip_model, clip_processor, mclip_model = load_models()

    log.info("Step 3 — Computing per-image CLIP-Score + mCLIP, collecting embeddings for Vendi")
    per_image_df, summary_df = run_all_metrics(
        prompts_en, native_captions, clip_model, clip_processor, mclip_model
    )

    log.info("Step 4 — Computing CLCG table")
    clcg_df = compute_clcg(summary_df)

    log.info("Step 5 — Computing Spearman rho (RQ1)")
    spearman_df = compute_spearman(per_image_df, prompts_en)

    log.info("Step 6 — Saving all outputs")
    save_outputs(summary_df, per_image_df, clcg_df, spearman_df)

    log.info("Phase 1B metric computation complete.")