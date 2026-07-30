"""
phase1a_realism_baseline/phase1a_cache_cmmd_embeddings.py

Extracts and caches CLIP embeddings (used by CMMD) for every image in
Phase 1A - the 50 MS-COCO reference images, plus BOTH seeds (42 and 123)
of every generated image across all model x language conditions.
Then validates that recomputing CMMD from cached embeddings reproduces
the already-reported values in fid_cmmd_scores.csv.

Correction from v1: Phase 1A generated 2 seeds per prompt (seed42 +
seed123), unlike later phases which use a single seed. The original
compute_cmmd() globs ALL files in each folder (both seeds pooled
together as one 100-image set per model x language condition) - this
version matches that behaviour exactly.

Does NOT modify metrics/cmmd.py or phase1a_metrics.py.

Outputs:
  results/phase1a/cache/cmmd_embeddings.pkl
"""

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
GENERATED_ROOT = PROJECT_ROOT / "data" / "generated_images" / "phase1a"
REFERENCE_DIR  = PROJECT_ROOT / "data" / "reference_images" / "coco_phase1a"
PROMPTS_CSV    = PROJECT_ROOT / "data" / "prompts" / "en" / "coco_prompts_en.csv"
CMMD_MODULE    = PROJECT_ROOT / "metrics" / "cmmd.py"
CACHE_DIR      = PROJECT_ROOT / "results" / "phase1a" / "cache"
REPORTED_CSV   = PROJECT_ROOT / "results" / "phase1a" / "fid_cmmd_scores.csv"

MODELS    = ["sd15", "flux"]
LANGUAGES = ["en", "de", "fr", "es", "ar"]
SEEDS     = [42, 123]   # Phase 1A used BOTH seeds - confirmed via folder listing


def load_cmmd_module():
    spec = importlib.util.spec_from_file_location("cmmd", str(CMMD_MODULE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prompt_to_image_id() -> dict:
    df = pd.read_csv(PROMPTS_CSV)
    return dict(zip(df["prompt_id"], df["image_id"]))


def cache_all_embeddings(module) -> dict:
    """
    cache["reference"][prompt_id]                     -> embedding
    cache["generated"][(model, lang, prompt_id, seed)] -> embedding
    """
    model, preprocess = module._load_clip()
    prompt_to_image_id = build_prompt_to_image_id()

    cache = {"reference": {}, "generated": {}}

    print(f"Caching {len(prompt_to_image_id)} reference image embeddings...")
    for prompt_id, image_id in prompt_to_image_id.items():
        ref_path = REFERENCE_DIR / f"{image_id:012d}.jpg"
        if not ref_path.exists():
            print(f"  MISSING reference image: {ref_path}")
            continue
        emb = module._extract_features([str(ref_path)], model, preprocess)
        cache["reference"][prompt_id] = emb[0]

    total_expected = 0
    total_missing = 0

    for m in MODELS:
        for lang in LANGUAGES:
            print(f"Caching generated embeddings: {m}/{lang} (both seeds) ...")
            for prompt_id in prompt_to_image_id.keys():
                for seed in SEEDS:
                    total_expected += 1
                    img_path = GENERATED_ROOT / m / lang / f"{prompt_id}_seed{seed}.png"
                    if not img_path.exists():
                        print(f"  MISSING: {img_path}")
                        total_missing += 1
                        continue
                    emb = module._extract_features([str(img_path)], model, preprocess)
                    cache["generated"][(m, lang, prompt_id, seed)] = emb[0]

    print(f"\nExpected generated images: {total_expected}  |  Missing: {total_missing}")
    return cache


def save_cache(cache: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "cmmd_embeddings.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved cache: {out_path}")
    print(f"  Reference embeddings cached: {len(cache['reference'])}")
    print(f"  Generated embeddings cached: {len(cache['generated'])}")


def recompute_cmmd_from_cache(cache: dict, module, model: str, lang: str) -> tuple:
    """
    Pools BOTH seeds together as one generated set per (model, lang) -
    matching what the original get_image_paths()-based compute_cmmd() did.
    Returns (cmmd_score, n_generated_images_used).
    """
    ref_embeddings = np.stack(list(cache["reference"].values()))
    gen_embeddings = np.stack([
        emb for (m, l, pid, seed), emb in cache["generated"].items()
        if m == model and l == lang
    ])

    k_XX = module._gaussian_rbf_kernel(ref_embeddings, ref_embeddings, module._SIGMA)
    k_YY = module._gaussian_rbf_kernel(gen_embeddings, gen_embeddings, module._SIGMA)
    k_XY = module._gaussian_rbf_kernel(ref_embeddings, gen_embeddings, module._SIGMA)

    cmmd = module._SCALE * (k_XX + k_YY - 2 * k_XY)
    return float(cmmd), gen_embeddings.shape[0]


def validate_against_reported(cache: dict, module):
    reported_df = pd.read_csv(REPORTED_CSV)

    print("\n" + "=" * 80)
    print("VALIDATION: cached-embedding CMMD vs originally reported CMMD")
    print("=" * 80)

    rows = []
    for m in MODELS:
        for lang in LANGUAGES:
            recomputed, n_gen = recompute_cmmd_from_cache(cache, module, m, lang)
            reported_row = reported_df[(reported_df["model"] == m) & (reported_df["language"] == lang)]
            reported = float(reported_row["cmmd"].iloc[0]) if len(reported_row) else float("nan")
            reported_n = int(reported_row["n_images"].iloc[0]) if len(reported_row) else None
            diff = recomputed - reported
            rows.append({
                "model": m, "language": lang,
                "n_images_cached": n_gen, "n_images_reported": reported_n,
                "reported_cmmd": reported, "recomputed_cmmd": recomputed,
                "difference": diff,
            })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    result_df["relative_difference"] = (result_df["difference"].abs() /
                                         result_df["reported_cmmd"].abs())
    max_rel_diff = result_df["relative_difference"].max()
    print(f"\nMax relative difference: {max_rel_diff:.6%}")
    if max_rel_diff < 0.01:  # within 1% - generous given float32 summation noise
        print("PASS - cached-embedding recomputation matches reported values (within floating-point tolerance).")
    else:
        print("FAIL - discrepancy too large relative to score magnitude. Do not proceed to bootstrap until resolved.")


if __name__ == "__main__":
    print("Loading CMMD module and CLIP model...")
    cmmd_module = load_cmmd_module()

    print("\nExtracting and caching embeddings (this runs the CLIP model once per image)...")
    embedding_cache = cache_all_embeddings(cmmd_module)

    save_cache(embedding_cache)

    validate_against_reported(embedding_cache, cmmd_module)