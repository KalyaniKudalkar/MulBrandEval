"""
phase1a_realism_baseline/phase1a_cache_fid_features.py

Extracts and caches InceptionV3 features (used by FID) for every image in
Phase 1A - the 50 MS-COCO reference images, plus both seeds (42 and 123)
of every generated image across all model x language conditions. Then
validates that recomputing FID from cached features, using our own
Frechet distance formula, reproduces the already-reported values in
fid_cmmd_scores.csv.

Reuses torchmetrics' own InceptionV3 network (via FrechetInceptionDistance's
internal .inception attribute) for feature extraction, so features are
guaranteed to come from the exact same network as the original computation -
only the Frechet distance FORMULA itself is implemented independently here,
since torchmetrics doesn't expose a way to feed pre-computed features back in.

Frechet distance is computed via symmetric eigendecomposition (eigh) rather
than scipy.linalg.sqrtm - mathematically identical result, since
trace(sqrtm(sigma1 @ sigma2)) == trace(sqrtm(sigma1^0.5 @ sigma2 @ sigma1^0.5)),
and the right-hand matrix is symmetric PSD, so the fast symmetric eigensolver
applies. Meaningfully faster than general-purpose sqrtm on 2048x2048 matrices,
with no complex-number cleanup needed.

Does NOT modify phase1a_metrics.py.

Outputs:
  results/phase1a/cache/fid_features.pkl
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
GENERATED_ROOT = PROJECT_ROOT / "data" / "generated_images" / "phase1a"
REFERENCE_DIR  = PROJECT_ROOT / "data" / "reference_images" / "coco_phase1a"
PROMPTS_CSV    = PROJECT_ROOT / "data" / "prompts" / "en" / "coco_prompts_en.csv"
CACHE_DIR      = PROJECT_ROOT / "results" / "phase1a" / "cache"
REPORTED_CSV   = PROJECT_ROOT / "results" / "phase1a" / "fid_cmmd_scores.csv"

MODELS    = ["sd15", "flux"]
LANGUAGES = ["en", "de", "fr", "es", "ar"]
SEEDS     = [42, 123]   # confirmed via folder listing - same as CMMD caching

TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
])


def build_prompt_to_image_id() -> dict:
    df = pd.read_csv(PROMPTS_CSV)
    return dict(zip(df["prompt_id"], df["image_id"]))


def load_inception_extractor():
    """
    Instantiates torchmetrics' FrechetInceptionDistance purely to access
    its internal InceptionV3 network - we never call .update()/.compute()
    on this object, only its .inception submodule directly, so we can
    extract and cache features one image at a time.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_obj = FrechetInceptionDistance(feature=2048, normalize=True)
    fid_obj.eval()
    return fid_obj.inception


def extract_feature(image_path: Path, inception_net) -> np.ndarray:
    """
    Loads one image, applies the same Resize+ToTensor transform as the
    original compute_fid(), converts to uint8 [0,255] (matching what
    torchmetrics does internally when normalize=True), and runs it
    through the cached InceptionV3 network.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)          # (1, 3, 299, 299), float [0,1]
    tensor_uint8 = (tensor * 255).byte()          # match torchmetrics' internal conversion

    with torch.no_grad():
        feat = inception_net(tensor_uint8)

    return feat.squeeze(0).cpu().numpy()


def cache_all_features(inception_net) -> dict:
    prompt_to_image_id = build_prompt_to_image_id()
    cache = {"reference": {}, "generated": {}}

    print(f"Caching {len(prompt_to_image_id)} reference image features...")
    for prompt_id, image_id in prompt_to_image_id.items():
        ref_path = REFERENCE_DIR / f"{image_id:012d}.jpg"
        if not ref_path.exists():
            print(f"  MISSING reference image: {ref_path}")
            continue
        cache["reference"][prompt_id] = extract_feature(ref_path, inception_net)

    total_expected = 0
    total_missing = 0

    for m in MODELS:
        for lang in LANGUAGES:
            print(f"Caching generated features: {m}/{lang} (both seeds) ...")
            for prompt_id in prompt_to_image_id.keys():
                for seed in SEEDS:
                    total_expected += 1
                    img_path = GENERATED_ROOT / m / lang / f"{prompt_id}_seed{seed}.png"
                    if not img_path.exists():
                        print(f"  MISSING: {img_path}")
                        total_missing += 1
                        continue
                    cache["generated"][(m, lang, prompt_id, seed)] = extract_feature(img_path, inception_net)

    print(f"\nExpected generated images: {total_expected}  |  Missing: {total_missing}")
    return cache


def save_cache(cache: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "fid_features.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved cache: {out_path}")
    print(f"  Reference features cached: {len(cache['reference'])}")
    print(f"  Generated features cached: {len(cache['generated'])}")


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """
    Frechet distance via symmetric eigendecomposition instead of scipy's
    general sqrtm - mathematically identical result
    (trace(sqrtm(sigma1 @ sigma2)) == trace(sqrtm(sigma1^0.5 @ sigma2 @ sigma1^0.5))),
    and the right-hand matrix is symmetric PSD, so the fast symmetric
    eigensolver (eigh/eigvalsh) applies instead of general-purpose sqrtm.
    """
    diff = mu1 - mu2

    eigvals1, eigvecs1 = np.linalg.eigh(sigma1)
    eigvals1 = np.clip(eigvals1, 0, None)  # guard against tiny negative numerical noise
    sigma1_sqrt = eigvecs1 @ np.diag(np.sqrt(eigvals1)) @ eigvecs1.T

    M = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    eigvals_M = np.linalg.eigvalsh(M)       # eigenvalues only - cheaper than full eigh
    eigvals_M = np.clip(eigvals_M, 0, None)
    trace_sqrt_M = np.sum(np.sqrt(eigvals_M))

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_sqrt_M)


def recompute_fid_from_cache(cache: dict, model: str, lang: str) -> tuple:
    """
    Pools both seeds together as one generated set per (model, lang) -
    matching the original folder-glob behaviour. Returns (fid_score, n_used).
    """
    ref_features = np.stack(list(cache["reference"].values()))
    gen_features = np.stack([
        feat for (m, l, pid, seed), feat in cache["generated"].items()
        if m == model and l == lang
    ])

    mu_ref, sigma_ref = ref_features.mean(axis=0), np.cov(ref_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    fid_val = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
    return fid_val, gen_features.shape[0]


def validate_against_reported(cache: dict):
    reported_df = pd.read_csv(REPORTED_CSV)

    print("\n" + "=" * 80)
    print("VALIDATION: cached-feature FID vs originally reported FID")
    print("=" * 80)

    rows = []
    for m in MODELS:
        for lang in LANGUAGES:
            recomputed, n_gen = recompute_fid_from_cache(cache, m, lang)
            reported_row = reported_df[(reported_df["model"] == m) & (reported_df["language"] == lang)]
            reported = float(reported_row["fid"].iloc[0]) if len(reported_row) else float("nan")
            reported_n = int(reported_row["n_images"].iloc[0]) if len(reported_row) else None
            diff = recomputed - reported
            rows.append({
                "model": m, "language": lang,
                "n_images_cached": n_gen, "n_images_reported": reported_n,
                "reported_fid": reported, "recomputed_fid": recomputed,
                "difference": diff,
            })

    result_df = pd.DataFrame(rows)
    result_df["relative_difference"] = result_df["difference"].abs() / result_df["reported_fid"].abs()
    print(result_df.to_string(index=False))

    max_rel_diff = result_df["relative_difference"].max()
    print(f"\nMax relative difference: {max_rel_diff:.6%}")
    if max_rel_diff < 0.01:
        print("PASS - cached-feature recomputation matches reported values (within floating-point/numerical tolerance).")
    else:
        print("FAIL - discrepancy too large relative to score magnitude. Do not proceed to bootstrap until resolved.")


if __name__ == "__main__":
    print("Loading InceptionV3 feature extractor (via torchmetrics)...")
    inception_net = load_inception_extractor()

    print("\nExtracting and caching features (this runs InceptionV3 once per image)...")
    feature_cache = cache_all_features(inception_net)

    save_cache(feature_cache)

    validate_against_reported(feature_cache)