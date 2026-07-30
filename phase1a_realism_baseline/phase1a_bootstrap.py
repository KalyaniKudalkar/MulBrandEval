"""
phase1a_realism_baseline/phase1a_bootstrap.py

Point 3 statistical testing for Phase 1A: cluster bootstrap CIs (BCa) for
CMMD (primary metric), and subsampling-based stability ranges for FID
(corroborating only, per the agreed design).

Frechet distance uses symmetric eigendecomposition (eigh) rather than
scipy.linalg.sqrtm - same result, meaningfully faster on 2048x2048
InceptionV3 covariance matrices. For the FID subsampling loop, the
reference set's expensive eigendecomposition is computed ONCE per repeat
and reused for both the language and English comparison, instead of
being recomputed twice - since the matrix dimension (2048x2048) is fixed
regardless of sample size, this and reducing N_SUB are the two levers
that matter for runtime.

Reads from the validated caches built by phase1a_cache_cmmd_embeddings.py
and phase1a_cache_fid_features.py. Does not touch images or re-run
CLIP/InceptionV3 - everything here operates on cached embeddings/features.

Confirmatory hypothesis family:
    RQ2 - per-language CLCG gap (EN vs DE/FR/ES/AR), each model, CMMD only
          (Wilcoxon does not apply - set-level metric, no per-prompt score;
          this is the documented Phase 1A Wilcoxon-inapplicability carve-out)
    RQ4 - cross-model double-difference (does FLUX shrink the language gap?)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from stats.bootstrap_testing import cluster_bootstrap, jackknife_estimates, bca_ci

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "results" / "phase1a" / "cache"
PROMPTS_CSV  = PROJECT_ROOT / "data" / "prompts" / "en" / "coco_prompts_en.csv"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "phase1a"
CMMD_MODULE  = PROJECT_ROOT / "metrics" / "cmmd.py"

MODELS    = ["sd15", "flux"]
LANGUAGES = ["de", "fr", "es", "ar"]
SEEDS     = [42, 123]
B         = 10000        # CMMD bootstrap replicates (primary metric)
N_SUB     = 30            # FID subsampling repeats - reduced from 100, then 30.
                          # Matrix size (2048x2048) is fixed regardless of sample
                          # size, so N_SUB is the main remaining runtime lever;
                          # FID remains corroborating-only, not primary.
SUB_SIZE  = 40            # FID subsample size (of 50 prompts)
SEED      = 42
ALPHA     = 0.05


def load_cmmd_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmmd", str(CMMD_MODULE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_caches():
    with open(CACHE_DIR / "cmmd_embeddings.pkl", "rb") as f:
        cmmd_cache = pickle.load(f)
    with open(CACHE_DIR / "fid_features.pkl", "rb") as f:
        fid_cache = pickle.load(f)
    return cmmd_cache, fid_cache


def gather_generated(cache: dict, model: str, lang: str, prompt_ids) -> np.ndarray:
    """Pools BOTH seeds for every prompt_id in prompt_ids (duplicates included)."""
    feats = []
    for pid in prompt_ids:
        for seed in SEEDS:
            key = (model, lang, pid, seed)
            if key in cache["generated"]:
                feats.append(cache["generated"][key])
    return np.stack(feats)


def gather_reference(cache: dict, prompt_ids) -> np.ndarray:
    return np.stack([cache["reference"][pid] for pid in prompt_ids])


# ── CMMD: full bootstrap (primary metric) ──────────────────────────────────

def make_cmmd_gap_fn(cache, cmmd_module, model, lang):
    def metric_fn(df: pd.DataFrame) -> float:
        prompt_ids = df["prompt_id"].tolist()
        X = gather_reference(cache, prompt_ids)

        Y_en = gather_generated(cache, model, "en", prompt_ids)
        Y_lang = gather_generated(cache, model, lang, prompt_ids)

        def cmmd_score(Y):
            k_XX = cmmd_module._gaussian_rbf_kernel(X, X, cmmd_module._SIGMA)
            k_YY = cmmd_module._gaussian_rbf_kernel(Y, Y, cmmd_module._SIGMA)
            k_XY = cmmd_module._gaussian_rbf_kernel(X, Y, cmmd_module._SIGMA)
            return cmmd_module._SCALE * (k_XX + k_YY - 2 * k_XY)

        return cmmd_score(Y_lang) - cmmd_score(Y_en)
    return metric_fn


def make_cmmd_double_diff_fn(cache, cmmd_module, lang):
    def metric_fn(df: pd.DataFrame) -> float:
        prompt_ids = df["prompt_id"].tolist()
        X = gather_reference(cache, prompt_ids)

        def cmmd_score(model_key, lang_key):
            Y = gather_generated(cache, model_key, lang_key, prompt_ids)
            k_XX = cmmd_module._gaussian_rbf_kernel(X, X, cmmd_module._SIGMA)
            k_YY = cmmd_module._gaussian_rbf_kernel(Y, Y, cmmd_module._SIGMA)
            k_XY = cmmd_module._gaussian_rbf_kernel(X, Y, cmmd_module._SIGMA)
            return cmmd_module._SCALE * (k_XX + k_YY - 2 * k_XY)

        sd_gap = cmmd_score("sd15", lang) - cmmd_score("sd15", "en")
        flux_gap = cmmd_score("flux", lang) - cmmd_score("flux", "en")
        return sd_gap - flux_gap
    return metric_fn


# ── FID: subsampling stability range (corroborating only) ──────────────────

def sqrt_psd(sigma: np.ndarray) -> np.ndarray:
    """Symmetric PSD matrix square root via eigh - factored out so it can
    be computed once per repeat and reused, instead of recomputed twice."""
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def frechet_distance_precomputed(mu1, sigma1, sigma1_sqrt, mu2, sigma2) -> float:
    """Same formula as a standard Frechet distance, but takes sigma1_sqrt
    as an argument instead of recomputing it - lets the caller reuse it
    across multiple comparisons against the same reference set."""
    diff = mu1 - mu2
    M = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.clip(eigvals_M, 0, None)
    trace_sqrt_M = np.sum(np.sqrt(eigvals_M))
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_sqrt_M)


def fid_subsampling_range(cache, model, lang, all_prompt_ids, rng) -> tuple:
    """
    Subsampling WITHOUT replacement (b=40 of 50), repeated N_SUB times.
    Reference set's expensive eigendecomposition is computed ONCE per
    repeat and reused for both the language and English comparison,
    instead of being recomputed twice.
    Returns (median_gap, low, high) as a stability range, NOT a formal CI -
    per the agreed design, FID is corroborating only.
    """
    gaps = np.empty(N_SUB)
    for i in range(N_SUB):
        sub_ids = rng.choice(all_prompt_ids, size=SUB_SIZE, replace=False)

        ref_feats = gather_reference(cache, sub_ids)
        mu_ref = ref_feats.mean(axis=0)
        sigma_ref = np.cov(ref_feats, rowvar=False)
        sigma_ref_sqrt = sqrt_psd(sigma_ref)  # computed once, reused twice below

        gen_en = gather_generated(cache, model, "en", sub_ids)
        gen_lang = gather_generated(cache, model, lang, sub_ids)

        mu_en, sigma_en = gen_en.mean(axis=0), np.cov(gen_en, rowvar=False)
        mu_lang, sigma_lang = gen_lang.mean(axis=0), np.cov(gen_lang, rowvar=False)

        fid_en = frechet_distance_precomputed(mu_ref, sigma_ref, sigma_ref_sqrt, mu_en, sigma_en)
        fid_lang = frechet_distance_precomputed(mu_ref, sigma_ref, sigma_ref_sqrt, mu_lang, sigma_lang)

        gaps[i] = fid_lang - fid_en

    return float(np.median(gaps)), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def main():
    print("Loading caches...")
    cmmd_cache, fid_cache = load_caches()
    cmmd_module = load_cmmd_module()

    prompt_ids = pd.read_csv(PROMPTS_CSV)["prompt_id"].tolist()
    index_df = pd.DataFrame({"prompt_id": prompt_ids})
    rng = np.random.default_rng(SEED)

    rq2_rows = []
    for model in MODELS:
        for lang in LANGUAGES:
            print(f"RQ2 (CMMD) | {model} / en-vs-{lang}")
            metric_fn = make_cmmd_gap_fn(cmmd_cache, cmmd_module, model, lang)
            original = metric_fn(index_df)
            replicates = cluster_bootstrap(index_df, metric_fn, cluster_col="prompt_id", B=B, seed=SEED)
            jack = jackknife_estimates(index_df, metric_fn, cluster_col="prompt_id")
            ci_lo, ci_hi = bca_ci(replicates, original, jack, alpha=ALPHA)
            rq2_rows.append({
                "hypothesis_family": "RQ2", "metric": "cmmd", "model": model, "language": lang,
                "estimate": original, "bca_ci_low": ci_lo, "bca_ci_high": ci_hi,
            })

            print(f"RQ2 (FID, subsampling, N_SUB={N_SUB}) | {model} / en-vs-{lang}")
            med, lo, hi = fid_subsampling_range(fid_cache, model, lang, prompt_ids, rng)
            rq2_rows.append({
                "hypothesis_family": "RQ2", "metric": "fid_stability_range", "model": model, "language": lang,
                "estimate": med, "bca_ci_low": lo, "bca_ci_high": hi,
            })

    rq4_rows = []
    for lang in LANGUAGES:
        print(f"RQ4 (CMMD) | double-diff / {lang}")
        metric_fn = make_cmmd_double_diff_fn(cmmd_cache, cmmd_module, lang)
        original = metric_fn(index_df)
        replicates = cluster_bootstrap(index_df, metric_fn, cluster_col="prompt_id", B=B, seed=SEED)
        jack = jackknife_estimates(index_df, metric_fn, cluster_col="prompt_id")
        ci_lo, ci_hi = bca_ci(replicates, original, jack, alpha=ALPHA)
        rq4_rows.append({
            "hypothesis_family": "RQ4", "metric": "cmmd", "model": "sd15_vs_flux", "language": lang,
            "estimate": original, "bca_ci_low": ci_lo, "bca_ci_high": ci_hi,
        })

    results_df = pd.concat([pd.DataFrame(rq2_rows), pd.DataFrame(rq4_rows)], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "clcg_phase1a_with_ci.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print("\nPreview:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()