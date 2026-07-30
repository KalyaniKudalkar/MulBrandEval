"""
phase1b_alignment_baseline/phase1b_bootstrap.py

Point 3 statistical testing for Phase 1B: cluster bootstrap CIs (BCa) and
Wilcoxon signed-rank tests with effect sizes, on the CLIP-Score / mCLIP
CLCG gaps already computed in per_image_scores.csv.

Confirmatory hypothesis family (Bonferroni-corrected):
    RQ2 - per-language CLCG gap (EN vs DE/FR/ES/AR), each model, each metric
    RQ4 - cross-model double-difference (does FLUX shrink the language gap
          vs SD v1.5?), each language, each metric

Does NOT modify or re-run phase1b_metrics.py - reads its existing output.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stats.bootstrap_testing import (
    cluster_bootstrap,
    jackknife_estimates,
    percentile_ci,
    bca_ci,
    wilcoxon_with_effect_size,
)

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_PATH  = PROJECT_ROOT / "results" / "phase1b" / "per_image_scores.csv"
OUTPUT_DIR   = PROJECT_ROOT / "results" / "phase1b"

MODELS     = ["sd15", "flux"]
LANGUAGES  = ["de", "fr", "es", "ar"]   # non-English, compared against "en"
METRICS    = ["clip_score", "mclip_score"]
B          = 10000
SEED       = 42
ALPHA      = 0.05
N_TESTS    = len(LANGUAGES)             # Bonferroni family size = 4 languages


def load_and_pivot(scores_path: Path) -> pd.DataFrame:
    """
    Loads the long-format per_image_scores.csv and pivots it to wide format:
    one row per prompt_id, columns named like 'clip_score_sd15_en',
    'mclip_score_flux_ar', etc.
    """
    long_df = pd.read_csv(scores_path)

    wide_df = long_df.pivot_table(
        index="prompt_id",
        columns=["model", "language"],
        values=["clip_score", "mclip_score"],
    )
    # Flatten the MultiIndex columns: ('clip_score', 'sd15', 'en') -> 'clip_score_sd15_en'
    wide_df.columns = ["_".join(col) for col in wide_df.columns]
    wide_df = wide_df.reset_index()

    return wide_df


def make_gap_metric_fn(model: str, lang: str, metric: str):
    """
    Returns a metric_fn computing the CLCG gap (EN - lang) for one
    model/language/metric combination, to be passed into cluster_bootstrap.
    """
    en_col   = f"{metric}_{model}_en"
    lang_col = f"{metric}_{model}_{lang}"

    def metric_fn(df: pd.DataFrame) -> float:
        return df[en_col].mean() - df[lang_col].mean()

    return metric_fn


def make_double_diff_metric_fn(lang: str, metric: str):
    """
    Cross-model (RQ4) double-difference: how much bigger is SD v1.5's
    language-induced drop than FLUX's, for this language/metric.
    Positive value = SD v1.5 drops more than FLUX (FLUX is more robust).
    """
    def metric_fn(df: pd.DataFrame) -> float:
        sd_gap   = df[f"{metric}_sd15_en"].mean() - df[f"{metric}_sd15_{lang}"].mean()
        flux_gap = df[f"{metric}_flux_en"].mean() - df[f"{metric}_flux_{lang}"].mean()
        return sd_gap - flux_gap

    return metric_fn


def run_ci_and_test(
    wide_df: pd.DataFrame,
    metric_fn,
    en_col: str = None,
    lang_col: str = None,
) -> dict:
    """
    Runs bootstrap CI (BCa) for a given metric_fn, and - if en_col/lang_col
    are provided - also runs the paired Wilcoxon test on the raw per-prompt
    values (not the resampled aggregate).
    """
    original = metric_fn(wide_df)
    replicates = cluster_bootstrap(wide_df, metric_fn, cluster_col="prompt_id", B=B, seed=SEED)
    jack_ests = jackknife_estimates(wide_df, metric_fn, cluster_col="prompt_id")

    perc_lo, perc_hi = percentile_ci(replicates, alpha=ALPHA)
    bca_lo, bca_hi = bca_ci(replicates, original, jack_ests, alpha=ALPHA)

    result = {
        "estimate": original,
        "percentile_ci_low": perc_lo,
        "percentile_ci_high": perc_hi,
        "bca_ci_low": bca_lo,
        "bca_ci_high": bca_hi,
    }

    if en_col is not None and lang_col is not None:
        wtest = wilcoxon_with_effect_size(
            wide_df[en_col].to_numpy(), wide_df[lang_col].to_numpy()
        )
        result.update({
            "wilcoxon_statistic": wtest["statistic"],
            "wilcoxon_p_raw": wtest["p_value"],
            "wilcoxon_effect_size_r": wtest["effect_size_r"],
            "n_pairs": wtest["n_pairs"],
        })

    return result


def main():
    print(f"Loading {SCORES_PATH} ...")
    wide_df = load_and_pivot(SCORES_PATH)
    print(f"Pivoted to wide format: {wide_df.shape[0]} prompts, {wide_df.shape[1]} columns\n")

    rq2_rows = []
    for model in MODELS:
        for metric in METRICS:
            for lang in LANGUAGES:
                print(f"RQ2  |  {model} / {metric} / en-vs-{lang}")
                metric_fn = make_gap_metric_fn(model, lang, metric)
                res = run_ci_and_test(
                    wide_df, metric_fn,
                    en_col=f"{metric}_{model}_en",
                    lang_col=f"{metric}_{model}_{lang}",
                )
                res.update({"hypothesis_family": "RQ2", "model": model, "metric": metric, "language": lang})
                rq2_rows.append(res)

    rq4_rows = []
    for metric in METRICS:
        for lang in LANGUAGES:
            print(f"RQ4  |  {metric} / double-diff / {lang}")
            metric_fn = make_double_diff_metric_fn(lang, metric)
            res = run_ci_and_test(wide_df, metric_fn)  # no single Wilcoxon pair here
            res.update({"hypothesis_family": "RQ4", "model": "sd15_vs_flux", "metric": metric, "language": lang})
            rq4_rows.append(res)

    results_df = pd.concat([pd.DataFrame(rq2_rows), pd.DataFrame(rq4_rows)], ignore_index=True)

    # Bonferroni correction, applied within each (hypothesis_family, model, metric) group
    # across the 4-language family - this is the confirmatory tier.
    results_df["wilcoxon_p_bonferroni"] = np.nan
    rq2_mask = results_df["hypothesis_family"] == "RQ2"
    results_df.loc[rq2_mask, "wilcoxon_p_bonferroni"] = (
        results_df.loc[rq2_mask, "wilcoxon_p_raw"] * N_TESTS
    ).clip(upper=1.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "clcg_phase1b_with_ci.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Total rows: {len(results_df)} (RQ2: {len(rq2_rows)}, RQ4: {len(rq4_rows)})")
    print("\nPreview:")
    print(results_df[["hypothesis_family", "model", "metric", "language", "estimate",
                       "bca_ci_low", "bca_ci_high", "wilcoxon_p_bonferroni"]].to_string(index=False))


if __name__ == "__main__":
    main()