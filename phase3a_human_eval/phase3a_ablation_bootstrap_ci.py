"""
phase3a_human_eval/phase3a_ablation_bootstrap_ci.py

Bootstrap confidence intervals on the Spearman rho differences between
pipeline configurations (vs. human overall ratings), using the same
cluster-bootstrap + BCa engine (stats/bootstrap_testing.py) applied
throughout this thesis.

Resampling is clustered on brief_id (not row-independent), since the same
brief can appear more than once in the 120-image sample under different
model/language combinations, and those aren't independent observations.

Tests all three pairwise comparisons, consistent with the RQ3 pattern
already used in Phase 2A (B vs A, C vs A, B vs C).

Run from project root:
    python -m phase3a_human_eval.phase3a_ablation_bootstrap_ci
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from stats.bootstrap_testing import cluster_bootstrap, jackknife_estimates, bca_ci

INPUT_PATH = "results/phase3a/phase3a_120_primary_rater_joined.csv"
OUTPUT_PATH = "results/phase3a/phase3a_ablation_bootstrap_ci.csv"

CONFIG_COLS = {
    "configA": "clip_score_norm_configA",
    "configB": "compliance_score_configB",
    "configC": "compliance_score_configC",
}


def make_rho_diff_metric(col_x: str, col_y: str):
    """Returns a metric_fn: Spearman rho(overall, col_x) - Spearman rho(overall, col_y)."""
    def _metric(d: pd.DataFrame) -> float:
        rho_x = scipy_stats.spearmanr(d["overall"], d[col_x]).correlation
        rho_y = scipy_stats.spearmanr(d["overall"], d[col_y]).correlation
        return rho_x - rho_y
    return _metric


def main():
    df = pd.read_csv(INPUT_PATH)

    comparisons = [
        ("configB", "configA", "Config B vs Config A"),
        ("configC", "configA", "Config C vs Config A"),
        ("configC", "configB", "Config C vs Config B"),
    ]

    results = []
    for cfg_x, cfg_y, label in comparisons:
        col_x, col_y = CONFIG_COLS[cfg_x], CONFIG_COLS[cfg_y]
        metric_fn = make_rho_diff_metric(col_x, col_y)

        original = metric_fn(df)
        replicates = cluster_bootstrap(df, metric_fn, cluster_col="brief_id", B=10000, seed=42)
        jack_ests = jackknife_estimates(df, metric_fn, cluster_col="brief_id")
        ci_low, ci_high = bca_ci(replicates, original, jack_ests)
        significant = not (ci_low <= 0 <= ci_high)

        results.append({
            "comparison": label,
            "rho_difference": original,
            "bca_ci_low": ci_low,
            "bca_ci_high": ci_high,
            "significant_at_95": significant,
        })
        print(f"{label}:")
        print(f"  rho difference: {original:.4f}  BCa 95% CI [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  Significant (CI excludes 0): {significant}\n")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()