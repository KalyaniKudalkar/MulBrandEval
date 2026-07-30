"""
stats/bootstrap_testing.py

Reusable statistical testing utilities for cross-lingual / cross-condition
evaluation of paired, clustered data. Built for MulBrandEval but has no
project-specific assumptions - works on any dataset where observations
are grouped by a cluster identifier (e.g. one prompt measured under
several conditions).
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def cluster_bootstrap(
    data: pd.DataFrame,
    metric_fn,
    cluster_col: str = "prompt_id",
    B: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    Prompt-level (cluster) bootstrap.

    Resamples whole clusters (e.g. all rows belonging to one prompt_id)
    with replacement, and recomputes a user-supplied metric on each
    resampled dataset. Returns the array of B replicate values, from
    which any confidence interval (percentile, BCa, etc.) can be built.

    Performance note: row positions for each cluster are precomputed once
    before the loop, and each replicate is built with a single .iloc[]
    call on a concatenated array of positions - avoiding the repeated
    groupby lookups + DataFrame concatenation of the original version.
    Produces identical results to that version, just faster at large B.
    """
    rng = np.random.default_rng(seed)
    clusters = data[cluster_col].unique()
    n_clusters = len(clusters)

    # Precompute row positions per cluster ONCE, outside the loop.
    col_values = data[cluster_col].to_numpy()
    cluster_to_positions = {
        c: np.where(col_values == c)[0] for c in clusters
    }

    replicates = np.empty(B)

    for i in range(B):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        positions = np.concatenate([cluster_to_positions[c] for c in sampled_clusters])
        resampled_data = data.iloc[positions]
        replicates[i] = metric_fn(resampled_data)

    return replicates


def jackknife_estimates(
    data: pd.DataFrame,
    metric_fn,
    cluster_col: str = "prompt_id",
) -> np.ndarray:
    """
    Leave-one-cluster-out jackknife estimates, used only as an input
    to the BCa acceleration correction below. Unchanged - only runs
    n_clusters times (not B times), so not a performance concern.
    """
    clusters = data[cluster_col].unique()
    estimates = np.empty(len(clusters))

    for i, c in enumerate(clusters):
        subset = data[data[cluster_col] != c]
        estimates[i] = metric_fn(subset)

    return estimates


def percentile_ci(replicates: np.ndarray, alpha: float = 0.05) -> tuple:
    """Plain percentile confidence interval."""
    lower = np.percentile(replicates, 100 * (alpha / 2))
    upper = np.percentile(replicates, 100 * (1 - alpha / 2))
    return lower, upper


def bca_ci(
    replicates: np.ndarray,
    original_estimate: float,
    jackknife_ests: np.ndarray,
    alpha: float = 0.05,
) -> tuple:
    """
    Bias-corrected and accelerated (BCa) confidence interval.
    """
    prop_less = np.mean(replicates < original_estimate)
    if prop_less == 0 or prop_less == 1:
        z0 = 0.0
    else:
        z0 = scipy_stats.norm.ppf(prop_less)

    jack_mean = jackknife_ests.mean()
    numerator = np.sum((jack_mean - jackknife_ests) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jackknife_ests) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def adjusted_percentile(z):
        return scipy_stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    p_lo = adjusted_percentile(z_lo)
    p_hi = adjusted_percentile(z_hi)

    ci_low = np.percentile(replicates, 100 * p_lo)
    ci_high = np.percentile(replicates, 100 * p_hi)

    return ci_low, ci_high


def wilcoxon_with_effect_size(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
) -> dict:
    """
    Paired Wilcoxon signed-rank test, plus matched-pairs rank-biserial
    correlation as an effect size.
    """
    sample_a = np.asarray(sample_a)
    sample_b = np.asarray(sample_b)

    if len(sample_a) != len(sample_b):
        raise ValueError("sample_a and sample_b must be the same length (paired data).")

    diffs = sample_a - sample_b
    nonzero_diffs = diffs[diffs != 0]
    n_pairs = len(nonzero_diffs)

    if n_pairs == 0:
        return {
            "statistic": np.nan,
            "p_value": 1.0,
            "effect_size_r": 0.0,
            "n_pairs": 0,
        }

    result = scipy_stats.wilcoxon(sample_a, sample_b, zero_method="wilcox")

    abs_diffs = np.abs(nonzero_diffs)
    ranks = scipy_stats.rankdata(abs_diffs)
    pos_rank_sum = ranks[nonzero_diffs > 0].sum()
    neg_rank_sum = ranks[nonzero_diffs < 0].sum()
    total_rank_sum = n_pairs * (n_pairs + 1) / 2.0
    effect_size_r = (pos_rank_sum - neg_rank_sum) / total_rank_sum

    return {
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "effect_size_r": effect_size_r,
        "n_pairs": n_pairs,
    }


if __name__ == "__main__":
    print("Running smoke test for cluster_bootstrap() + bca_ci() + wilcoxon_with_effect_size()...\n")

    rng = np.random.default_rng(0)
    n_prompts = 20
    fake_data = pd.DataFrame({
        "prompt_id": [f"P{i:03d}" for i in range(n_prompts)] * 2,
        "language": ["en"] * n_prompts + ["de"] * n_prompts,
        "score": np.concatenate([
            rng.normal(0.80, 0.03, n_prompts),
            rng.normal(0.75, 0.03, n_prompts),
        ]),
    })

    def clcg_gap_metric(df: pd.DataFrame) -> float:
        en_mean = df.loc[df["language"] == "en", "score"].mean()
        de_mean = df.loc[df["language"] == "de", "score"].mean()
        return en_mean - de_mean

    original_gap = clcg_gap_metric(fake_data)
    replicates = cluster_bootstrap(fake_data, clcg_gap_metric, B=2000)
    jack_ests = jackknife_estimates(fake_data, clcg_gap_metric)

    perc_low, perc_high = percentile_ci(replicates)
    bca_low, bca_high = bca_ci(replicates, original_gap, jack_ests)

    print(f"Original CLCG gap (EN - DE): {original_gap:.4f}")
    print(f"Bootstrap replicates: {len(replicates)}")
    print(f"Replicate mean:       {replicates.mean():.4f}")
    print(f"Replicate std:        {replicates.std():.4f}\n")
    print(f"Percentile 95% CI:    [{perc_low:.4f}, {perc_high:.4f}]")
    print(f"BCa 95% CI:           [{bca_low:.4f}, {bca_high:.4f}]\n")

    en_scores = fake_data.loc[fake_data["language"] == "en"].sort_values("prompt_id")["score"].to_numpy()
    de_scores = fake_data.loc[fake_data["language"] == "de"].sort_values("prompt_id")["score"].to_numpy()

    wilcoxon_result = wilcoxon_with_effect_size(en_scores, de_scores)

    print("Wilcoxon signed-rank test (EN vs DE):")
    print(f"  statistic:     {wilcoxon_result['statistic']:.4f}")
    print(f"  p_value:       {wilcoxon_result['p_value']:.6f}")
    print(f"  effect_size_r: {wilcoxon_result['effect_size_r']:.4f}")
    print(f"  n_pairs:       {wilcoxon_result['n_pairs']}")