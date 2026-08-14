"""
stats/phase2a_statistical_comparison.py

Statistical testing for the Phase 2A ablation study (RQ3):
Configuration A (CLIP-Score baseline) / TIFA baseline /
Configuration C (flat DAG, no short-circuit) / Configuration B (full DAG).

For each configuration:
  - SD v1.5 and FLUX mean scores with BCa 95% CIs (cluster bootstrap by brief_id)
  - Absolute and relative ((FLUX-SD v1.5)/SD v1.5) cross-model gap with BCa 95% CI
  - Paired Wilcoxon signed-rank test (SD v1.5 vs FLUX, paired by brief_id) with
    rank-biserial effect size, Bonferroni-corrected across the 4-configuration
    hypothesis family ("is there a cross-model gap under approach X", tested
    4 times)

Across configurations (the RQ3 test — does the gap significantly GROW):
  - Paired-resample bootstrap CI on the DIFFERENCE in relative gap between
    Configuration B and each of the other three configurations. Because all
    four configs share the same 150 briefs, each bootstrap replicate resamples
    brief_id ONCE and computes both configs' relative gaps from that same
    resampled set — a valid paired comparison, not two independent CIs whose
    overlap is eyeballed.

Reuses stats/bootstrap_testing.py (same engine as Phase 1A/1B).

Run from project root:
    python -m stats.phase2a_statistical_comparison
"""
import numpy as np
import pandas as pd
from pathlib import Path

from stats.bootstrap_testing import (
    cluster_bootstrap,
    jackknife_estimates,
    bca_ci,
    wilcoxon_with_effect_size,
)

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/phase2a")
OUTPUT_DIR  = Path("results/phase2a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "A_clip_baseline": {
        "label": "Configuration A (CLIP-Score baseline)",
        "file": RESULTS_DIR / "configuration_a_clip_baseline.csv",
        "score_col": "clip_score",
    },
    "TIFA_baseline": {
        "label": "TIFA Baseline",
        "file": RESULTS_DIR / "tifa_baseline.csv",
        "score_col": "tifa_score",
    },
    "C_flat_dag": {
        "label": "Configuration C (flat DAG, no short-circuit)",
        "file": RESULTS_DIR / "configuration_c_flat_pipeline.csv",
        "score_col": "compliance_score",
    },
    "B_full_dag": {
        "label": "Configuration B (full DAG, short-circuit)",
        "file": RESULTS_DIR / "english_ablation_table.csv",
        "score_col": "compliance_score",
    },
}

B_REPLICATES = 10_000
ALPHA = 0.05
CLUSTER_COL = "brief_id"


def load_config(key: str) -> pd.DataFrame:
    cfg = CONFIGS[key]
    df = pd.read_csv(cfg["file"])
    df = df[["brief_id", "model", "language", cfg["score_col"]]].copy()
    df = df.rename(columns={cfg["score_col"]: "score"})
    return df


def mean_metric(model: str):
    def _metric(d: pd.DataFrame) -> float:
        return d.loc[d["model"] == model, "score"].mean()
    return _metric


def abs_gap_metric(d: pd.DataFrame) -> float:
    flux = d.loc[d["model"] == "flux", "score"].mean()
    sd15 = d.loc[d["model"] == "sd15", "score"].mean()
    return flux - sd15


def rel_gap_metric(d: pd.DataFrame) -> float:
    # (FLUX - SD v1.5) / SD v1.5 — gap expressed as a fraction of the WEAKER
    # model's score. Matches the convention already used in prior Phase 2A
    # reporting (Config A ~9%, TIFA ~79%, Config C ~85%, Config B ~144%).
    flux = d.loc[d["model"] == "flux", "score"].mean()
    sd15 = d.loc[d["model"] == "sd15", "score"].mean()
    return (flux - sd15) / sd15


def paired_arrays(df: pd.DataFrame) -> tuple:
    """Return (sd15_scores, flux_scores) aligned by brief_id."""
    wide = df.pivot(index="brief_id", columns="model", values="score").sort_index()
    return wide["sd15"].to_numpy(), wide["flux"].to_numpy()


def analyse_config(key: str) -> tuple:
    cfg = CONFIGS[key]
    df = load_config(key)

    results = {"config": key, "label": cfg["label"]}

    # ── Per-model means + BCa CI ──────────────────────────────────────────
    for model in ("sd15", "flux"):
        metric = mean_metric(model)
        est = metric(df)
        reps = cluster_bootstrap(df, metric, cluster_col=CLUSTER_COL, B=B_REPLICATES)
        jk = jackknife_estimates(df, metric, cluster_col=CLUSTER_COL)
        lo, hi = bca_ci(reps, est, jk, alpha=ALPHA)
        results[f"{model}_mean"] = est
        results[f"{model}_bca_lo"] = lo
        results[f"{model}_bca_hi"] = hi

    # ── Absolute + relative gap with BCa CI ───────────────────────────────
    for name, metric in (("abs_gap", abs_gap_metric), ("rel_gap", rel_gap_metric)):
        est = metric(df)
        reps = cluster_bootstrap(df, metric, cluster_col=CLUSTER_COL, B=B_REPLICATES)
        jk = jackknife_estimates(df, metric, cluster_col=CLUSTER_COL)
        lo, hi = bca_ci(reps, est, jk, alpha=ALPHA)
        results[f"{name}_estimate"] = est
        results[f"{name}_bca_lo"] = lo
        results[f"{name}_bca_hi"] = hi

    # ── Paired Wilcoxon (FLUX vs SD v1.5, matched by brief_id) ────────────
    sd15_scores, flux_scores = paired_arrays(df)
    wres = wilcoxon_with_effect_size(flux_scores, sd15_scores)
    results["wilcoxon_statistic"] = wres["statistic"]
    results["wilcoxon_p_raw"] = wres["p_value"]
    results["wilcoxon_effect_size_r"] = wres["effect_size_r"]
    results["n_pairs"] = wres["n_pairs"]

    return results, df


def main():
    print("=" * 70)
    print("PHASE 2A — STATISTICAL COMPARISON (A / TIFA / C / B)")
    print(f"Cluster bootstrap: B={B_REPLICATES}, cluster_col='{CLUSTER_COL}', alpha={ALPHA}")
    print("=" * 70)

    all_results = []
    config_dfs = {}

    for key in CONFIGS:
        print(f"\n--- {CONFIGS[key]['label']} ---")
        res, df = analyse_config(key)
        config_dfs[key] = df
        all_results.append(res)

        print(f"  SD v1.5 mean: {res['sd15_mean']:.4f}  BCa 95% CI [{res['sd15_bca_lo']:.4f}, {res['sd15_bca_hi']:.4f}]")
        print(f"  FLUX    mean: {res['flux_mean']:.4f}  BCa 95% CI [{res['flux_bca_lo']:.4f}, {res['flux_bca_hi']:.4f}]")
        print(f"  Absolute gap: {res['abs_gap_estimate']:.4f}  BCa 95% CI [{res['abs_gap_bca_lo']:.4f}, {res['abs_gap_bca_hi']:.4f}]")
        print(f"  Relative gap: {res['rel_gap_estimate']:.2%}  BCa 95% CI [{res['rel_gap_bca_lo']:.2%}, {res['rel_gap_bca_hi']:.2%}]")
        print(f"  Wilcoxon (FLUX vs SD v1.5): stat={res['wilcoxon_statistic']:.1f}, "
              f"p={res['wilcoxon_p_raw']:.6f}, effect_r={res['wilcoxon_effect_size_r']:.4f}, n={res['n_pairs']}")

    # ── Bonferroni correction across the 4-configuration family ──────────
    n_tests = len(all_results)
    for res in all_results:
        res["wilcoxon_p_bonferroni"] = min(res["wilcoxon_p_raw"] * n_tests, 1.0)

    results_df = pd.DataFrame(all_results)
    out_path = OUTPUT_DIR / "phase2a_statistical_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved per-configuration results: {out_path}")

    # ── RQ3: does the relative gap significantly GROW from baseline to DAG? ─
    print("\n" + "=" * 70)
    print("RQ3 — CROSS-CONFIGURATION GAP COMPARISON")
    print("Bootstrap CI on (relative_gap[Config B] - relative_gap[Config X])")
    print("Paired resampling: same brief_id draw used for both configs per replicate")
    print("=" * 70)

    # Merge all four configs into one wide table keyed by brief_id + model,
    # so a single bootstrap resample of brief_id can be applied to all
    # configs at once (valid pairing — same underlying 150 briefs throughout).
    merged = None
    for key in CONFIGS:
        d = config_dfs[key][["brief_id", "model", "score"]].rename(columns={"score": key})
        merged = d if merged is None else merged.merge(d, on=["brief_id", "model"], how="inner")

    assert len(merged) == 300, f"Expected 300 rows after merge, got {len(merged)}"

    def make_gap_diff_metric(config_x: str, config_ref: str = "B_full_dag"):
        def _metric(d: pd.DataFrame) -> float:
            # Same (FLUX - SD1.5) / SD1.5 convention as rel_gap_metric above.
            flux_ref = d.loc[d["model"] == "flux", config_ref].mean()
            sd15_ref = d.loc[d["model"] == "sd15", config_ref].mean()
            rel_ref = (flux_ref - sd15_ref) / sd15_ref

            flux_x = d.loc[d["model"] == "flux", config_x].mean()
            sd15_x = d.loc[d["model"] == "sd15", config_x].mean()
            rel_x = (flux_x - sd15_x) / sd15_x

            return rel_ref - rel_x
        return _metric

    cross_results = []
    for key in CONFIGS:
        if key == "B_full_dag":
            continue
        metric = make_gap_diff_metric(key)
        est = metric(merged)
        reps = cluster_bootstrap(merged, metric, cluster_col=CLUSTER_COL, B=B_REPLICATES)
        jk = jackknife_estimates(merged, metric, cluster_col=CLUSTER_COL)
        lo, hi = bca_ci(reps, est, jk, alpha=ALPHA)
        significant = not (lo <= 0 <= hi)

        cross_results.append({
            "comparison": f"B_full_dag vs {key}",
            "label": f"{CONFIGS['B_full_dag']['label']} vs {CONFIGS[key]['label']}",
            "rel_gap_difference": est,
            "bca_ci_low": lo,
            "bca_ci_high": hi,
            "significant_at_95": significant,
        })

        print(f"\n  Config B vs {CONFIGS[key]['label']}:")
        print(f"    Relative-gap difference: {est:.2%}  BCa 95% CI [{lo:.2%}, {hi:.2%}]")
        print(f"    Significant at 95% (CI excludes 0): {significant}")

    cross_df = pd.DataFrame(cross_results)
    cross_out_path = OUTPUT_DIR / "phase2a_rq3_gap_comparison.csv"
    cross_df.to_csv(cross_out_path, index=False)
    print(f"\nSaved cross-configuration RQ3 results: {cross_out_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()