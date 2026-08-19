"""
MulBrandEval — Phase 2B Statistical Testing
============================================
Runs the full statistical treatment on Config C's per-node CLCG values
(the primary, unmasked claim — see phase2b_heatmap.py for context).

Tests performed:
  - Wilcoxon signed-rank (two-sided) for continuous nodes: 2, 3, 5,
    and overall compliance_score.
  - McNemar (with Yates continuity correction, or exact binomial for
    n_discordant < 25) for binary pass/fail nodes: 1 and 4 (FLUX only;
    SD v1.5 Node 4 is N/A — architecturally undefined).

Effect sizes:
  - Wilcoxon: r = |Z| / sqrt(N_nonzero)   [small=0.1, medium=0.3, large=0.5]
  - McNemar:  phi = (b-c) / sqrt(n_discordant)

Bootstrap:
  - Cluster bootstrap (B=10,000 resamples on brief_id) for BCa 95% CIs
    on the CLCG point estimate.
  - Same engine and discipline as Phase 1A/1B/2A statistical testing.

Multiple testing corrections (two families):

  1. PRE-REGISTERED RQ2 FAMILY (Bonferroni):
     Overall compliance_score CLCG per (language, model) — 6 tests.
     Directly answers RQ2: "Is the cross-lingual compliance degradation
     statistically significant?"

  2. EXPLORATORY HEATMAP FAMILY (Benjamini-Hochberg):
     Per-node CLCG across all interpretable cells — 27 tests
     (5 nodes × 3 languages × 2 models − 3 N/A cells for SD v1.5 Node 4).
     Answers: "Which nodes drive the gap, and is each significant?"

Inputs (relative to MulBrandEval/ project root):
  results/phase2b/configuration_c_flat_pipeline_multilingual.csv
  results/phase2a/configuration_c_flat_pipeline.csv

Output:
  results/phase2b/phase2b_statistical_results.csv

Run from MulBrandEval/ project root:
    python phase2b_multilingual/phase2b_statistical_testing.py
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Input / Output ────────────────────────────────────────────────────────────

C_MULTILINGUAL = Path("results/phase2b/configuration_c_flat_pipeline_multilingual.csv")
C_ENGLISH      = Path("results/phase2a/configuration_c_flat_pipeline.csv")
OUTPUT_DIR     = Path("results/phase2b")
OUTPUT_FILE    = OUTPUT_DIR / "phase2b_statistical_results.csv"

# ── Constants ─────────────────────────────────────────────────────────────────

LANGUAGES = ["de", "fr", "es"]
MODELS    = ["sd15", "flux"]
B_BOOTSTRAP = 10_000
RNG_SEED    = 42
ALPHA       = 0.05

# Node definitions: name → (score_col, pass_col or None, test_type)
# test_type: "wilcoxon" for continuous, "mcnemar" for binary pass/fail
NODES = {
    "node1": ("node1_score", "node1_pass",  "mcnemar"),
    "node2": ("node2_score", None,           "wilcoxon"),
    "node3": ("node3_score", None,           "wilcoxon"),
    "node4": ("node4_score", "node4_pass",   "mcnemar"),
    "node5": ("node5_score", None,           "wilcoxon"),
}

# N/A cells: SD v1.5 Node 4 — architecturally undefined
NA_CELLS = {("node4", "sd15")}

# Pre-registered RQ2 family: overall compliance_score per (language, model)
# Separately identified from the per-node exploratory family
COMPLIANCE_COL = "compliance_score"


# ── Cluster Bootstrap (BCa) ───────────────────────────────────────────────────

def cluster_bootstrap_bca(
    en_vals: np.ndarray,
    lang_vals: np.ndarray,
    brief_ids: np.ndarray,
    B: int = B_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """
    Cluster bootstrap (B resamples on brief_id) for BCa 95% CI on mean CLCG.

    CLCG = mean(en_vals) − mean(lang_vals), computed per resample.
    Returns (point_estimate, bca_ci_low, bca_ci_high).
    """
    rng = np.random.default_rng(seed)
    unique_briefs = np.unique(brief_ids)
    n = len(unique_briefs)

    observed = float(np.mean(en_vals) - np.mean(lang_vals))

    # ── Bootstrap resamples ──
    boot_stats = np.empty(B)
    for i in range(B):
        sampled = rng.choice(unique_briefs, size=n, replace=True)
        b_en, b_lang = [], []
        for brief in sampled:
            mask = brief_ids == brief
            b_en.append(en_vals[mask])
            b_lang.append(lang_vals[mask])
        b_en   = np.concatenate(b_en)
        b_lang = np.concatenate(b_lang)
        boot_stats[i] = np.mean(b_en) - np.mean(b_lang)

    # ── BCa bias correction (z0) ──
    prop_below = np.mean(boot_stats < observed)
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = norm.ppf(prop_below)

    # ── BCa acceleration (a) via jackknife ──
    jack_stats = np.empty(n)
    for i, brief in enumerate(unique_briefs):
        mask = brief_ids != brief
        jack_stats[i] = np.mean(en_vals[mask]) - np.mean(lang_vals[mask])
    jack_mean = np.mean(jack_stats)
    num   = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = float(num / denom) if denom != 0 else 0.0

    # ── BCa quantiles ──
    def _bca_quantile(alpha_side):
        z = norm.ppf(alpha_side)
        adjusted = norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))
        adjusted = np.clip(adjusted, 0.001, 0.999)
        return float(np.percentile(boot_stats, adjusted * 100))

    ci_low  = _bca_quantile(ALPHA / 2)
    ci_high = _bca_quantile(1 - ALPHA / 2)

    return observed, ci_low, ci_high


# ── Wilcoxon Test ─────────────────────────────────────────────────────────────

def run_wilcoxon(
    en_vals: np.ndarray,
    lang_vals: np.ndarray,
) -> tuple[float, float, float, int]:
    """
    Two-sided Wilcoxon signed-rank test on paired (lang − en) differences.
    Returns (statistic W, p_raw, effect_size_r, n_nonzero).
    A significant negative median difference indicates language degradation.
    """
    diffs = lang_vals - en_vals
    nonzero_mask = diffs != 0
    n_nonzero = int(nonzero_mask.sum())

    if n_nonzero < 10:
        log.warning("Wilcoxon: only %d non-zero differences — result unreliable", n_nonzero)
        return np.nan, np.nan, np.nan, n_nonzero

    stat, p = wilcoxon(diffs[nonzero_mask], alternative="two-sided", zero_method="wilcox")

    # Effect size r = |Z| / sqrt(N_nonzero)
    mu_w    = n_nonzero * (n_nonzero + 1) / 4.0
    sigma_w = np.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
    z       = (stat - mu_w) / sigma_w
    r       = float(abs(z) / np.sqrt(n_nonzero))

    return float(stat), float(p), r, n_nonzero


# ── McNemar Test ──────────────────────────────────────────────────────────────

def run_mcnemar(
    en_pass: np.ndarray,
    lang_pass: np.ndarray,
) -> tuple[float, float, float, int]:
    """
    McNemar test on paired binary pass/fail data.
    Uses exact binomial when n_discordant < 25, chi-squared with Yates
    continuity correction otherwise.
    Returns (statistic, p_raw, effect_size_phi, n_discordant).
    """
    en_p   = en_pass.astype(bool)
    lang_p = lang_pass.astype(bool)

    b = int(np.sum(en_p & ~lang_p))   # EN pass, Lang fail
    c = int(np.sum(~en_p & lang_p))   # EN fail, Lang pass
    n_discordant = b + c

    if n_discordant == 0:
        return np.nan, 1.0, 0.0, 0

    # Build 2×2 table: [[both_pass, en_pass_lang_fail], [en_fail_lang_pass, both_fail]]
    both_pass = int(np.sum(en_p & lang_p))
    both_fail = int(np.sum(~en_p & ~lang_p))
    table = np.array([[both_pass, b], [c, both_fail]])

    exact = n_discordant < 25
    result = mcnemar_test(table, exact=exact, correction=True)
    stat   = float(result.statistic) if not exact else np.nan
    p      = float(result.pvalue)

    # Effect size phi = (b − c) / sqrt(n_discordant)
    phi = float((b - c) / np.sqrt(n_discordant))

    return stat, p, phi, n_discordant


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_cell(
    en_df:   pd.DataFrame,
    lang_df: pd.DataFrame,
    score_col: str,
    pass_col:  str | None,
    test_type: str,
    node:     str,
    language: str,
    model:    str,
) -> dict:
    """Run bootstrap + statistical test for one (node, language, model) cell."""
    # Align by brief_id
    merged = en_df[["brief_id", score_col]].merge(
        lang_df[["brief_id", score_col]].rename(columns={score_col: f"{score_col}_lang"}),
        on="brief_id",
    )
    en_vals   = merged[score_col].values.astype(float)
    lang_vals = merged[f"{score_col}_lang"].values.astype(float)
    brief_ids = merged["brief_id"].values

    # Bootstrap BCa CI
    estimate, ci_low, ci_high = cluster_bootstrap_bca(en_vals, lang_vals, brief_ids)

    result = {
        "node":       node,
        "language":   language,
        "model":      model,
        "test_type":  test_type,
        "n_pairs":    len(merged),
        "clcg_estimate": round(estimate, 4),
        "bca_ci_low":    round(ci_low,   4),
        "bca_ci_high":   round(ci_high,  4),
        "statistic":     np.nan,
        "p_raw":         np.nan,
        "effect_size":   np.nan,
        "n_nonzero_or_discordant": np.nan,
        "node4_clcg_interpretable": (node, model) not in NA_CELLS,
    }

    if test_type == "wilcoxon":
        stat, p, r, n_nz = run_wilcoxon(en_vals, lang_vals)
        result["statistic"] = round(stat, 4) if not np.isnan(stat) else np.nan
        result["p_raw"]     = round(p,    6) if not np.isnan(p)    else np.nan
        result["effect_size"] = round(r,  4) if not np.isnan(r)    else np.nan
        result["n_nonzero_or_discordant"] = n_nz

    elif test_type == "mcnemar" and pass_col is not None:
        # Align pass columns
        en_p_col   = pass_col
        lang_p_col = pass_col + "_lang"
        m2 = en_df[["brief_id", pass_col]].merge(
            lang_df[["brief_id", pass_col]].rename(columns={pass_col: lang_p_col}),
            on="brief_id",
        )
        en_pass_arr   = m2[en_p_col].values.astype(bool)
        lang_pass_arr = m2[lang_p_col].values.astype(bool)

        stat, p, phi, n_disc = run_mcnemar(en_pass_arr, lang_pass_arr)
        result["statistic"] = round(stat, 4) if not np.isnan(stat) else np.nan
        result["p_raw"]     = round(p,    6) if not np.isnan(p)    else np.nan
        result["effect_size"] = round(phi, 4) if not np.isnan(phi)  else np.nan
        result["n_nonzero_or_discordant"] = n_disc

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # --- Validate inputs ---
    for fpath in [C_MULTILINGUAL, C_ENGLISH]:
        if not fpath.exists():
            log.error("Missing input: %s", fpath)
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    log.info("Loading Config C evaluation data...")
    c_ml = pd.read_csv(C_MULTILINGUAL)
    c_en = pd.read_csv(C_ENGLISH)

    log.info("Multilingual: %d rows | English: %d rows", len(c_ml), len(c_en))

    records = []

    # ═══════════════════════════════════════════════════════════════════════
    # FAMILY 1 — Pre-registered RQ2: overall compliance_score (6 tests)
    # ═══════════════════════════════════════════════════════════════════════
    log.info("Running pre-registered RQ2 family (overall compliance_score, 6 tests)...")

    for model in MODELS:
        en_sub = c_en[c_en["model"] == model].copy()
        for lang in LANGUAGES:
            lang_sub = c_ml[(c_ml["model"] == model) & (c_ml["language"] == lang)].copy()

            rec = analyse_cell(
                en_sub, lang_sub,
                score_col  = COMPLIANCE_COL,
                pass_col   = None,
                test_type  = "wilcoxon",
                node       = "compliance_overall",
                language   = lang,
                model      = model,
            )
            rec["hypothesis_family"] = "rq2_preregistered"
            records.append(rec)
            log.info(
                "  compliance/%s/%s: CLCG=%.4f  p_raw=%.4f",
                model, lang, rec["clcg_estimate"], rec["p_raw"],
            )

    # ═══════════════════════════════════════════════════════════════════════
    # FAMILY 2 — Exploratory heatmap: per-node (27 interpretable tests)
    # ═══════════════════════════════════════════════════════════════════════
    log.info("Running exploratory heatmap family (per-node, up to 27 tests)...")

    for model in MODELS:
        en_sub = c_en[c_en["model"] == model].copy()
        for lang in LANGUAGES:
            lang_sub = c_ml[(c_ml["model"] == model) & (c_ml["language"] == lang)].copy()

            for node, (score_col, pass_col, test_type) in NODES.items():

                # Skip N/A cells
                if (node, model) in NA_CELLS:
                    log.info("  %s/%s/%s: N/A — skipped", node, model, lang)
                    continue

                rec = analyse_cell(
                    en_sub, lang_sub,
                    score_col  = score_col,
                    pass_col   = pass_col,
                    test_type  = test_type,
                    node       = node,
                    language   = lang,
                    model      = model,
                )
                rec["hypothesis_family"] = "exploratory_heatmap"
                records.append(rec)
                log.info(
                    "  %s/%s/%s: CLCG=%.4f  p_raw=%.4f  effect=%.4f",
                    node, model, lang,
                    rec["clcg_estimate"],
                    rec["p_raw"] if not np.isnan(rec["p_raw"]) else -1,
                    rec["effect_size"] if not np.isnan(rec["effect_size"]) else -1,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Multiple testing corrections
    # ═══════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(records)

    # --- Bonferroni for pre-registered family ---
    rq2_mask = df["hypothesis_family"] == "rq2_preregistered"
    n_rq2    = rq2_mask.sum()
    df.loc[rq2_mask, "p_bonferroni"] = np.minimum(
        df.loc[rq2_mask, "p_raw"] * n_rq2, 1.0
    ).round(6)
    df.loc[~rq2_mask, "p_bonferroni"] = np.nan
    df.loc[rq2_mask, "significant_bonferroni"] = df.loc[rq2_mask, "p_bonferroni"] < ALPHA

    # --- Benjamini-Hochberg for exploratory family ---
    exp_mask = df["hypothesis_family"] == "exploratory_heatmap"
    exp_p    = df.loc[exp_mask, "p_raw"].values

    # Handle NaN p-values (N/A cells already excluded, but guard anyway)
    valid_mask  = ~np.isnan(exp_p)
    p_bh        = np.full(len(exp_p), np.nan)

    if valid_mask.sum() > 0:
        _, p_adj, _, _ = multipletests(
            exp_p[valid_mask], alpha=ALPHA, method="fdr_bh"
        )
        p_bh[valid_mask] = p_adj

    df.loc[exp_mask, "p_bh"] = np.round(p_bh, 6)
    df.loc[~exp_mask, "p_bh"] = np.nan
    df.loc[exp_mask, "significant_bh"] = df.loc[exp_mask, "p_bh"] < ALPHA

    # Fill missing correction columns with NaN for the other family
    if "significant_bonferroni" not in df.columns:
        df["significant_bonferroni"] = np.nan
    if "significant_bh" not in df.columns:
        df["significant_bh"] = np.nan

    # ═══════════════════════════════════════════════════════════════════════
    # Save output
    # ═══════════════════════════════════════════════════════════════════════
    col_order = [
        "hypothesis_family", "node", "language", "model",
        "test_type", "n_pairs", "n_nonzero_or_discordant",
        "clcg_estimate", "bca_ci_low", "bca_ci_high",
        "statistic", "p_raw", "effect_size",
        "p_bonferroni", "significant_bonferroni",
        "p_bh", "significant_bh",
        "node4_clcg_interpretable",
    ]
    df = df[col_order]
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved: %s  (%d rows)", OUTPUT_FILE, len(df))

    # ═══════════════════════════════════════════════════════════════════════
    # Console summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("FAMILY 1 — Pre-registered RQ2 (Bonferroni, n_tests=6)")
    print("=" * 72)
    rq2 = df[rq2_mask][["language","model","clcg_estimate","bca_ci_low","bca_ci_high","p_raw","p_bonferroni","significant_bonferroni"]]
    print(rq2.to_string(index=False))

    print("\n" + "=" * 72)
    print("FAMILY 2 — Exploratory Heatmap (BH, n_tests=%d)" % exp_mask.sum())
    print("=" * 72)
    exp = df[exp_mask][["node","language","model","test_type","clcg_estimate","bca_ci_low","bca_ci_high","p_raw","effect_size","p_bh","significant_bh"]]
    print(exp.to_string(index=False))

    print("\n" + "=" * 72)
    n_sig_rq2 = int(df.loc[rq2_mask, "significant_bonferroni"].sum())
    n_sig_bh  = int(df.loc[exp_mask, "significant_bh"].infer_objects(copy=False).fillna(False).sum())
    print(f"Summary: {n_sig_rq2}/{n_rq2} pre-registered RQ2 tests significant (Bonferroni α={ALPHA})")
    print(f"Summary: {n_sig_bh}/{exp_mask.sum()} exploratory heatmap tests significant (BH α={ALPHA})")
    print("=" * 72)


if __name__ == "__main__":
    main()