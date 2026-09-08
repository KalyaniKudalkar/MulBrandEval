"""
phase3a_human_eval/phase3a_correlation_analysis.py

Primary rater evaluation: Spearman and Pearson correlation between each
pipeline configuration's compliance scores and human Likert ratings.

Config A uses clip_score_norm_configA (min-max normalized 0-1, scaled off
the full 900-image pool) so it's on the same scale as human_overall_scaled
for comparability, though Spearman is rank-based and unaffected by scale
either way.

Run from project root:
    python -m phase3a_human_eval.phase3a_correlation_analysis
"""
import pandas as pd
from scipy import stats

INPUT_PATH = "results/phase3a/phase3a_120_primary_rater_joined.csv"
OUTPUT_PATH = "results/phase3a/phase3a_correlation_results.csv"

NODE_MAP = {
    "objects": "node1_score",
    "spatial": "node2_score",
    "colour": "node3_score",
    "typography": "node4_score",
}


def correlate_overall(df, compliance_col, label):
    rows = []
    spearman = stats.spearmanr(df["overall"], df[compliance_col])
    pearson = stats.pearsonr(df["overall"], df[compliance_col])
    rows.append({
        "config": label, "dimension": "overall_vs_compliance",
        "n": len(df), "spearman_rho": spearman.correlation, "spearman_p": spearman.pvalue,
        "pearson_r": pearson[0], "pearson_p": pearson[1],
    })
    return rows


def correlate_per_node(df, node_suffix, label):
    rows = []
    for human_col, node_col_base in NODE_MAP.items():
        node_col = f"{node_col_base}{node_suffix}"
        sub = df.dropna(subset=[node_col])
        if len(sub) < 5:
            continue
        r = stats.spearmanr(sub[human_col], sub[node_col])
        rows.append({
            "config": label, "dimension": human_col,
            "n": len(sub), "spearman_rho": r.correlation, "spearman_p": r.pvalue,
            "pearson_r": None, "pearson_p": None,
        })
    return rows


def main():
    df = pd.read_csv(INPUT_PATH)
    results = []

    # Config A — overall only, no per-node breakdown (single-number baseline)
    results += correlate_overall(df, "clip_score_norm_configA", "Config A (CLIP baseline)")

    # Config C and B — overall plus per-node, as before
    results += correlate_overall(df, "compliance_score_configC", "Config C (flat)")
    results += correlate_per_node(df, "_configC", "Config C (flat)")

    results += correlate_overall(df, "compliance_score_configB", "Config B (operational)")
    results += correlate_per_node(df, "_configB", "Config B (operational)")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()