"""
phase3a_human_eval/phase3a_ablation_wilcoxon.py

Wilcoxon signed-rank tests on each pipeline configuration's absolute
disagreement from human ratings (Configuration A vs B, Configuration B
vs C), using the same wilcoxon_with_effect_size() engine from
stats/bootstrap_testing.py applied throughout this thesis.

Config A's raw clip_score, Config B/C's compliance_score, and human
Likert ratings are not on comparable scales, so a direct Wilcoxon test
between raw config outputs would only detect calibration differences,
not diagnostic accuracy. Instead, each config's ABSOLUTE DISAGREEMENT
from the human rating is computed on the shared 0-1 scale, and Wilcoxon
tests whether one config's disagreement is systematically smaller than
another's.

Run from project root:
    python -m phase3a_human_eval.phase3a_ablation_wilcoxon
"""
import pandas as pd

from stats.bootstrap_testing import wilcoxon_with_effect_size

INPUT_PATH = "results/phase3a/phase3a_120_primary_rater_joined.csv"
OUTPUT_PATH = "results/phase3a/phase3a_ablation_wilcoxon.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["human_overall_scaled"] = (df["overall"] - 1) / 4

    df["disagreement_configA"] = (df["human_overall_scaled"] - df["clip_score_norm_configA"]).abs()
    df["disagreement_configB"] = (df["human_overall_scaled"] - df["compliance_score_configB"]).abs()
    df["disagreement_configC"] = (df["human_overall_scaled"] - df["compliance_score_configC"]).abs()

    print("Mean absolute disagreement from human rating:")
    for cfg in ["configA", "configB", "configC"]:
        print(f"  {cfg}: {df[f'disagreement_{cfg}'].mean():.4f}")

    results = []
    for a, b, label in [("configA", "configB", "Config A vs Config B"),
                         ("configB", "configC", "Config B vs Config C")]:
        result = wilcoxon_with_effect_size(
            df[f"disagreement_{a}"].to_numpy(), df[f"disagreement_{b}"].to_numpy()
        )
        results.append({
            "comparison": label,
            "n_pairs": result["n_pairs"],
            "wilcoxon_statistic": result["statistic"],
            "p_value": result["p_value"],
            "effect_size_r": result["effect_size_r"],
            "mean_disagreement_first": df[f"disagreement_{a}"].mean(),
            "mean_disagreement_second": df[f"disagreement_{b}"].mean(),
        })
        print(f"\n{label}:")
        print(f"  Wilcoxon statistic={result['statistic']:.4f}, p={result['p_value']:.6f}, "
              f"effect_r={result['effect_size_r']:.4f}, n_pairs={result['n_pairs']}")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()