"""
phase3a_human_eval/phase3a_failure_categorization.py

Checks how well the pipeline's per-node pass/fail calls and its
failure_diagnosis text agree with the primary rater's Likert sub-scores,
using Config C (all nodes always scored, no short-circuit skips).

Per-node accuracy: human sub-score <= HUMAN_FAIL_THRESHOLD treated as
"failed", compared against the pipeline's node*_pass flag.

Primary-node match: on non-compliant images only, compares the human's
lowest-scoring sub-dimension against two different readings of the
pipeline's primary failing node — (1) the first node named in the
failure_diagnosis text, and (2) the node with the lowest score. These can
diverge because failure_diagnosis lists nodes in a fixed order rather than
severity order.

Run from project root:
    python -m phase3a_human_eval.phase3a_failure_categorization
"""
import re
import pandas as pd

INPUT_PATH = "results/phase3a/phase3a_120_primary_rater_joined.csv"
OUTPUT_PATH = "results/phase3a/phase3a_failure_categorization_results.csv"

HUMAN_FAIL_THRESHOLD = 2  # human sub-score <= this value counts as "failed"

NODE_MAP = {
    "objects": ("node1_pass_configC", "node1_score_configC", 1),
    "spatial": ("node2_pass_configC", "node2_score_configC", 2),
    "colour": ("node3_pass_configC", "node3_score_configC", 3),
    "typography": ("node4_pass_configC", "node4_score_configC", 4),
}


def pipeline_first_mentioned_node(diagnosis: str):
    m = re.search(r"Node(\d)", str(diagnosis))
    return int(m.group(1)) if m else None


def main():
    df = pd.read_csv(INPUT_PATH)

    per_node_rows = []
    for human_col, (pass_col, score_col, node_num) in NODE_MAP.items():
        human_fail = df[human_col] <= HUMAN_FAIL_THRESHOLD
        pipeline_fail = df[pass_col] == False
        accuracy = (human_fail == pipeline_fail).mean()
        per_node_rows.append({
            "node": node_num, "dimension": human_col, "accuracy": accuracy,
            "both_fail": int((human_fail & pipeline_fail).sum()),
            "both_pass": int((~human_fail & ~pipeline_fail).sum()),
            "human_fail_only": int((human_fail & ~pipeline_fail).sum()),
            "pipeline_fail_only": int((~human_fail & pipeline_fail).sum()),
        })
    per_node_df = pd.DataFrame(per_node_rows)

    def human_primary(row):
        subs = {n: row[n] for n in NODE_MAP}
        return NODE_MAP[min(subs, key=subs.get)][2]

    df["human_primary_node"] = df.apply(human_primary, axis=1)

    df["pipeline_primary_node_firstmention"] = df["failure_diagnosis_configC"].apply(
        pipeline_first_mentioned_node
    )

    def pipeline_lowest_score_node(row):
        scores = {node_num: row[score_col] for _, (_, score_col, node_num) in NODE_MAP.items()}
        return min(scores, key=scores.get)
    df["pipeline_primary_node_lowestscore"] = df.apply(pipeline_lowest_score_node, axis=1)

    noncompliant = df[df["compliant"] == False]
    match_firstmention = (
        noncompliant["human_primary_node"] == noncompliant["pipeline_primary_node_firstmention"]
    ).mean()
    match_lowestscore = (
        noncompliant["human_primary_node"] == noncompliant["pipeline_primary_node_lowestscore"]
    ).mean()

    print(per_node_df.to_string(index=False))
    print(f"\nMean per-node accuracy: {per_node_df['accuracy'].mean():.3f}")
    print(f"Primary-node match (first-mention definition, n={len(noncompliant)}): {match_firstmention:.3f}")
    print(f"Primary-node match (lowest-score definition, n={len(noncompliant)}): {match_lowestscore:.3f}")

    per_node_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()