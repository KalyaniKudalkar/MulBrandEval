"""
phase3b_error_analysis/phase3b_disagreement_analysis.py

Node-level error analysis on the cases where automated (Config C) compliance
scoring and human primary-rater judgment diverge most, per the exposé's
"difference exceeding one Likert point" selection rule.

Uses Config C (flat pipeline, corrected ground truth) rather than Config B,
since Config C is the fairer per-node comparison -- Config B's short-circuit
zero-imputation would conflate a genuine node-level disagreement with the
already-documented Deflation Artifact.

HUMAN_COMPLIANT_CUTOFF = 3 is not an independent choice: it is the DAG's own
COMPLIANCE_THRESHOLD = 0.5 expressed on the 1-5 Likert scale
(0.5 * 4 + 1 = 3), so "DAG compliant" and "human compliant" are held to the
same standard.

Two modes:

  select  Finds every case with |human_overall - dag_likert| > 1 and splits
          it two ways:
            - Compliance-side disagreement (DAG and human land on opposite
              sides of the cutoff) is auto-tagged false_positive or
              false_negative -- this is a factual, unambiguous determination
              given the locked cutoff, not a judgment call.
            - Same-side disagreement (DAG and human agree compliant vs.
              non-compliant, but still differ by more than a Likert point in
              degree) does NOT fit false_positive or false_negative by
              definition, and is left with an empty error_type for manual
              review -- this is where you decide whether the root cause is
              genuinely ambiguous brief wording (ambiguous_input) or
              something else worth noting.
          A suggested_primary_node column (the node with the lowest
          Config C per-node score among nodes 1-4) is included as a
          starting point for your review, not an authoritative answer --
          overwrite it if your own reading of the case disagrees.
          Output: results/phase3b/phase3b_disagreement_candidates.csv

  report  Reads your manually completed
          results/phase3b/phase3b_disagreement_candidates_labeled.csv
          (same file, error_type blanks filled in, notes column optionally
          used), prints the error_type distribution by language and by
          suggested_primary_node for Section 6.8, and saves each printed
          table as its own CSV so the numbers don't only exist as terminal
          output:
            results/phase3b/phase3b_error_type_distribution.csv
            results/phase3b/phase3b_error_by_language.csv
            results/phase3b/phase3b_error_by_node.csv
            results/phase3b/phase3b_error_by_language_node.csv

Run from project root:
    python -m phase3b_error_analysis.phase3b_disagreement_analysis select
    python -m phase3b_error_analysis.phase3b_disagreement_analysis report
"""
import sys

import pandas as pd

INPUT_PATH = "results/phase3a/phase3a_120_primary_rater_joined.csv"
CANDIDATES_PATH = "results/phase3b/phase3b_disagreement_candidates.csv"
LABELED_PATH = "results/phase3b/phase3b_disagreement_candidates_labeled.csv"
OUTPUT_DIR = "results/phase3b"

LIKERT_DISAGREEMENT_THRESHOLD = 1
HUMAN_COMPLIANT_CUTOFF = 3

NODE_MAP = {
    "objects": "node1_score_configC",
    "spatial": "node2_score_configC",
    "colour": "node3_score_configC",
    "typography": "node4_score_configC",
}


def select_candidates():
    df = pd.read_csv(INPUT_PATH)

    df["dag_likert_configC"] = df["compliance_score_configC"] * 4 + 1
    df["disagreement"] = (df["overall"] - df["dag_likert_configC"]).abs()
    df["dag_compliant"] = df["dag_likert_configC"] >= HUMAN_COMPLIANT_CUTOFF
    df["human_compliant"] = df["overall"] >= HUMAN_COMPLIANT_CUTOFF

    candidates = df[df["disagreement"] > LIKERT_DISAGREEMENT_THRESHOLD].copy()

    def suggest_primary_node(row):
        node_scores = {dim: row[col] for dim, col in NODE_MAP.items() if pd.notna(row[col])}
        if not node_scores:
            return None
        return min(node_scores, key=node_scores.get)

    def auto_tag(row):
        if row["dag_compliant"] and not row["human_compliant"]:
            return "false_positive"
        if not row["dag_compliant"] and row["human_compliant"]:
            return "false_negative"
        return ""  # same-side disagreement -- your manual call

    candidates["suggested_primary_node"] = candidates.apply(suggest_primary_node, axis=1)
    candidates["error_type"] = candidates.apply(auto_tag, axis=1)
    candidates["notes"] = ""

    output_cols = [
        "rating_id", "brief_id", "model", "language", "output_path",
        "overall", "objects", "spatial", "colour", "typography",
        "compliance_score_configC", "dag_likert_configC", "disagreement",
        "dag_compliant", "human_compliant",
        "node1_score_configC", "node2_score_configC", "node3_score_configC", "node4_score_configC",
        "suggested_primary_node", "error_type", "notes",
    ]
    candidates = candidates[output_cols].sort_values("disagreement", ascending=False)

    candidates.to_csv(CANDIDATES_PATH, index=False)

    n_fp = (candidates["error_type"] == "false_positive").sum()
    n_fn = (candidates["error_type"] == "false_negative").sum()
    n_blank = (candidates["error_type"] == "").sum()

    print(f"Total disagreement cases (>|{LIKERT_DISAGREEMENT_THRESHOLD}| Likert points): {len(candidates)} / {len(df)}")
    print(f"  Auto-tagged false_positive: {n_fp}")
    print(f"  Auto-tagged false_negative: {n_fn}")
    print(f"  Left blank for manual review (same-side disagreement): {n_blank}")
    print(f"\nSaved -> {CANDIDATES_PATH}")
    print(f"\nNext: fill in the {n_blank} blank error_type cells (false_positive / false_negative / "
          f"ambiguous_input -- your call per case), save as {LABELED_PATH}, then run report mode.")


def report():
    df = pd.read_csv(LABELED_PATH)

    if df["error_type"].isna().any() or (df["error_type"] == "").any():
        n_missing = df["error_type"].isna().sum() + (df["error_type"] == "").sum()
        print(f"WARNING: {n_missing} row(s) still have an empty error_type -- report will still run, "
              f"but these rows are excluded from the distributions below.")

    labeled = df[df["error_type"].notna() & (df["error_type"] != "")]

    print(f"\nTotal labeled cases: {len(labeled)} / {len(df)}\n")

    error_type_dist = labeled["error_type"].value_counts()
    print("=== Distribution by error_type ===")
    print(error_type_dist.to_string())

    by_language = pd.crosstab(labeled["language"], labeled["error_type"])
    print("\n=== By language ===")
    print(by_language.to_string())

    by_node = pd.crosstab(labeled["suggested_primary_node"], labeled["error_type"])
    print("\n=== By suggested_primary_node ===")
    print(by_node.to_string())

    by_language_node = pd.crosstab(
        [labeled["language"], labeled["suggested_primary_node"]], labeled["error_type"]
    )
    print("\n=== By language x suggested_primary_node (error_type counts) ===")
    print(by_language_node.to_string())

    error_type_dist.to_csv(f"{OUTPUT_DIR}/phase3b_error_type_distribution.csv", header=["count"])
    by_language.to_csv(f"{OUTPUT_DIR}/phase3b_error_by_language.csv")
    by_node.to_csv(f"{OUTPUT_DIR}/phase3b_error_by_node.csv")
    by_language_node.to_csv(f"{OUTPUT_DIR}/phase3b_error_by_language_node.csv")

    print(f"\nSaved four summary tables -> {OUTPUT_DIR}/phase3b_error_*.csv")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("select", "report"):
        print("Usage: python -m phase3b_error_analysis.phase3b_disagreement_analysis [select|report]")
        sys.exit(1)

    if sys.argv[1] == "select":
        select_candidates()
    else:
        report()


if __name__ == "__main__":
    main()