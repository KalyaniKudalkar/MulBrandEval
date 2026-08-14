"""
patch_mb056_mb090_node2.py

Re-checks Node 2 for MB056 and MB090 (both models: sd15, flux) against the
CORRECTED spatial_constraints ("right of" instead of the buggy "left of"),
using the images already generated in Phase 2A. Patches
results/phase2a/english_ablation_table.csv in place with the true
node2_score / node2_pass / compliance_score / failure_diagnosis for these
4 rows only. Every other row is untouched.

PREREQUISITE: run fix_spatial_constraints.py FIRST, so
data/prompts/mbb/mbb_briefs_en.csv (or mbb_briefs_en_v3_FINAL.csv, whichever
your generate/evaluate scripts actually read from) already has the
corrected constraint text before this script builds its state dicts.

This does NOT re-run Nodes 1/3/4/5 — those were never affected by this bug
(only Node 2's constraint string was wrong), so re-running them would only
burn API credits and risk GPT-4o mini's known non-determinism changing an
otherwise-correct value for no reason.

Run from project root, in the mulbrandeval env (needs OPENAI_API_KEY):

    python patch_mb056_mb090_node2.py
"""
import shutil
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # must run before node2_spatial_layout's OpenAI client is created

from phase2a_dag_pipeline.nodes.node2_spatial_layout import run_node2
from phase2a_dag_pipeline.nodes.node6_coordinator import BASE_WEIGHTS

CSV_PATH    = Path("results/phase2a/english_ablation_table.csv")
BACKUP_PATH = CSV_PATH.with_suffix(".csv.bak2")  # .bak already used by the node4 patch

IMAGE_DIR = Path("data/generated_images/phase2a")  # {model}/en/{brief_id}_seed42.png — adjust if your naming differs

# Corrected constraints (must match what fix_spatial_constraints.py wrote)
CORRECTED_CONSTRAINTS = {
    "MB056": ["oat milk carton right of ceramic bowl"],
    "MB090": ["leather sandals right of straw beach hat"],
}

MODELS = ["sd15", "flux"]


def image_path_for(model: str, brief_id: str) -> Path:
    # Adjust this if your actual Phase 2A naming convention differs —
    # check against a known-good path from phase2a_generate.py before running.
    return IMAGE_DIR / model / "en" / f"{brief_id}_seed42.png"


def recompute_compliance_score(row: pd.Series, new_node2_score: float) -> float:
    """
    Recompute compliance_score with the corrected node2_score, leaving every
    other node's contribution exactly as it already was in the row. Mirrors
    node6_coordinator.py's weighted-sum logic exactly (BASE_WEIGHTS import
    guarantees the weights can't silently drift out of sync).
    """
    node1_score = row["node1_score"]
    node3_score = row["node3_score"] if not row["node1_short_circuit"] else 0.0
    node4_score = row["node4_score"]  # already correctly 0.0-or-real per existing logic
    node5_score = row["node5_score"] if pd.notna(row["node5_score"]) else 0.0

    return round(
        node1_score * BASE_WEIGHTS["node1"] +
        new_node2_score * BASE_WEIGHTS["node2"] +
        node3_score * BASE_WEIGHTS["node3"] +
        node4_score * BASE_WEIGHTS["node4"] +
        node5_score * BASE_WEIGHTS["node5"],
        4,
    )


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found — run from project root.")

    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"Backup written: {BACKUP_PATH}\n")

    df = pd.read_csv(CSV_PATH)

    print("BEFORE patch:")
    before = df[df["brief_id"].isin(CORRECTED_CONSTRAINTS.keys())][
        ["brief_id", "model", "node2_score", "node2_pass", "compliance_score"]
    ]
    print(before.to_string(index=False))
    print()

    for brief_id, new_constraint in CORRECTED_CONSTRAINTS.items():
        for model in MODELS:
            mask = (df["brief_id"] == brief_id) & (df["model"] == model)
            if not mask.any():
                print(f"  WARNING — {brief_id}/{model} not found in CSV, skipping.")
                continue

            row = df.loc[mask].iloc[0]

            # Short-circuited rows never had Node 2 run for real (it was
            # forced to 0.0 by the coordinator) — no correction needed,
            # the constraint bug is irrelevant if Node 2 never executed.
            if row["node1_short_circuit"]:
                print(f"  {brief_id}/{model}: node1_short_circuit=True — "
                      f"Node 2 never ran, nothing to correct.")
                continue

            img_path = image_path_for(model, brief_id)
            if not img_path.exists():
                print(f"  WARNING — image not found at {img_path} for "
                      f"{brief_id}/{model}, skipping. Check IMAGE_DIR / naming.")
                continue

            state = {
                "brief_id": brief_id,
                "image_path": str(img_path),
                "spatial_constraints": new_constraint,
            }
            result = run_node2(state)
            new_score = result["node2_score"]
            new_pass  = result["node2_pass"]

            new_compliance = recompute_compliance_score(row, new_score)

            df.loc[mask, "node2_score"]      = new_score
            df.loc[mask, "node2_pass"]       = new_pass
            df.loc[mask, "compliance_score"] = new_compliance

            print(f"  {brief_id}/{model}: node2_score {row['node2_score']} -> {new_score}, "
                  f"compliance_score {row['compliance_score']} -> {new_compliance}")

    print("\nAFTER patch:")
    after = df[df["brief_id"].isin(CORRECTED_CONSTRAINTS.keys())][
        ["brief_id", "model", "node2_score", "node2_pass", "compliance_score"]
    ]
    print(after.to_string(index=False))

    df.to_csv(CSV_PATH, index=False)
    print(f"\nPatched file written: {CSV_PATH}")
    print("Verify row count is still 300, then commit this alongside the "
          "spatial_constraints fix and a one-line deviations-log entry, "
          "same pattern as the node4_clcg_interpretable patch.")


if __name__ == "__main__":
    main()