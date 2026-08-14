"""
patch_config_c_mb056_mb090_node2.py

Configuration B's MB056/MB090 rows were already re-verified against the
corrected spatial_constraints via patch_mb056_mb090_node2.py. Configuration
C (flat pipeline) runs the same Node 2 on the same images, but its stored
MB056/MB090 results were computed under the BUGGY constraint and happened
to score node2_pass=True already. That's a coincidence observed under the
wrong question, not a verified result under the right one — this script
actually asks Node 2 the corrected question and records whatever comes
back, rather than assuming the old value holds.

Unlike Configuration B, Configuration C never short-circuits (that's the
entire point of the flat-pipeline ablation), so Node 2 genuinely runs for
all 4 rows (MB056/MB090 x sd15/flux) regardless of Node 1's outcome — no
short-circuit skip check needed here.

Uses the SAME generated images as Configuration B (per Milestone M2: "the
same 300 images, evaluated four different ways") — same image_path_for
pattern, data/generated_images/phase2a/{model}/en/{brief_id}_seed42.png.

Run from project root, in the mulbrandeval env (needs OPENAI_API_KEY):

    python -m phase2a_dag_pipeline.patch_config_c_mb056_mb090_node2

PREREQUISITE: fix_spatial_constraints.py must already have corrected
data/prompts/mbb/mbb_briefs_en.csv (it did, in the same run that fixed
the file this script's CORRECTED_CONSTRAINTS below duplicates for
directness — no re-read of the CSV needed here, the corrected string is
hardcoded to match exactly what was verified in Configuration B's patch).
"""
import shutil
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from phase2a_dag_pipeline.nodes.node2_spatial_layout import run_node2
from phase2a_dag_pipeline.nodes.node6_coordinator_flat import BASE_WEIGHTS

CSV_PATH    = Path("results/phase2a/configuration_c_flat_pipeline.csv")
BACKUP_PATH = CSV_PATH.with_suffix(".csv.bak")

IMAGE_DIR = Path("data/generated_images/phase2a")

CORRECTED_CONSTRAINTS = {
    "MB056": ["oat milk carton right of ceramic bowl"],
    "MB090": ["leather sandals right of straw beach hat"],
}

MODELS = ["sd15", "flux"]


def image_path_for(model: str, brief_id: str) -> Path:
    return IMAGE_DIR / model / "en" / f"{brief_id}_seed42.png"


def recompute_compliance_score(row: pd.Series, new_node2_score: float) -> float:
    """
    Configuration C never zeroes node2/node3 based on short-circuit (that's
    the whole point of node6_coordinator_flat.py — see its own docstring).
    So every other node's contribution is taken as-is from the existing row.
    """
    node1_score = row["node1_score"]
    node3_score = row["node3_score"]
    node4_score = row["node4_score"]
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

    print("BEFORE re-verification (values computed under the BUGGY constraint —")
    print("these passed, but that was never actually checked against the correct question):")
    before = df[df["brief_id"].isin(CORRECTED_CONSTRAINTS.keys())][
        ["brief_id", "model", "node2_score", "node2_pass", "compliance_score"]
    ]
    print(before.to_string(index=False))
    print()

    any_flip = False

    for brief_id, new_constraint in CORRECTED_CONSTRAINTS.items():
        for model in MODELS:
            mask = (df["brief_id"] == brief_id) & (df["model"] == model)
            if not mask.any():
                print(f"  WARNING — {brief_id}/{model} not found, skipping.")
                continue

            row = df.loc[mask].iloc[0]
            img_path = image_path_for(model, brief_id)
            if not img_path.exists():
                print(f"  WARNING — image not found at {img_path}, skipping.")
                continue

            state = {
                "brief_id": brief_id,
                "image_path": str(img_path),
                "spatial_constraints": new_constraint,
            }
            result = run_node2(state)
            new_score = result["node2_score"]
            new_pass  = result["node2_pass"]

            old_score = row["node2_score"]
            new_compliance = recompute_compliance_score(row, new_score)

            df.loc[mask, "node2_score"]      = new_score
            df.loc[mask, "node2_pass"]       = new_pass
            df.loc[mask, "compliance_score"] = new_compliance

            flipped = (old_score >= 0.5) != (new_score >= 0.5)
            flag = "  <-- FLIPPED, was assumed stable" if flipped else ""
            if flipped:
                any_flip = True

            print(f"  {brief_id}/{model}: node2_score {old_score} -> {new_score}, "
                  f"compliance_score {row['compliance_score']} -> {new_compliance}{flag}")

    print("\nAFTER re-verification:")
    after = df[df["brief_id"].isin(CORRECTED_CONSTRAINTS.keys())][
        ["brief_id", "model", "node2_score", "node2_pass", "compliance_score"]
    ]
    print(after.to_string(index=False))

    if any_flip:
        print("\n*** At least one value FLIPPED from what was assumed. ***")
        print("*** Configuration C's stats MUST be rerun — do not skip that step. ***")
    else:
        print("\nAll 4 values held under the corrected constraint — the earlier "
              "assumption happened to be right, but this is now VERIFIED, not assumed.")

    df.to_csv(CSV_PATH, index=False)
    print(f"\nPatched file written: {CSV_PATH}")
    print("Row count should still be 300 — verify, then rerun "
          "python -m stats.phase2a_statistical_comparison regardless of whether "
          "any value flipped, since Configuration C's compliance_score column "
          "changed (even a same-value overwrite still means the file was rewritten).")


if __name__ == "__main__":
    main()