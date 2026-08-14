"""
patch_node4_clcg_interpretable.py

One-time fix for the node6_coordinator.py bug: node4_clcg_interpretable
was incorrectly True for SD v1.5 rows that short-circuited at Node 1
(should always be False for SD v1.5, regardless of short-circuit).

This patches results/phase2a/english_ablation_table.csv IN PLACE by
recomputing the single affected column directly from data already in
the CSV (model, node1_short_circuit) — no pipeline re-run, no API
calls, no risk of GPT-4o mini non-determinism shifting any other value.

A .bak backup of the original file is written first.

Run from project root:
    python patch_node4_clcg_interpretable.py
"""
import shutil
from pathlib import Path
import pandas as pd

CSV_PATH = Path("results/phase2a/english_ablation_table.csv")
BACKUP_PATH = CSV_PATH.with_suffix(".csv.bak")

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found — run this from project root.")

    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"Backup written: {BACKUP_PATH}")

    df = pd.read_csv(CSV_PATH)

    bad_before = (
        (df["model"] == "sd15")
        & (df["node1_short_circuit"] == True)
        & (df["node4_clcg_interpretable"] == True)
    ).sum()
    print(f"Rows with the bug before patch: {bad_before}")

    df["node4_clcg_interpretable"] = df["model"] != "sd15"

    bad_after = (
        (df["model"] == "sd15")
        & (df["node1_short_circuit"] == True)
        & (df["node4_clcg_interpretable"] == True)
    ).sum()
    print(f"Rows with the bug after patch:  {bad_after}")

    assert bad_after == 0, "Patch failed — bug still present."

    df.to_csv(CSV_PATH, index=False)
    print(f"Patched file written: {CSV_PATH}")
    print("Done. Verify row count is still 300, then commit both the CSV "
          "and the earlier node6_coordinator.py fix together.")

if __name__ == "__main__":
    main()