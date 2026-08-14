"""
fix_spatial_constraints.py

Corrects two direction-contradiction bugs found by check_spatial_v2.py:
  - MB056: spatial_constraints said "oat milk carton left of ceramic bowl"
           but prompt_text says "right of" — constraint corrected to match
           the prompt (the prompt is what actually gets rendered).
  - MB090: spatial_constraints said "leather sandals left of straw beach hat"
           but prompt_text says "right of" — same correction.

Applies the fix to all five brief files: the English master
(mbb_briefs_en_v3_FINAL.csv) plus all four translated files
(mbb_briefs_{de,fr,es,ar}.csv), since spatial_constraints is copied
unchanged into every language file.

Writes a .bak backup of each file before editing. Run from project root:

    python fix_spatial_constraints.py
"""
import shutil
from pathlib import Path
import pandas as pd

FILES = [
    "data/prompts/mbb/mbb_briefs_en.csv",   # renamed copy of v3_FINAL — kept as the canonical path
    "data/prompts/mbb/mbb_briefs_de.csv",
    "data/prompts/mbb/mbb_briefs_fr.csv",
    "data/prompts/mbb/mbb_briefs_es.csv",
    "data/prompts/mbb/mbb_briefs_ar.csv",
]

# brief_id -> (old_constraint_string, new_constraint_string)
FIXES = {
    "MB056": (
        '["oat milk carton left of ceramic bowl"]',
        '["oat milk carton right of ceramic bowl"]',
    ),
    "MB090": (
        '["leather sandals left of straw beach hat"]',
        '["leather sandals right of straw beach hat"]',
    ),
}


def fix_file(path_str: str):
    path = Path(path_str)
    if not path.exists():
        print(f"  SKIPPED — not found: {path}")
        return

    backup = path.with_suffix(".csv.bak")
    shutil.copy2(path, backup)

    df = pd.read_csv(path)
    if "brief_id" not in df.columns or "spatial_constraints" not in df.columns:
        print(f"  SKIPPED — missing expected columns: {path}")
        return

    changed = 0
    for brief_id, (old_val, new_val) in FIXES.items():
        mask = df["brief_id"] == brief_id
        if not mask.any():
            print(f"  WARNING — {brief_id} not found in {path}")
            continue

        current = df.loc[mask, "spatial_constraints"].iloc[0]
        if current != old_val:
            print(f"  WARNING — {brief_id} in {path} has unexpected value "
                  f"'{current}' (expected '{old_val}') — NOT overwritten, "
                  f"check manually.")
            continue

        df.loc[mask, "spatial_constraints"] = new_val
        changed += 1

    df.to_csv(path, index=False)
    print(f"  {path}: {changed}/{len(FIXES)} briefs corrected. Backup: {backup}")


def main():
    print("Fixing spatial_constraints direction bugs (MB056, MB090)\n")
    for f in FILES:
        fix_file(f)
    print("\nDone. Diff each .bak against its corrected file to confirm before committing.")


if __name__ == "__main__":
    main()