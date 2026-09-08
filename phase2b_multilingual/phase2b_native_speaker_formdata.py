"""
phase2b_native_speaker_formdata.py

Generates the single JS data file (Data_NativeSpeaker.gs.txt) consumed by
Generator.gs to programmatically build ONE Layer 3 native-speaker validation
Google Form, with a language-selector question that branches the rater
straight to their language's 10 items via Section navigation.

Each language contributes 10 items total (2 per category x 5 categories),
sampled deterministically from the interquartile range of cometkiwi_score
within each category, restricted to PASS-status rows. There is no
attention-check item and no rater-facing category grouping — the rater
only ever sees their language's 10 items, one after another, with no
mention of what category each one came from.

USAGE
-----
From the project root (conda env: mulbrandeval, which already has pandas):

    python -m phase2b_multilingual.phase2b_native_speaker_formdata

or, if run directly as a script:

    python phase2b_native_speaker_formdata.py

INPUT
-----
Expects the four CSVs at data/prompts/mbb/mbb_briefs_{lang}.csv, matching
the canonical path convention already used for mbb_briefs_en.csv elsewhere
in this project. Override with --input-dir if your files live elsewhere.

Required columns per CSV: brief_id, industry, status, cometkiwi_score,
prompt_text, prompt_text_en. (The translated files also carry brand_name,
back_translated, cc_score, etc. — those extra columns are ignored here.)

Confirmed actual `status` values in the current CSVs: PASS / REVIEW / FAIL.
Only PASS rows are eligible for sampling — a rater's first exposure to
this work should not be a translation you already know is flagged.

SAMPLING LOGIC (per language, per category)
--------------------------------------------
1. Filter to status == "PASS".
2. Sort the category's PASS rows by cometkiwi_score.
3. Take the interquartile range (middle 50%, dropping the top and bottom
   quartiles) as the "representative" pool — this avoids both cherry-picking
   your best translations (which would bias the human validation toward an
   artificially rosy result) and risking a genuine low-quality outlier
   landing in front of a rater by pure chance.
4. Deterministically sample N_ITEMS_PER_CATEGORY (default 2) items from
   that IQR pool using a fixed seed derived from (language, category).

Re-running this script against unchanged input CSVs always reproduces the
identical 10-item set per language — methodology can state the sampling is
deterministic, not "we eyeballed it."

OUTPUT
------
Writes ONE consolidated file to --output-dir (default: results/phase2b/):

    Data_NativeSpeaker.gs.txt

Defining a single JS constant:

    var BRIEF_DATA = {
      "de": [ {brief_id, category, text, text_en}, ... 10 items ],
      "fr": [ ... 10 items ],
      "es": [ ... 10 items ],
      "ar": [ ... 10 items ]
    };

One file (not four) because the new form is a single Apps Script build with
internal language branching, rather than four separate per-language
Form.create() calls — Generator.gs only needs one entry point to see every
language's items. `category` is included per item for your own downstream
analysis but is not intended to be rendered to the rater.
"""

import argparse
import hashlib
import json
from pathlib import Path
from random import Random

import pandas as pd

LANGUAGES = ["de", "fr", "es", "ar"]

# New spec: 2 items/category x 5 categories = 10 items/language (was 10+1
# attention-check per category = 11/category, 55/language).
N_ITEMS_PER_CATEGORY = 2

# Category order must match the 5 industries used throughout Phase 2B
# (see mbb_briefs_en.csv / component_validation_table.csv for the canonical list).
EXPECTED_CATEGORIES = [
    "tech_product",
    "food_beverage",
    "fashion_apparel",
    "automotive",
    "lifestyle_wellness",
]

PASS_STATUS = "PASS"


def stable_seed(*parts: str) -> int:
    """Deterministic seed derived from the given strings, stable across
    processes, machines, and Python versions.

    Do NOT use Python's built-in hash() for this: since Python 3.3, string
    hashing is deliberately randomized per-process (PYTHONHASHSEED, a
    security feature against hash-flooding attacks), so hash(("de",
    "tech_product")) returns a DIFFERENT number every time you start a new
    python process. That silently breaks reproducibility — two runs against
    identical input CSVs would sample different briefs. sha256 has no such
    randomization: same input string always produces the same digest.
    """
    key = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % (2**32)


def load_briefs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    required_cols = {"brief_id", "industry", "status", "cometkiwi_score", "prompt_text", "prompt_text_en"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} is missing required column(s): {missing_cols}")

    found_categories = set(df["industry"].unique())
    unexpected = found_categories - set(EXPECTED_CATEGORIES)
    if unexpected:
        raise ValueError(f"{csv_path} has unexpected industry values: {unexpected}")
    missing_categories = set(EXPECTED_CATEGORIES) - found_categories
    if missing_categories:
        raise ValueError(f"{csv_path} is missing industry categories: {missing_categories}")

    df["cometkiwi_score"] = df["cometkiwi_score"].astype(float)
    return df


def sample_representative(group: pd.DataFrame, lang: str, category: str, n: int) -> list[dict]:
    """Pick n items from the IQR (middle 50% by cometkiwi_score) of this
    category's PASS-status rows, deterministically."""
    pass_rows = group[group["status"] == PASS_STATUS].sort_values("cometkiwi_score").reset_index(drop=True)

    if len(pass_rows) == 0:
        raise ValueError(f"No PASS-status rows for lang={lang}, category={category} — cannot sample.")

    q1_idx = len(pass_rows) // 4
    q3_idx = (len(pass_rows) * 3) // 4
    iqr_pool = pass_rows.iloc[q1_idx:q3_idx + 1].reset_index(drop=True)

    # Fallback: if PASS rows are too few for a meaningful IQR slice (e.g. a
    # heavily-flagged language), widen the pool to all PASS rows rather than
    # erroring out, but never fall back to REVIEW/FAIL rows.
    if len(iqr_pool) < n:
        iqr_pool = pass_rows

    seed = stable_seed(lang, category)
    rng = Random(seed)
    chosen_idx = rng.sample(range(len(iqr_pool)), min(n, len(iqr_pool)))
    chosen = iqr_pool.iloc[chosen_idx]

    return [
        {
            "brief_id": row["brief_id"],
            "category": category,
            "text": row["prompt_text"],
            "text_en": row["prompt_text_en"],
        }
        for _, row in chosen.iterrows()
    ]


def build_language_items(df: pd.DataFrame, lang: str) -> list[dict]:
    """Build the flat 10-item list (2/category x 5 categories) for one
    language. Flat, not nested by category, since the rater never sees
    categories in the new single-form spec."""
    items: list[dict] = []
    for category in EXPECTED_CATEGORIES:
        group = df[df["industry"] == category].reset_index(drop=True)
        items.extend(sample_representative(group, lang, category, N_ITEMS_PER_CATEGORY))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/prompts/mbb"),
        help="Directory containing mbb_briefs_{lang}.csv files (default: data/prompts/mbb)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2b"),
        help="Directory to write Data_NativeSpeaker.gs.txt into (default: results/phase2b)",
    )
    args = parser.parse_args()

    print(f"Reading briefs from: {args.input_dir.resolve()}")
    print(f"Writing output to:   {args.output_dir.resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_lang_data = {}
    for lang in LANGUAGES:
        csv_path = args.input_dir / f"mbb_briefs_{lang}.csv"
        print(f"Processing {lang} ({csv_path.name})...")
        df = load_briefs(csv_path)
        items = build_language_items(df, lang)

        cat_counts = {}
        for item in items:
            cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
        print(f"  category counts: {cat_counts}")
        print(f"  total items: {len(items)}")

        assert len(items) == N_ITEMS_PER_CATEGORY * len(EXPECTED_CATEGORIES), (
            f"{lang}: expected {N_ITEMS_PER_CATEGORY * len(EXPECTED_CATEGORIES)} items, got {len(items)}"
        )

        all_lang_data[lang] = items

    js_obj = json.dumps(all_lang_data, ensure_ascii=False, indent=1)
    content = (
        "// Auto-generated brief data for the Native-Speaker survey (single form,\n"
        "// language-branched, no attention check).\n"
        "// Generated by phase2b_native_speaker_formdata.py — do not edit by hand.\n"
        "// Regenerate from mbb_briefs_{lang}.csv if brief text or PASS/REVIEW/FAIL\n"
        "// status changes.\n"
        "// NOTE: uses 'var', not 'const' — Apps Script's V8 runtime scopes\n"
        "// top-level const/let to the file they're declared in, so Generator.gs\n"
        "// would not be able to see this constant if declared with const.\n"
        "// 'category' is included per item for your own analysis traceability\n"
        "// only — Generator.gs should not render it to the rater.\n"
        f"var BRIEF_DATA = {js_obj};\n"
    )
    out_path = args.output_dir / "Data_NativeSpeaker.gs.txt"
    out_path.write_text(content, encoding="utf-8")
    print(f"\nWrote consolidated file: {out_path.resolve()}")
    print("Paste its content into a single Apps Script file named Data_NativeSpeaker,")
    print("alongside the updated Generator.gs (single form, language-branching).")


if __name__ == "__main__":
    main()