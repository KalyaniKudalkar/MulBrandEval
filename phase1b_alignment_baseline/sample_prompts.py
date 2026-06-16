"""
MulBrandEval — Phase 1B: Prompt Sampling (DrawBench + GenAI-Bench)
==================================================================
Produces three output files:
  data/prompts/en/drawbench_prompts_en.csv    — 50 DrawBench prompts
  data/prompts/en/genaibench_prompts_en.csv   — 50 GenAI-Bench prompts
  data/prompts/en/phase1b_prompts_en.csv      — 100 combined master file

DrawBench category → DAG Node mapping:
  Colors     (13 from 25) → Node 3: Attribute Compliance
  Counting   (13 from 19) → Node 1: Object Presence
  Positional (12 from 20) → Node 2: Spatial Layout
  Text       (12 from 21) → Node 4: Typography

GenAI-Bench tag → DAG Node mapping:
  Spatial Relation (25 from available) → Node 2: Spatial Layout
  Attribute        (25 from remaining) → Node 3: Attribute Compliance

GenAI-Bench is used because it has published human alignment ratings,
enabling Spearman correlation without a separate human study (RQ1).
Human score per prompt = mean of all rater scores across all 6 models.

Run from MulBrandEval/ root:
  python phase1b_alignment_baseline/sample_prompts.py
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────────

DRAWBENCH_URL = "https://docs.google.com/spreadsheets/d/1y7nAbmR4FREi6npB1u-Bo3GFdwdOPYJc617rBOxIRHY/gviz/tq?tqx=out:csv"

OUT_DB      = Path("data/prompts/en/drawbench_prompts_en.csv")
OUT_GB      = Path("data/prompts/en/genaibench_prompts_en.csv")
OUT_MASTER  = Path("data/prompts/en/phase1b_prompts_en.csv")

for p in [OUT_DB, OUT_GB, OUT_MASTER]:
    p.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

DRAWBENCH_CONFIG = {
    "Colors":     {"n": 13, "dag_node": "node3_attribute"},
    "Counting":   {"n": 13, "dag_node": "node1_object_presence"},
    "Positional": {"n": 12, "dag_node": "node2_spatial"},
    "Text":       {"n": 12, "dag_node": "node4_typography"},
}

# ── PART 1: DrawBench ───────────────────────────────────────────────────────────

print("=" * 60)
print("PART 1 — DrawBench Prompt Sampling")
print("=" * 60)

print("Loading DrawBench...")
db_raw = pd.read_csv(DRAWBENCH_URL)
db_raw.columns = db_raw.columns.str.lower()
print(f"Loaded {len(db_raw)} total prompts")

db_parts = []
for category, config in DRAWBENCH_CONFIG.items():
    subset = db_raw[db_raw["category"] == category].copy()
    sampled = subset.sample(n=config["n"], random_state=RANDOM_SEED)
    sampled = sampled.copy()
    sampled["dag_node"] = config["dag_node"]
    db_parts.append(sampled)
    print(f"  {category}: selected {config['n']} from {len(subset)} available")

db_result = pd.concat(db_parts, ignore_index=True)
db_result = db_result.rename(columns={"prompts": "caption"})
db_result["prompt_id"] = [f"DB{str(i+1).zfill(3)}" for i in range(len(db_result))]
db_result["source"] = "DrawBench"
db_result["human_score"] = np.nan  # DrawBench has no human ratings

db_final = db_result[["prompt_id", "caption", "category", "dag_node",
                        "source", "human_score"]].copy()

assert len(db_final) == 50
db_final.to_csv(OUT_DB, index=False)
print(f"\nDrawBench CSV saved: {OUT_DB}  ({len(db_final)} prompts)")

# ── PART 2: GenAI-Bench ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PART 2 — GenAI-Bench Prompt Sampling")
print("=" * 60)

print("Loading GenAI-Bench (from local cache)...")
ds = load_dataset("BaiqiL/GenAI-Bench", split="train")
gb_raw = ds.to_pandas()
print(f"Loaded {len(gb_raw)} total prompts")

# ── Helper: check if a tag exists in the basic array ───────────────────────────

def has_tag(tags_dict, tag_name):
    """Return True if tag_name is in the basic tags array."""
    try:
        return tag_name in tags_dict["basic"]
    except Exception:
        return False

# ── Helper: compute mean human score across all models and all raters ───────────

def mean_human_score(ratings_dict):
    """
    ratings_dict = {'DALLE_3': [5,5,5], 'Midjourney_6': [3,3,3], ...}
    Returns the mean of all individual ratings across all models.
    This gives a single prompt-level human alignment score for Spearman ρ.
    """
    try:
        all_scores = []
        for scores in ratings_dict.values():
            all_scores.extend(list(scores))
        return round(float(np.mean(all_scores)), 4)
    except Exception:
        return np.nan

# ── Apply tag filters ───────────────────────────────────────────────────────────

gb_raw["has_spatial"]   = gb_raw["Tags"].apply(lambda t: has_tag(t, "Spatial Relation"))
gb_raw["has_attribute"] = gb_raw["Tags"].apply(lambda t: has_tag(t, "Attribute"))
gb_raw["human_score"]   = gb_raw["HumanRatings"].apply(mean_human_score)

spatial_pool    = gb_raw[gb_raw["has_spatial"]].copy()
print(f"\n  Prompts with 'Spatial Relation' tag: {len(spatial_pool)}")

# Sample 25 Spatial prompts
spatial_sampled = spatial_pool.sample(n=25, random_state=RANDOM_SEED)
spatial_sampled = spatial_sampled.copy()
spatial_sampled["dag_node"] = "node2_spatial"
spatial_sampled["category"] = "Spatial Relation"

# For Attribute: exclude already-selected prompts to avoid overlap
already_selected = set(spatial_sampled["Index"].tolist())
attribute_pool   = gb_raw[
    gb_raw["has_attribute"] & ~gb_raw["Index"].isin(already_selected)
].copy()
print(f"  Prompts with 'Attribute' tag (excluding Spatial selection): {len(attribute_pool)}")

attribute_sampled = attribute_pool.sample(n=25, random_state=RANDOM_SEED)
attribute_sampled = attribute_sampled.copy()
attribute_sampled["dag_node"] = "node3_attribute"
attribute_sampled["category"] = "Attribute"

gb_combined = pd.concat([spatial_sampled, attribute_sampled], ignore_index=True)

# ── Standardise columns ─────────────────────────────────────────────────────────

gb_combined = gb_combined.rename(columns={"Prompt": "caption"})
gb_combined["prompt_id"] = [f"GB{str(i+1).zfill(3)}" for i in range(len(gb_combined))]
gb_combined["source"] = "GenAI-Bench"

gb_final = gb_combined[["prompt_id", "caption", "category", "dag_node",
                          "source", "human_score"]].copy()

assert len(gb_final) == 50
assert gb_final["prompt_id"].nunique() == 50
assert gb_final["caption"].isnull().sum() == 0

gb_final.to_csv(OUT_GB, index=False)
print(f"\nGenAI-Bench CSV saved: {OUT_GB}  ({len(gb_final)} prompts)")
print(f"  Human score range: {gb_final['human_score'].min():.3f} — "
      f"{gb_final['human_score'].max():.3f}")
print(f"  Human score mean:  {gb_final['human_score'].mean():.3f}")

# ── PART 3: Combine into master file ───────────────────────────────────────────

print("\n" + "=" * 60)
print("PART 3 — Combined Master File")
print("=" * 60)

master = pd.concat([db_final, gb_final], ignore_index=True)

assert len(master) == 100
assert master["prompt_id"].nunique() == 100

master.to_csv(OUT_MASTER, index=False)
print(f"Master CSV saved: {OUT_MASTER}  ({len(master)} prompts)")
print(f"\nBreakdown by source:")
print(master.groupby(["source", "category", "dag_node"]).size().to_string())
print(f"\nHuman scores available: {master['human_score'].notna().sum()} / 100")
print("(DrawBench has no human ratings — NaN is expected and correct)")
print("\nSample rows from master:")
print(master[["prompt_id", "caption", "category", "dag_node",
              "source", "human_score"]].head(6).to_string(index=False))
print("\n" + "=" * 60)
print("Phase 1B prompt sampling complete.")
print("=" * 60)