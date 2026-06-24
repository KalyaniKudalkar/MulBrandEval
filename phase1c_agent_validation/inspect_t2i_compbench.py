# ══════════════════════════════════════════════════════════════════════════════
# inspect_t2i_compbench.py
# Phase 1C — Node 3 (Attribute/Colour): T2I-CompBench Dataset Inspection
# ══════════════════════════════════════════════════════════════════════════════
#
# Purpose:
#   Documents the dataset discovery process and confirms the structure of the
#   T2I-CompBench color_val split used as the Node 3 ground truth source.
#
# Dataset decision history (Phase 1C Node 3):
#   Originally planned: DALL-EVAL PaintSkills colour split
#   → Does not exist: all PaintSkills objects have color='plain' (no colour GT)
#
#   Alternatives investigated and rejected:
#   → TIFA:                     no colour-specific labels, no downloadable images
#   → Karine-Huang/T2I-CompBench (HuggingFace): gated, 401 authentication errors
#   → NinaKarine/t2i-compbench: text prompts only, no images or human scores
#   → GenAI-Bench Attribute:    reuses Phase 1B data; tests general alignment,
#                               not colour accuracy specifically
#
#   Final decision: sayakpaul/t2i-compbench, color_val split
#   → 300 two-object colour-binding sentences (e.g. "a red car and a blue bus")
#   → No authentication required
#   → Single 'text' column: raw T2I-CompBench prompt .txt files in HF format
#   → No images or human scores included; BLIP-VQA used as reference metric
#     (following T2I-CompBench's own colour evaluation protocol)
#   → 30 prompts sampled (seed 42) for Node 3 validation
#
# Usage:
#   python phase1c_agent_validation/inspect_t2i_compbench.py
#
# ══════════════════════════════════════════════════════════════════════════════

from datasets import load_dataset
import re

# ── Load the dataset ──────────────────────────────────────────────────────────
print("Loading sayakpaul/t2i-compbench (color_val split)...")
ds = load_dataset("sayakpaul/t2i-compbench", "color_val", split="val")
print(f"  Total prompts: {len(ds)}")
print(f"  Columns:       {ds.column_names}")
print(f"  Features:      {ds.features}")

# ── Show first 15 prompts ─────────────────────────────────────────────────────
# Confirmed during Phase 1C investigation: all prompts are two-object
# colour-binding sentences of the form '[colour] [object] ... [colour] [object]'
print("\nFirst 15 color_val prompts:")
for i in range(15):
    print(f"  {i + 1:2d}. {ds[i]['text']}")

# ── Verify colour-binding structure ──────────────────────────────────────────
# Each prompt should contain at least one colour word paired with an object
COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink",
          "brown", "black", "white", "gray", "grey", "cyan", "magenta", "gold"]

prompts_with_two_colours = 0
prompts_with_one_colour  = 0
prompts_with_no_colour   = 0

for row in ds:
    text       = row["text"].lower()
    color_hits = sum(1 for c in COLORS if re.search(rf"\b{c}\b", text))
    if color_hits >= 2:
        prompts_with_two_colours += 1
    elif color_hits == 1:
        prompts_with_one_colour  += 1
    else:
        prompts_with_no_colour   += 1

total = len(ds)
print(f"\nColour-binding analysis across all {total} prompts:")
print(f"  Two or more colour words : {prompts_with_two_colours} "
      f"({prompts_with_two_colours / total:.1%})")
print(f"  Exactly one colour word  : {prompts_with_one_colour}  "
      f"({prompts_with_one_colour / total:.1%})")
print(f"  No colour words detected : {prompts_with_no_colour}  "
      f"({prompts_with_no_colour / total:.1%})")

# ── Show the 30 prompts sampled for Node 3 (seed 42) ─────────────────────────
import random
random.seed(42)
selected_indices = sorted(random.sample(range(len(ds)), 30))

print(f"\n30 prompts sampled for Node 3 validation (seed=42):")
for rank, idx in enumerate(selected_indices, 1):
    print(f"  CV{rank:03d} [idx={idx:3d}] {ds[idx]['text']}")