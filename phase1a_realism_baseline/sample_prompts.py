# phase1a_realism_baseline/sample_prompts.py
# Task 1: Sample 50 prompts from MS-COCO 2017 validation set
# Categorisation uses OFFICIAL COCO instance annotations (instances_val2017.json)
# NOT keyword matching — fully reproducible and methodologically defensible

import json
import random
import pandas as pd
from pathlib import Path

random.seed(42)  # Fixed seed for full reproducibility

# ── Paths ─────────────────────────────────────────────────────
CAPTIONS_FILE  = Path("annotations/captions_val2017.json")
INSTANCES_FILE = Path("annotations/instances_val2017.json")
OUTPUT_DIR     = Path("data/prompts/en")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── COCO supercategory → thesis category mapping ───────────────
# Source: instances_val2017.json "categories" field
SUPERCATEGORY_MAP = {
    "person":  "people",
    "vehicle": "vehicles",
    "food":    "food",
    "animal":  "animals"
}

# Caption relevance keywords — ensures caption describes the correct subject
# Applied AFTER COCO annotation filtering for double verification
CAPTION_KEYWORDS = {
    "people":   ["person", "man", "woman", "boy", "girl", "child",
                 "people", "player", "skater", "rider", "crowd"],
    "vehicles": ["car", "truck", "bus", "train", "motorcycle", "bike",
                 "bicycle", "boat", "vehicle", "monorail", "van", "scooter",
                 "traffic", "highway", "road", "driving", "parked"],
    "food":     ["food", "pizza", "sandwich", "cake", "donut", "fruit",
                 "vegetable", "banana", "apple", "hot dog", "burger",
                 "meal", "eating", "plate", "fork", "dish", "lunch",
                 "dinner", "cheese", "bread"],
    "animals":  ["dog", "cat", "bird", "horse", "cow", "elephant",
                 "zebra", "giraffe", "bear", "sheep", "animal",
                 "cattle", "herd", "wildlife"]
}

def caption_matches_category(caption: str, category: str) -> bool:
    """Check that caption actually describes the assigned category."""
    caption_lower = caption.lower()
    return any(kw in caption_lower for kw in CAPTION_KEYWORDS[category])

PROMPTS_PER_CATEGORY = {
    "people":   13,
    "vehicles": 13,
    "food":     12,
    "animals":  12
}

def load_image_categories(instances_file: Path) -> dict:
    """
    Build a mapping of image_id → thesis category
    using official COCO supercategory annotations.
    Each image is assigned the thesis category of its
    dominant (most frequent) annotated object supercategory.
    """
    with open(instances_file, "r") as f:
        data = json.load(f)

    # Build category_id → supercategory lookup
    cat_id_to_supercategory = {
        cat["id"]: cat["supercategory"]
        for cat in data["categories"]
    }

    # Count supercategory occurrences per image
    from collections import defaultdict, Counter
    image_supercategory_counts = defaultdict(Counter)

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        supercategory = cat_id_to_supercategory.get(cat_id, "other")
        image_supercategory_counts[img_id][supercategory] += 1

    # Assign each image its dominant thesis category
    image_to_thesis_category = {}
    for img_id, counts in image_supercategory_counts.items():
        # Find the dominant supercategory that maps to a thesis category
        for supercategory, _ in counts.most_common():
            if supercategory in SUPERCATEGORY_MAP:
                image_to_thesis_category[img_id] = SUPERCATEGORY_MAP[supercategory]
                break

    return image_to_thesis_category


def main():
    print("Loading COCO instance annotations...")
    image_to_category = load_image_categories(INSTANCES_FILE)
    print(f"Images with valid thesis category: {len(image_to_category)}")

    print("Loading COCO captions...")
    with open(CAPTIONS_FILE, "r") as f:
        captions_data = json.load(f)

    # One caption per image (first occurrence)
    captions_by_id = {}
    for ann in captions_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in captions_by_id:
            captions_by_id[img_id] = ann["caption"].strip()

    # Combine captions with verified COCO categories + caption relevance check
    categorised = {cat: [] for cat in PROMPTS_PER_CATEGORY}
    skipped = 0
    for img_id, caption in captions_by_id.items():
        category = image_to_category.get(img_id)
        if category in categorised:
            if caption_matches_category(caption, category):
                categorised[category].append({
                    "image_id": img_id,
                    "caption":  caption,
                    "category": category
                })
            else:
                skipped += 1

    print(f"Captions filtered (annotation ✓ but caption mismatch): {skipped}")

    print("\nAvailable images per category (COCO-verified):")
    for cat, items in categorised.items():
        print(f"  {cat:<12}: {len(items)} images")

    # Stratified random sampling
    sampled = []
    for cat, n in PROMPTS_PER_CATEGORY.items():
        pool = categorised[cat]
        if len(pool) < n:
            raise ValueError(f"Not enough images for {cat}: {len(pool)} < {n}")
        selected = random.sample(pool, n)
        sampled.extend(selected)

    # Build output dataframe
    df = pd.DataFrame(sampled).reset_index(drop=True)
    df.insert(0, "prompt_id", [f"P{i+1:03d}" for i in range(len(df))])

    print(f"\nTotal sampled: {len(df)} prompts")
    print(df["category"].value_counts().to_string())

    # Save to CSV
    csv_path = OUTPUT_DIR / "coco_prompts_en.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Preview one prompt per category
    print("\nSample prompts (COCO-verified categories):")
    for cat in PROMPTS_PER_CATEGORY:
        sample = df[df["category"] == cat].iloc[0]
        print(f"  [{cat}] {sample['caption'][:80]}")

if __name__ == "__main__":
    main()