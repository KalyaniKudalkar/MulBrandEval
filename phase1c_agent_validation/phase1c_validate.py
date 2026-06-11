import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
import base64
import time
import csv
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SEED = 42
random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
COCO_IMAGES_DIR   = Path("data/reference_images/coco_phase1a")
COCO_ANNOTATIONS  = Path("annotations/instances_val2017.json")
PROMPTS_CSV       = Path("data/prompts/en/coco_prompts_en.csv")
RESULTS_DIR       = Path("results/phase1c")
NODE1_CSV         = RESULTS_DIR / "node1_object_presence.csv"
VALIDATION_TABLE  = RESULTS_DIR / "component_validation_table.csv"
SUMMARY_TXT       = RESULTS_DIR / "phase1c_summary.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Object Presence Validation
# Validates: GPT-4o mini VQA correctly detects object presence/absence
# Dataset:   50 MS-COCO val2017 images + instances_val2017.json ground truth
# Metric:    Precision, Recall, F1   |   Target: F1 >= 0.80
# ══════════════════════════════════════════════════════════════════════════════

def run_node1():
    print("=" * 65)
    print("NODE 1 — Object Presence Validation")
    print("=" * 65)

    # ── Load COCO annotations ─────────────────────────────────────────────────
    print("\nLoading COCO annotations...")
    with open(COCO_ANNOTATIONS, "r", encoding="utf-8") as f:
        coco = json.load(f)

    category_map      = {cat["id"]: cat["name"] for cat in coco["categories"]}
    all_category_names = set(category_map.values())
    print(f"  COCO categories loaded: {len(all_category_names)}")

    # ── Load our 50 prompt image IDs ──────────────────────────────────────────
    prompts_df    = pd.read_csv(PROMPTS_CSV)
    our_image_ids = set(prompts_df["image_id"].tolist())
    print(f"  Prompt image IDs loaded: {len(our_image_ids)}")

    # ── Build ground truth: image_id -> set of present category names ─────────
    print("  Building per-image ground truth...")
    image_categories = {iid: set() for iid in our_image_ids}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid in our_image_ids:
            image_categories[iid].add(category_map[ann["category_id"]])

    present_counts = [len(v) for v in image_categories.values()]
    print(f"  Avg objects per image: {sum(present_counts)/len(present_counts):.1f}  "
          f"(min {min(present_counts)}, max {max(present_counts)})")

    # ── GPT-4o mini VQA helpers ───────────────────────────────────────────────
    def encode_image(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def query_object_present(image_path: Path, object_name: str, retries: int = 3) -> int:
        b64 = encode_image(image_path)
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "low"
                                }
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Is a {object_name} clearly visible in this image? "
                                    "Answer with only 'yes' or 'no'."
                                )
                            }
                        ]
                    }],
                    max_tokens=5,
                    temperature=0
                )
                answer = response.choices[0].message.content.strip().lower()
                return 1 if answer.startswith("yes") else 0
            except Exception as e:
                print(f"    Retry {attempt + 1}/3 ({object_name}): {e}")
                time.sleep(3)
        return 0  # Conservative default on failure

    # ── Run validation ────────────────────────────────────────────────────────
    results = []
    total_queries = 0

    print(f"\nRunning VQA queries across {len(prompts_df)} images...")
    print("  (This will take several minutes due to API rate limiting)\n")

    for idx, row in prompts_df.iterrows():
        image_id       = row["image_id"]
        image_filename = f"{image_id:012d}.jpg"
        image_path     = COCO_IMAGES_DIR / image_filename

        if not image_path.exists():
            print(f"  [WARNING] Image not found: {image_filename} — skipping")
            continue

        present  = list(image_categories[image_id])
        absent   = list(all_category_names - image_categories[image_id])
        random.shuffle(absent)
        negatives = absent[:len(present)]   # equal count to positives

        test_cases = [(obj, 1) for obj in present] + [(obj, 0) for obj in negatives]
        total_queries += len(test_cases)

        print(f"  [{idx + 1:2d}/50] image_id={image_id} | "
              f"+{len(present)} pos  -{len(negatives)} neg  "
              f"({len(test_cases)} queries)")

        for object_name, ground_truth in test_cases:
            prediction = query_object_present(image_path, object_name)
            results.append({
                "image_id":     image_id,
                "object_name":  object_name,
                "ground_truth": ground_truth,
                "prediction":   prediction,
                "correct":      int(prediction == ground_truth)
            })
            time.sleep(0.4)   # ~2.5 req/s — stays well within GPT-4o mini limits

    # ── Save detailed results ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(NODE1_CSV, index=False, encoding="utf-8")
    print(f"\nDetailed results saved: {NODE1_CSV}  ({len(results_df)} rows)")

    # ── Compute Precision / Recall / F1 ──────────────────────────────────────
    tp = len(results_df[(results_df["ground_truth"] == 1) & (results_df["prediction"] == 1)])
    fp = len(results_df[(results_df["ground_truth"] == 0) & (results_df["prediction"] == 1)])
    fn = len(results_df[(results_df["ground_truth"] == 1) & (results_df["prediction"] == 0)])
    tn = len(results_df[(results_df["ground_truth"] == 0) & (results_df["prediction"] == 0)])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    TARGET     = 0.80
    pass_fail  = "PASS" if f1 >= TARGET else "FAIL"
    threshold  = "binary yes/no VQA boundary"

    print("\n" + "-" * 45)
    print(f"  Total queries : {total_queries}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1            : {f1:.4f}  (target >= {TARGET})")
    print(f"  Status        : {pass_fail}")
    print("-" * 45)

    if pass_fail == "FAIL":
        print("\n  NOTE: F1 below target. Consider adjusting the VQA prompt")
        print("  or lowering the detection threshold before Phase 2A.")

    # ── Append to component validation table ─────────────────────────────────
    table_row = {
        "node":                 "Node 1",
        "capability":           "Object Presence",
        "tool":                 "GPT-4o mini VQA",
        "dataset":              "MS-COCO val2017 (n=50 images)",
        "metric":               "F1",
        "score":                round(f1, 4),
        "target":               f">= {TARGET}",
        "pass_fail":            pass_fail,
        "calibrated_threshold": threshold
    }

    write_header = not VALIDATION_TABLE.exists()
    with open(VALIDATION_TABLE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(table_row)

    print(f"\nComponent validation table updated: {VALIDATION_TABLE}")

    # ── Append Node 1 row to summary text ────────────────────────────────────
    node1_summary = (
        f"Node 1 | Object Presence | GPT-4o mini VQA | "
        f"F1={f1:.4f} | target>=0.80 | {pass_fail} | "
        f"P={precision:.4f} R={recall:.4f} | "
        f"TP={tp} FP={fp} FN={fn} TN={tn}\n"
    )

    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write("MulBrandEval Phase 1C - Component Validation Summary\n")
        f.write("=" * 60 + "\n")
        f.write(node1_summary)

    print(f"Summary written: {SUMMARY_TXT}")
    print("\nNode 1 validation complete.")
    return f1, pass_fail


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    f1, status = run_node1()
    sys.exit(0 if status == "PASS" else 1)