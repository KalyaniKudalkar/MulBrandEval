import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
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
PHASE1B_IMAGES_DIR = Path("data/generated_images/phase1b")
PHASE1B_PROMPTS_EN = Path("data/prompts/en/phase1b_prompts_en.csv")
RESULTS_DIR       = Path("results/phase1c")
NODE1_CSV         = RESULTS_DIR / "node1_object_presence.csv"
NODE4_CSV         = RESULTS_DIR / "node4_typography.csv"
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
# NODE 4 — Typography Validation
# Validates: EasyOCR correctly extracts text from T2I-generated images
# Dataset:   12 DrawBench Text prompts, English, seed 42, both models (24 images)
# Metric:    Levenshtein accuracy   |   Target: accuracy >= 0.75
# ══════════════════════════════════════════════════════════════════════════════

def run_node4():
    print("=" * 65)
    print("NODE 4 — Typography Validation")
    print("=" * 65)

    # ── Load DrawBench Text prompts ───────────────────────────────────────────
    df = pd.read_csv(PHASE1B_PROMPTS_EN)
    text_df = df[(df["category"] == "Text") & (df["source"] == "DrawBench")].copy()
    print(f"\n  DrawBench Text prompts loaded: {len(text_df)}")

    # ── Extract required text string from each caption ────────────────────────
    def extract_required_text(caption: str) -> str:
        match = re.search(r"'([^']+)'", caption)
        return match.group(1) if match else ""

    text_df["required_text"] = text_df["caption"].apply(extract_required_text)

    print("  Required text strings:")
    for _, row in text_df.iterrows():
        print(f"    {row['prompt_id']} → '{row['required_text']}'")

    # ── Levenshtein similarity helpers ────────────────────────────────────────
    def levenshtein_distance(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    def norm_lev_sim(s1: str, s2: str) -> float:
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        dist = levenshtein_distance(s1, s2)
        return 1.0 - dist / max(len(s1), len(s2))

    def best_match_score(required: str, detected_texts: list) -> float:
        """
        Compute best match score between required text and EasyOCR detections.
        Strategy 1: substring match in joined detections → score = 1.0
        Strategy 2: Levenshtein similarity vs each fragment and joined text → take max
        """
        if not detected_texts:
            return 0.0
        req_lower = required.lower().strip()
        joined    = " ".join(detected_texts).lower()

        # Exact substring match is the strongest signal
        if req_lower in joined:
            return 1.0

        # Levenshtein fallback: try each fragment and the full join
        candidates = [t.lower() for t in detected_texts] + [joined]
        return max(norm_lev_sim(req_lower, c) for c in candidates)

    # ── Initialise EasyOCR ────────────────────────────────────────────────────
    import easyocr
    print("\n  Initialising EasyOCR (English, CPU)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print("  EasyOCR ready.\n")

    MODELS        = ["sd15", "flux"]
    CONF_THRESHOLD = 0.3   # EasyOCR confidence filter
    HIT_THRESHOLD  = 0.5   # Levenshtein similarity to count as a hit

    results = []

    print(f"  Processing {len(text_df)} prompts × {len(MODELS)} models "
          f"= {len(text_df) * len(MODELS)} images...\n")

    for _, row in text_df.iterrows():
        prompt_id     = row["prompt_id"]
        required_text = row["required_text"]

        for model in MODELS:
            image_path = PHASE1B_IMAGES_DIR / model / "en" / f"{prompt_id}_seed42.png"

            if not image_path.exists():
                print(f"  [WARNING] Not found: {image_path} — skipping")
                results.append({
                    "prompt_id":       prompt_id,
                    "model":           model,
                    "required_text":   required_text,
                    "extracted_text":  "",
                    "best_similarity": 0.0,
                    "hit":             0,
                    "image_found":     0,
                })
                continue

            # Run EasyOCR
            ocr_out = reader.readtext(str(image_path), detail=1)
            detected = [text for _, text, conf in ocr_out if conf >= CONF_THRESHOLD]
            joined   = " ".join(detected)

            score = best_match_score(required_text, detected)
            hit   = 1 if score >= HIT_THRESHOLD else 0

            status_str = "HIT " if hit else "MISS"
            print(f"  [{status_str}] {prompt_id} | {model:4s} | "
                  f"required: '{required_text}' | "
                  f"extracted: '{joined}' | sim={score:.3f}")

            results.append({
                "prompt_id":       prompt_id,
                "model":           model,
                "required_text":   required_text,
                "extracted_text":  joined,
                "best_similarity": round(score, 4),
                "hit":             hit,
                "image_found":     1,
            })

    # ── Save detailed results ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(NODE4_CSV, index=False, encoding="utf-8")
    print(f"\n  Detailed results saved: {NODE4_CSV}  ({len(results_df)} rows)")

    # ── Compute accuracy ──────────────────────────────────────────────────────
    valid = results_df[results_df["image_found"] == 1]
    total = len(valid)
    hits  = int(valid["hit"].sum())
    accuracy = hits / total if total > 0 else 0.0

    print("\n  Per-model breakdown:")
    for model in MODELS:
        m = valid[valid["model"] == model]
        m_hits = int(m["hit"].sum())
        m_acc  = m_hits / len(m) if len(m) > 0 else 0.0
        print(f"    {model:4s}: {m_hits}/{len(m)} hits  accuracy={m_acc:.4f}")

    TARGET    = 0.75
    pass_fail = "PASS" if accuracy >= TARGET else "FAIL"
    threshold = (f"EasyOCR confidence >= {CONF_THRESHOLD}; "
                 f"Levenshtein similarity >= {HIT_THRESHOLD} or substring match")

    print("\n" + "-" * 45)
    print(f"  Total images : {total}")
    print(f"  Hits         : {hits}")
    print(f"  Accuracy     : {accuracy:.4f}  (target >= {TARGET})")
    print(f"  Status       : {pass_fail}")
    print("-" * 45)

    if pass_fail == "FAIL":
        print("\n  NOTE: Accuracy below target.")
        print("  This likely reflects T2I model text-rendering quality")
        print("  (especially SD v1.5), not EasyOCR capability.")
        print("  Review the per-model breakdown above for details.")

    # ── Append to component validation table ─────────────────────────────────
    table_row = {
        "node":                 "Node 4",
        "capability":           "Typography",
        "tool":                 "EasyOCR 1.7.1",
        "dataset":              "DrawBench Text (n=12 prompts, 24 images)",
        "metric":               "Levenshtein accuracy",
        "score":                round(accuracy, 4),
        "target":               f">= {TARGET}",
        "pass_fail":            pass_fail,
        "calibrated_threshold": threshold,
    }

    write_header = not VALIDATION_TABLE.exists()
    with open(VALIDATION_TABLE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(table_row)

    print(f"\n  Component validation table updated: {VALIDATION_TABLE}")

    # ── Append Node 4 row to summary text ────────────────────────────────────
    node4_summary = (
        f"Node 4 | Typography | EasyOCR 1.7.1 | "
        f"Accuracy={accuracy:.4f} | target>=0.75 | {pass_fail} | "
        f"hits={hits}/{total} | "
        f"conf_threshold={CONF_THRESHOLD} | lev_threshold={HIT_THRESHOLD}\n"
    )
    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write(node4_summary)

    print(f"  Summary written: {SUMMARY_TXT}")
    print("\nNode 4 validation complete.")
    return accuracy, pass_fail


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# Usage:
#   python phase1c_agent_validation/phase1c_validate.py 1   → run Node 1 only
#   python phase1c_agent_validation/phase1c_validate.py 4   → run Node 4 only
#   python phase1c_agent_validation/phase1c_validate.py     → run all nodes
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    node = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}

    if node in ("1", "all"):
        f1, status = run_node1()
        results["node1"] = status

    if node in ("4", "all"):
        accuracy, status = run_node4()
        results["node4"] = status

    all_pass = all(v == "PASS" for v in results.values())
    sys.exit(0 if all_pass else 1)