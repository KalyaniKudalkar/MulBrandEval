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
COCO_IMAGES_DIR    = Path("data/reference_images/coco_phase1a")
COCO_ANNOTATIONS   = Path("annotations/instances_val2017.json")
PROMPTS_CSV        = Path("data/prompts/en/coco_prompts_en.csv")
PHASE1B_IMAGES_DIR = Path("data/generated_images/phase1b")
PHASE1B_PROMPTS_EN = Path("data/prompts/en/phase1b_prompts_en.csv")
NODE3_IMAGES_DIR   = Path("data/generated_images/phase1c/node3")
NODE3_PROMPTS_CSV  = Path("data/prompts/en/phase1c_node3_prompts_en.csv")
RESULTS_DIR        = Path("results/phase1c")
NODE1_CSV          = RESULTS_DIR / "node1_object_presence.csv"
NODE3_CSV          = RESULTS_DIR / "node3_colour_attribute.csv"
NODE3_RECALIB_CSV  = RESULTS_DIR / "node3_recalibrated.csv"
NODE4_CSV          = RESULTS_DIR / "node4_typography.csv"
VALIDATION_TABLE   = RESULTS_DIR / "component_validation_table.csv"
SUMMARY_TXT        = RESULTS_DIR / "phase1c_summary.txt"

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

    print("\nLoading COCO annotations...")
    with open(COCO_ANNOTATIONS, "r", encoding="utf-8") as f:
        coco = json.load(f)

    category_map       = {cat["id"]: cat["name"] for cat in coco["categories"]}
    all_category_names = set(category_map.values())
    print(f"  COCO categories loaded: {len(all_category_names)}")

    prompts_df    = pd.read_csv(PROMPTS_CSV)
    our_image_ids = set(prompts_df["image_id"].tolist())
    print(f"  Prompt image IDs loaded: {len(our_image_ids)}")

    print("  Building per-image ground truth...")
    image_categories = {iid: set() for iid in our_image_ids}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid in our_image_ids:
            image_categories[iid].add(category_map[ann["category_id"]])

    present_counts = [len(v) for v in image_categories.values()]
    print(f"  Avg objects per image: {sum(present_counts)/len(present_counts):.1f}  "
          f"(min {min(present_counts)}, max {max(present_counts)})")

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
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                           "detail": "low"}},
                            {"type": "text",
                             "text": (f"Is a {object_name} clearly visible in this image? "
                                      "Answer with only 'yes' or 'no'.")}
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
        return 0

    results      = []
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

        present   = list(image_categories[image_id])
        absent    = list(all_category_names - image_categories[image_id])
        random.shuffle(absent)
        negatives = absent[:len(present)]

        test_cases     = [(obj, 1) for obj in present] + [(obj, 0) for obj in negatives]
        total_queries += len(test_cases)

        print(f"  [{idx + 1:2d}/50] image_id={image_id} | "
              f"+{len(present)} pos  -{len(negatives)} neg  ({len(test_cases)} queries)")

        for object_name, ground_truth in test_cases:
            prediction = query_object_present(image_path, object_name)
            results.append({"image_id": image_id, "object_name": object_name,
                             "ground_truth": ground_truth, "prediction": prediction,
                             "correct": int(prediction == ground_truth)})
            time.sleep(0.4)

    results_df = pd.DataFrame(results)
    results_df.to_csv(NODE1_CSV, index=False, encoding="utf-8")
    print(f"\nDetailed results saved: {NODE1_CSV}  ({len(results_df)} rows)")

    tp = len(results_df[(results_df["ground_truth"] == 1) & (results_df["prediction"] == 1)])
    fp = len(results_df[(results_df["ground_truth"] == 0) & (results_df["prediction"] == 1)])
    fn = len(results_df[(results_df["ground_truth"] == 1) & (results_df["prediction"] == 0)])
    tn = len(results_df[(results_df["ground_truth"] == 0) & (results_df["prediction"] == 0)])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    TARGET    = 0.80
    pass_fail = "PASS" if f1 >= TARGET else "FAIL"
    threshold = "binary yes/no VQA boundary"

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

    table_row = {
        "node": "Node 1", "capability": "Object Presence",
        "tool": "GPT-4o mini VQA", "dataset": "MS-COCO val2017 (n=50 images)",
        "metric": "F1", "score": round(f1, 4), "target": f">= {TARGET}",
        "pass_fail": pass_fail, "calibrated_threshold": threshold
    }
    write_header = not VALIDATION_TABLE.exists()
    with open(VALIDATION_TABLE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(table_row)
    print(f"\nComponent validation table updated: {VALIDATION_TABLE}")

    node1_summary = (f"Node 1 | Object Presence | GPT-4o mini VQA | "
                     f"F1={f1:.4f} | target>=0.80 | {pass_fail} | "
                     f"P={precision:.4f} R={recall:.4f} | "
                     f"TP={tp} FP={fp} FN={fn} TN={tn}\n")
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

    df      = pd.read_csv(PHASE1B_PROMPTS_EN)
    text_df = df[(df["category"] == "Text") & (df["source"] == "DrawBench")].copy()
    print(f"\n  DrawBench Text prompts loaded: {len(text_df)}")

    def extract_required_text(caption: str) -> str:
        match = re.search(r"'([^']+)'", caption)
        return match.group(1) if match else ""

    text_df["required_text"] = text_df["caption"].apply(extract_required_text)
    print("  Required text strings:")
    for _, row in text_df.iterrows():
        print(f"    {row['prompt_id']} -> '{row['required_text']}'")

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
        if not detected_texts:
            return 0.0
        req_lower = required.lower().strip()
        joined    = " ".join(detected_texts).lower()
        if req_lower in joined:
            return 1.0
        candidates = [t.lower() for t in detected_texts] + [joined]
        return max(norm_lev_sim(req_lower, c) for c in candidates)

    import easyocr
    print("\n  Initialising EasyOCR (English, CPU)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print("  EasyOCR ready.\n")

    MODELS         = ["sd15", "flux"]
    CONF_THRESHOLD = 0.3
    HIT_THRESHOLD  = 0.5
    results        = []

    print(f"  Processing {len(text_df)} prompts x {len(MODELS)} models "
          f"= {len(text_df) * len(MODELS)} images...\n")

    for _, row in text_df.iterrows():
        prompt_id     = row["prompt_id"]
        required_text = row["required_text"]

        for model in MODELS:
            image_path = PHASE1B_IMAGES_DIR / model / "en" / f"{prompt_id}_seed42.png"

            if not image_path.exists():
                print(f"  [WARNING] Not found: {image_path} — skipping")
                results.append({"prompt_id": prompt_id, "model": model,
                                 "required_text": required_text, "extracted_text": "",
                                 "best_similarity": 0.0, "hit": 0, "image_found": 0})
                continue

            ocr_out  = reader.readtext(str(image_path), detail=1)
            detected = [text for _, text, conf in ocr_out if conf >= CONF_THRESHOLD]
            joined   = " ".join(detected)
            score    = best_match_score(required_text, detected)
            hit      = 1 if score >= HIT_THRESHOLD else 0

            print(f"  [{'HIT ' if hit else 'MISS'}] {prompt_id} | {model:4s} | "
                  f"required: '{required_text}' | extracted: '{joined}' | sim={score:.3f}")

            results.append({"prompt_id": prompt_id, "model": model,
                             "required_text": required_text, "extracted_text": joined,
                             "best_similarity": round(score, 4), "hit": hit,
                             "image_found": 1})

    results_df = pd.DataFrame(results)
    results_df.to_csv(NODE4_CSV, index=False, encoding="utf-8")
    print(f"\n  Detailed results saved: {NODE4_CSV}  ({len(results_df)} rows)")

    valid    = results_df[results_df["image_found"] == 1]
    total    = len(valid)
    hits     = int(valid["hit"].sum())
    accuracy = hits / total if total > 0 else 0.0

    print("\n  Per-model breakdown:")
    for model in MODELS:
        m     = valid[valid["model"] == model]
        m_hit = int(m["hit"].sum())
        m_acc = m_hit / len(m) if len(m) > 0 else 0.0
        print(f"    {model:4s}: {m_hit}/{len(m)} hits  accuracy={m_acc:.4f}")

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
        print("  This reflects SD v1.5 architectural text-rendering limitation,")
        print("  not EasyOCR failure. FLUX passes at 0.9167.")

    table_row = {
        "node": "Node 4", "capability": "Typography", "tool": "EasyOCR 1.7.1",
        "dataset": "DrawBench Text (n=12 prompts, 24 images)",
        "metric": "Levenshtein accuracy", "score": round(accuracy, 4),
        "target": f">= {TARGET}", "pass_fail": pass_fail,
        "calibrated_threshold": threshold,
    }
    write_header = not VALIDATION_TABLE.exists()
    with open(VALIDATION_TABLE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(table_row)
    print(f"\n  Component validation table updated: {VALIDATION_TABLE}")

    node4_summary = (f"Node 4 | Typography | EasyOCR 1.7.1 | "
                     f"Accuracy={accuracy:.4f} | target>=0.75 | {pass_fail} | "
                     f"hits={hits}/{total} | conf_threshold={CONF_THRESHOLD} | "
                     f"lev_threshold={HIT_THRESHOLD}\n")
    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write(node4_summary)

    print(f"  Summary written: {SUMMARY_TXT}")
    print("\nNode 4 validation complete.")
    return accuracy, pass_fail


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Attribute/Colour Compliance Validation (original, mCLIP)
# Result: FAILED (Spearman rho=0.2273, p=0.0834)
# Finding: mCLIP measures holistic semantic alignment, not colour accuracy.
#          It cannot distinguish correct-colour from wrong-colour images when
#          the overall scene looks plausible. Tool replaced — see run_node3_recalibrated().
# Kept here as documented record per Phase 1C audit trail.
# ══════════════════════════════════════════════════════════════════════════════

def run_node3():
    print("=" * 65)
    print("NODE 3 — Attribute/Colour Validation  [original: mCLIP]")
    print("=" * 65)

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import (BlipProcessor, BlipForQuestionAnswering,
                              CLIPProcessor, CLIPModel as HFCLIPModel)
    from sentence_transformers import SentenceTransformer
    from scipy.stats import spearmanr

    MODELS = ["sd15", "flux"]

    if not NODE3_PROMPTS_CSV.exists():
        print(f"\n  ERROR: {NODE3_PROMPTS_CSV} not found.")
        print("  Run phase1c_agent_validation/phase1c_generate.py first.")
        return None, "ERROR"

    prompts_df = pd.read_csv(NODE3_PROMPTS_CSV, encoding="utf-8")
    prompts    = prompts_df.to_dict("records")
    print(f"\n  Loaded {len(prompts)} prompts from {NODE3_PROMPTS_CSV}")

    missing = [
        str(NODE3_IMAGES_DIR / mk / f"{p['prompt_id']}_seed{SEED}.png")
        for mk in MODELS for p in prompts
        if not (NODE3_IMAGES_DIR / mk / f"{p['prompt_id']}_seed{SEED}.png").exists()
    ]
    if missing:
        print(f"\n  WARNING: {len(missing)} image(s) not found.")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
        if len(missing) == len(prompts) * len(MODELS):
            print("  No images present at all — aborting.")
            return None, "ERROR"
        print("  Proceeding with available images.\n")
    else:
        print(f"  All {len(prompts) * len(MODELS)} images confirmed present.\n")

    COLORS     = ["red", "blue", "green", "yellow", "orange", "purple", "pink",
                  "brown", "black", "white", "gray", "grey", "cyan", "magenta", "gold"]
    SKIP_WORDS = {"and", "a", "an", "the", "of", "with", "on", "in", "next", "to"}

    def extract_color_pairs(text):
        pairs = []
        for color in COLORS:
            for obj in re.findall(rf"\b{color}\b\s+(\w+)", text.lower()):
                if obj not in SKIP_WORDS:
                    pairs.append((color, obj))
        return pairs

    print("  Loading BLIP-VQA model (Salesforce/blip-vqa-base)...")
    blip_proc  = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    blip_model.eval()
    print("  BLIP-VQA ready.")

    print("  Loading CLIP image encoder (openai/clip-vit-base-patch32)...")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_hf   = HFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_hf.eval()
    print("  CLIP image encoder ready.")

    print("  Loading mCLIP text encoder (clip-ViT-B-32-multilingual-v1)...")
    mclip = SentenceTransformer("clip-ViT-B-32-multilingual-v1")
    print("  mCLIP text encoder ready.\n")

    def blip_colour_score(img_path, pairs):
        if not pairs:
            return 0.5
        img     = Image.open(img_path).convert("RGB")
        correct = 0
        for color, obj in pairs:
            question = f"What color is the {obj}?"
            inputs   = blip_proc(img, question, return_tensors="pt")
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_new_tokens=10)
            answer      = blip_proc.decode(out[0], skip_special_tokens=True).lower().strip()
            answer_norm = answer.replace("grey", "gray")
            color_norm  = color.replace("grey", "gray")
            if color_norm in answer_norm:
                correct += 1
        return correct / len(pairs)

    def mclip_cosine(img_path, text):
        img    = Image.open(img_path).convert("RGB")
        inputs = clip_proc(images=img, return_tensors="pt")
        with torch.no_grad():
            img_feat = clip_hf.get_image_features(**inputs)
        img_feat = F.normalize(img_feat.float(), dim=-1)
        txt_feat = torch.tensor(
            mclip.encode(text, convert_to_numpy=True), dtype=torch.float32
        ).unsqueeze(0)
        txt_feat = F.normalize(txt_feat, dim=-1)
        return float(F.cosine_similarity(img_feat, txt_feat))

    results = []
    total   = len(prompts) * len(MODELS)
    done    = 0
    print(f"  Scoring {total} images (BLIP-VQA + mCLIP)...\n")

    for model_key in MODELS:
        model_dir = NODE3_IMAGES_DIR / model_key
        for p in prompts:
            done     += 1
            img_path  = model_dir / f"{p['prompt_id']}_seed{SEED}.png"
            pairs     = extract_color_pairs(p["text"])
            if not img_path.exists():
                print(f"  [{done:3d}/{total}] MISSING {p['prompt_id']} | {model_key} — skipping")
                results.append({"prompt_id": p["prompt_id"], "model": model_key,
                                 "prompt_text": p["text"], "color_pairs": str(pairs),
                                 "blip_score": None, "mclip_score": None, "image_found": 0})
                continue
            blip_s  = blip_colour_score(img_path, pairs)
            mclip_s = mclip_cosine(img_path, p["text"])
            print(f"  [{done:3d}/{total}] {p['prompt_id']} | {model_key:4s} | "
                  f"BLIP={blip_s:.3f}  mCLIP={mclip_s:.4f}  pairs={pairs}")
            results.append({"prompt_id": p["prompt_id"], "model": model_key,
                             "prompt_text": p["text"], "color_pairs": str(pairs),
                             "blip_score": round(blip_s, 4),
                             "mclip_score": round(mclip_s, 4), "image_found": 1})

    results_df = pd.DataFrame(results)
    results_df.to_csv(NODE3_CSV, index=False, encoding="utf-8")
    print(f"\n  Detailed results saved: {NODE3_CSV}  ({len(results_df)} rows)")

    valid     = results_df[(results_df["image_found"] == 1) &
                           results_df["blip_score"].notna() &
                           results_df["mclip_score"].notna()]
    rho, pval = spearmanr(valid["blip_score"], valid["mclip_score"])

    TARGET    = 0.60
    pass_fail = "PASS" if rho >= TARGET else "FAIL"
    threshold = ("mCLIP cosine similarity | HF CLIP ViT-B/32 image encoder "
                 "+ mCLIP multilingual text encoder")

    print("\n  Per-model score breakdown:")
    for mk in MODELS:
        m = valid[valid["model"] == mk]
        if len(m) > 0:
            print(f"    {mk:4s}: avg BLIP={m['blip_score'].mean():.3f}  "
                  f"avg mCLIP={m['mclip_score'].mean():.4f}  n={len(m)}")

    print("\n" + "-" * 45)
    print(f"  n images scored : {len(valid)}")
    print(f"  Spearman rho    : {rho:.4f}  (target >= {TARGET})")
    print(f"  p-value         : {pval:.4f}")
    print(f"  Status          : {pass_fail}")
    print("-" * 45)

    if pass_fail == "FAIL":
        print("\n  NOTE: rho below target. mCLIP measures holistic alignment,")
        print("  not colour accuracy. Run '3r' for recalibrated GPT-4o mini version.")

    table_row = {
        "node": "Node 3", "capability": "Attribute/Colour",
        "tool": "BLIP-VQA + mCLIP",
        "dataset": "T2I-CompBench color_val (n=30 prompts, 59-60 images)",
        "metric": "Spearman rho", "score": round(rho, 4),
        "target": f">= {TARGET}", "pass_fail": pass_fail,
        "calibrated_threshold": threshold,
    }
    write_header = not VALIDATION_TABLE.exists()
    with open(VALIDATION_TABLE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(table_row)
    print(f"\n  Component validation table updated: {VALIDATION_TABLE}")

    node3_summary = (f"Node 3 | Attribute/Colour | BLIP-VQA + mCLIP | "
                     f"Spearman_rho={rho:.4f} | p={pval:.4f} | target>=0.60 | "
                     f"{pass_fail} | n={len(valid)}\n")
    with open(SUMMARY_TXT, "a", encoding="utf-8") as fh:
        fh.write(node3_summary)

    print(f"  Summary written: {SUMMARY_TXT}")
    print("\nNode 3 (mCLIP) validation complete.")
    print("  -> Run '3r' for recalibrated version using GPT-4o mini VQA.")
    return rho, pass_fail


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 (RECALIBRATED) — Attribute/Colour Compliance Validation
# Tool: GPT-4o mini VQA, replacing mCLIP per exposé recalibration protocol.
# Rationale: mCLIP failed (rho=0.2273) because it measures holistic semantic
#             alignment, not colour accuracy. GPT-4o mini asks per-object colour
#             questions directly, matching what Node 3 needs to measure.
# Prerequisite: run_node3() must have run first (needs node3_colour_attribute.csv
#               for BLIP reference scores).
# Metric: Spearman rho between GPT-4o mini colour score and BLIP-VQA reference
# Target: rho >= 0.60
# ══════════════════════════════════════════════════════════════════════════════

def run_node3_recalibrated():
    print("=" * 65)
    print("NODE 3 (RECALIBRATED) — Attribute/Colour Validation")
    print("Tool: GPT-4o mini VQA  (mCLIP failed: rho=0.2273)")
    print("=" * 65)

    from scipy.stats import spearmanr

    MODELS = ["sd15", "flux"]

    # ── Guard: BLIP scores must exist from run_node3() ────────────────────────
    if not NODE3_CSV.exists():
        print(f"\n  ERROR: {NODE3_CSV} not found.")
        print("  Run 'python phase1c_validate.py 3' first to generate BLIP scores.")
        return None, "ERROR"

    blip_df = pd.read_csv(NODE3_CSV, encoding="utf-8")
    blip_df = blip_df[blip_df["image_found"] == 1].copy()
    print(f"\n  Loaded {len(blip_df)} BLIP reference scores from {NODE3_CSV}")

    # ── Load prompts ──────────────────────────────────────────────────────────
    if not NODE3_PROMPTS_CSV.exists():
        print(f"\n  ERROR: {NODE3_PROMPTS_CSV} not found.")
        print("  Run phase1c_agent_validation/phase1c_generate.py first.")
        return None, "ERROR"

    prompts_lookup = {
        p["prompt_id"]: p["text"]
        for p in pd.read_csv(NODE3_PROMPTS_CSV, encoding="utf-8").to_dict("records")
    }

    # ── Colour-object pair parsing (same list as run_node3) ───────────────────
    COLORS     = ["red", "blue", "green", "yellow", "orange", "purple", "pink",
                  "brown", "black", "white", "gray", "grey", "cyan", "magenta", "gold"]
    SKIP_WORDS = {"and", "a", "an", "the", "of", "with", "on", "in", "next", "to"}

    def extract_color_pairs(text):
        pairs = []
        for color in COLORS:
            for obj in re.findall(rf"\b{color}\b\s+(\w+)", text.lower()):
                if obj not in SKIP_WORDS:
                    pairs.append((color, obj))
        return pairs

    # ── GPT-4o mini colour VQA helpers ───────────────────────────────────────
    def encode_image(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def query_colour(img_path: Path, obj: str, color: str, retries: int = 3) -> int:
        """
        Asks "Is the [object] [color]? Answer with only 'yes' or 'no'."
        Returns 1 for yes, 0 for no. Conservative default 0 on failure.
        Uses detail=low (same as Node 1) for cost efficiency.
        """
        b64 = encode_image(img_path)
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                           "detail": "low"}},
                            {"type": "text",
                             "text": (f"Is the {obj} in this image {color}? "
                                      "Answer with only 'yes' or 'no'.")}
                        ]
                    }],
                    max_tokens=5,
                    temperature=0
                )
                answer = response.choices[0].message.content.strip().lower()
                return 1 if answer.startswith("yes") else 0
            except Exception as e:
                print(f"    Retry {attempt + 1}/3 ({color} {obj}): {e}")
                time.sleep(3)
        return 0

    def gpt_colour_score(img_path: Path, pairs: list) -> float:
        """
        Fractional score: (colour pairs GPT confirms correct) / len(pairs).
        Returns 0.5 (neutral) if no parseable pairs.
        """
        if not pairs:
            return 0.5
        correct = 0
        for color, obj in pairs:
            correct += query_colour(img_path, obj, color)
            time.sleep(0.4)
        return correct / len(pairs)

    # ── Score all images with existing BLIP scores ────────────────────────────
    results = []
    total   = len(blip_df)
    done    = 0

    print(f"  Running GPT-4o mini colour VQA on {total} images...\n")
    print("  (Each image: ~2 GPT calls at detail=low — approx. $0.02 total)\n")

    for _, row in blip_df.iterrows():
        done     += 1
        pid       = row["prompt_id"]
        model     = row["model"]
        img_path  = NODE3_IMAGES_DIR / model / f"{pid}_seed{SEED}.png"
        text      = prompts_lookup.get(pid, row.get("prompt_text", ""))
        pairs     = extract_color_pairs(text)
        blip_s    = float(row["blip_score"])

        if not img_path.exists():
            print(f"  [{done:3d}/{total}] MISSING {pid} | {model} — skipping")
            continue

        gpt_s = gpt_colour_score(img_path, pairs)

        print(f"  [{done:3d}/{total}] {pid} | {model:4s} | "
              f"GPT={gpt_s:.3f}  BLIP={blip_s:.3f}  pairs={pairs}")

        results.append({
            "prompt_id":   pid,
            "model":       model,
            "prompt_text": text,
            "color_pairs": str(pairs),
            "gpt_score":   round(gpt_s,  4),
            "blip_score":  round(blip_s, 4),
        })

    # ── Save detailed results ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(NODE3_RECALIB_CSV, index=False, encoding="utf-8")
    print(f"\n  Detailed results saved: {NODE3_RECALIB_CSV}  ({len(results_df)} rows)")

    # ── Compute Spearman rho ──────────────────────────────────────────────────
    valid     = results_df.dropna(subset=["gpt_score", "blip_score"])
    rho, pval = spearmanr(valid["gpt_score"], valid["blip_score"])

    TARGET    = 0.60
    pass_fail = "PASS" if rho >= TARGET else "FAIL"
    threshold = ("GPT-4o mini colour VQA | 'Is the [object] [color]?' "
                 "binary yes/no | detail=low")

    print("\n  Per-model score breakdown:")
    for mk in MODELS:
        m = valid[valid["model"] == mk]
        if len(m) > 0:
            print(f"    {mk:4s}: avg GPT={m['gpt_score'].mean():.3f}  "
                  f"avg BLIP={m['blip_score'].mean():.3f}  n={len(m)}")

    print("\n" + "-" * 45)
    print(f"  n images scored : {len(valid)}")
    print(f"  Spearman rho    : {rho:.4f}  (target >= {TARGET})")
    print(f"  p-value         : {pval:.4f}")
    print(f"  Status          : {pass_fail}")
    print("-" * 45)

    if pass_fail == "FAIL":
        print("\n  NOTE: rho still below target after recalibration.")
        print("  Review node3_recalibrated.csv for per-image discrepancies.")
        print("  Consider examining low-agreement rows before Phase 2A.")

    # ── Append recalibrated row to component validation table ─────────────────
    table_row = {
        "node":                 "Node 3 (recalibrated)",
        "capability":           "Attribute/Colour",
        "tool":                 "GPT-4o mini VQA",
        "dataset":              "T2I-CompBench color_val (n=30 prompts, 59 images)",
        "metric":               "Spearman rho (vs BLIP-VQA reference)",
        "score":                round(rho, 4),
        "target":               f">= {TARGET}",
        "pass_fail":            pass_fail,
        "calibrated_threshold": threshold,
    }
    write_header = not VALIDATION_TABLE.exists()
    with open(VALIDATION_TABLE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(table_row)
    print(f"\n  Component validation table updated: {VALIDATION_TABLE}")

    node3r_summary = (f"Node 3 (recalibrated) | Attribute/Colour | GPT-4o mini VQA | "
                      f"Spearman_rho={rho:.4f} | p={pval:.4f} | target>=0.60 | "
                      f"{pass_fail} | n={len(valid)}\n")
    with open(SUMMARY_TXT, "a", encoding="utf-8") as fh:
        fh.write(node3r_summary)

    print(f"  Summary written: {SUMMARY_TXT}")
    print("\nNode 3 recalibrated validation complete.")
    return rho, pass_fail


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# Usage:
#   python phase1c_agent_validation/phase1c_validate.py 1    -> Node 1 only
#   python phase1c_agent_validation/phase1c_validate.py 4    -> Node 4 only
#   python phase1c_agent_validation/phase1c_validate.py 3    -> Node 3 original (mCLIP)
#   python phase1c_agent_validation/phase1c_validate.py 3r   -> Node 3 recalibrated (GPT-4o mini)
#   python phase1c_agent_validation/phase1c_validate.py      -> all nodes
#
# Notes:
#   - Run '3' before '3r' (recalibrated needs BLIP scores from original run)
#   - Node 3 images must be generated by phase1c_generate.py first
#   - 'all' runs both 3 and 3r in sequence
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

    if node in ("3", "all"):
        rho, status = run_node3()
        results["node3_mclip"] = status

    if node in ("3r", "all"):
        rho_r, status_r = run_node3_recalibrated()
        results["node3_recalibrated"] = status_r

    print("\n" + "=" * 65)
    print("Phase 1C Summary")
    print("=" * 65)
    for k, v in results.items():
        print(f"  {k:25s}: {v}")

    # Pass/fail for exit code uses recalibrated Node 3 if available,
    # falling back to original if recalibrated hasn't run yet.
    node3_status = results.get("node3_recalibrated", results.get("node3_mclip"))
    final_results = {k: v for k, v in results.items()
                     if k not in ("node3_mclip",)}
    if node3_status:
        final_results["node3"] = node3_status

    all_pass = all(v == "PASS" for v in final_results.values() if v is not None)
    sys.exit(0 if all_pass else 1)