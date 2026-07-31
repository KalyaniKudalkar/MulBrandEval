"""
Phase 2A — DAG Smoke Test
Run from project root: python phase2a_dag_pipeline/smoke_test.py
"""
import os
import pandas as pd
from dotenv import load_dotenv
from phase2a_dag_pipeline.pipeline import app

# Load .env so OPENAI_API_KEY is available
load_dotenv()

# ── Load first brief from MBB CSV ─────────────────────────────────────────────
df    = pd.read_csv("data/prompts/mbb/mbb_briefs_en.csv")
brief = df.iloc[0].to_dict()

# Phase 2A is English-only — prompt_text_en is always identical to prompt_text.
# This becomes a real distinction only in Phase 2B, where prompt_text is
# translated and prompt_text_en stays locked to the English original for
# Node 5 (Point 2 resolution — see node5_brand_quality.py).
brief["prompt_text_en"] = brief["prompt_text"]

# ── Use a Phase 1A image as test input ────────────────────────────────────────
# This is a COCO image, NOT an MBB image — wrong content on purpose.
# Expect Node 1 FAIL + short-circuit. Proves the DAG routing works correctly.
brief["image_path"] = "data/generated_images/phase1a/flux/ar/P001_seed42.png"
brief["model"]      = "flux"
brief["language"]   = "en"

print("=" * 60)
print("SMOKE TEST — MB001 + Phase1A image (expect Node 1 FAIL)")
print("=" * 60)
print(f"Brief : {brief['brief_id']} | {brief['brand_name']} | {brief['industry']}")
print(f"Image : {brief['image_path']}")
print()

# ── Run the full DAG ──────────────────────────────────────────────────────────
result = app.invoke(brief)

# ── Print results ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"Compliance Score  : {result['compliance_score']}")
print(f"Failure Diagnosis : {result['failure_diagnosis']}")
print()
print("PER-NODE SUMMARY")
print("-" * 60)
r = result["compliance_report"]
print(f"Node 1 — Object Presence  : pass={r['node1']['pass']}, score={r['node1']['score']}, short_circuit={r['node1']['short_circuit']}")
print(f"Node 2 — Spatial Layout   : pass={r['node2']['pass']}, skipped={r['node2']['skipped']}")
print(f"Node 3 — Colour Attribute : pass={r['node3']['pass']}, skipped={r['node3']['skipped']}")
print(f"Node 4 — Typography       : pass={r['node4']['pass']}, skipped={r['node4']['skipped']}, clcg_interpretable={r['node4']['clcg_interpretable']}")
print(f"Node 5 — Brand Quality    : pass={r['node5']['pass']}, score={r['node5']['score']}")
print()
print("SMOKE TEST COMPLETE")