"""
Phase 2A — Node 5 Fix Verification
Confirms prompt_text_en is correctly wired through the state and that
Node 5 runs without KeyError, using a real brief + real generated image
(so Node 1 should PASS and the full DAG chain executes through Node 5).

Run from project root: python -m phase2a_dag_pipeline.test_node5_fix
"""
import pandas as pd
from dotenv import load_dotenv
from phase2a_dag_pipeline.pipeline import app

load_dotenv()

df    = pd.read_csv("data/prompts/mbb/mbb_briefs_en.csv")
brief = df[df["brief_id"] == "MB001"].iloc[0].to_dict()

brief["prompt_text_en"] = brief["prompt_text"]  # Phase 2A: identical, as designed
brief["image_path"]     = "data/generated_images/phase2a/flux/en/MB001_seed42.png"
brief["model"]          = "flux"
brief["language"]       = "en"

print("=" * 60)
print("NODE 5 FIX VERIFICATION — MB001 + real FLUX image (expect full pass-through)")
print("=" * 60)
print(f"Brief : {brief['brief_id']} | {brief['brand_name']} | {brief['industry']}")
print(f"Image : {brief['image_path']}")
print()

result = app.invoke(brief)

print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"Compliance Score  : {result['compliance_score']}")
print(f"Failure Diagnosis : {result['failure_diagnosis']}")
print()
r = result["compliance_report"]
print(f"Node 1 — Object Presence  : pass={r['node1']['pass']}, score={r['node1']['score']}, short_circuit={r['node1']['short_circuit']}")
print(f"Node 5 — Brand Quality    : pass={r['node5']['pass']}, score={r['node5']['score']}")
print()
if r["node5"]["pass"] is not None:
    print("✅ Node 5 EXECUTED — prompt_text_en fix confirmed working (no KeyError)")
else:
    print("⚠️  Node 5 did not run — check whether Node 1 short-circuited")