"""
Phase 2A — MB001 Model Comparison (SD v1.5 vs FLUX)
Runs the full DAG on both models for the same brief and prints a
side-by-side node-by-node comparison.

Run from project root: python -m phase2a_dag_pipeline.test_mb001_comparison
"""
import pandas as pd
from dotenv import load_dotenv
from phase2a_dag_pipeline.pipeline import app

load_dotenv()

df    = pd.read_csv("data/prompts/mbb/mbb_briefs_en.csv")
brief_row = df[df["brief_id"] == "MB001"].iloc[0].to_dict()

IMAGE_PATHS = {
    "sd15": "data/generated_images/phase2a/sd15/en/MB001_seed42.png",
    "flux": "data/generated_images/phase2a/flux/en/MB001_seed42.png",
}

results = {}

for model_key, image_path in IMAGE_PATHS.items():
    brief = brief_row.copy()
    brief["prompt_text_en"] = brief["prompt_text"]  # Phase 2A: English-only, identical
    brief["image_path"]     = image_path
    brief["model"]          = model_key
    brief["language"]       = "en"

    print("=" * 60)
    print(f"RUNNING DAG — MB001 + {model_key.upper()}")
    print("=" * 60)
    print(f"Image : {image_path}")
    print()

    result = app.invoke(brief)
    results[model_key] = result

    print()

# ── Side-by-side comparison ─────────────────────────────────────────────────
print("=" * 70)
print("MB001 — SD v1.5 vs FLUX.1-dev — SIDE-BY-SIDE COMPARISON")
print("=" * 70)

sd  = results["sd15"]["compliance_report"]
fx  = results["flux"]["compliance_report"]

def row(label, sd_val, fx_val):
    print(f"{label:<28} {str(sd_val):<20} {str(fx_val):<20}")

print(f"{'':28} {'SD v1.5':<20} {'FLUX.1-dev':<20}")
print("-" * 70)
row("Node 1 (objects)",     f"pass={sd['node1']['pass']}",  f"pass={fx['node1']['pass']}")
row("  -> short_circuit",   sd['node1']['short_circuit'],   fx['node1']['short_circuit'])
row("Node 2 (spatial)",     f"pass={sd['node2']['pass']}",  f"pass={fx['node2']['pass']}")
row("Node 3 (colour)",      f"pass={sd['node3']['pass']}",  f"pass={fx['node3']['pass']}")
row("Node 4 (typography)",  f"pass={sd['node4']['pass']}",  f"pass={fx['node4']['pass']}")
row("  -> extracted text",  repr(sd['node4']['extracted']), repr(fx['node4']['extracted']))
row("Node 5 (PickScore)",   f"{sd['node5']['score']}",      f"{fx['node5']['score']}")
print("-" * 70)
row("COMPLIANCE SCORE",     sd['compliance_score'],         fx['compliance_score'])
print("=" * 70)
print()
print(f"SD v1.5 diagnosis : {results['sd15']['failure_diagnosis']}")
print(f"FLUX    diagnosis : {results['flux']['failure_diagnosis']}")