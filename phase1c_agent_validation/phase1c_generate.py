import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import time
import requests
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
SEED         = 42
RESULTS_DIR  = Path("results/phase1c")
NODE3_IMGS   = Path("data/generated_images/phase1c/node3")
PROMPTS_CSV  = Path("data/prompts/en/phase1c_node3_prompts_en.csv")   # read by phase1c_validate.py

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NODE3_IMGS.mkdir(parents=True, exist_ok=True)

REPLICATE_MODELS = {
    "sd15": {
        "id":     ("stability-ai/stable-diffusion:"
                   "ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"),
        "params": {"num_inference_steps": 50, "guidance_scale": 7.5, "seed": SEED},
    },
    "flux": {
        "id":     "black-forest-labs/flux-dev",
        "params": {"num_inference_steps": 50, "guidance": 3.5, "seed": SEED},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# Generates 30 sampled T2I-CompBench color_val prompts × 2 models = 60 images.
# Saves sampled prompts to data/prompts/en/phase1c_node3_prompts_en.csv — this CSV is the
# source of truth read by run_node3() in phase1c_validate.py.
#
# Usage:
#   python phase1c_agent_validation/phase1c_generate.py
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import replicate
    from datasets import load_dataset

    print("=" * 65)
    print("Phase 1C — Node 3 Image Generation")
    print("T2I-CompBench color_val  ×  2 models  =  60 images")
    print("=" * 65)

    # ── Sample 30 prompts from color_val ──────────────────────────────────────
    print("\nLoading T2I-CompBench color_val (sayakpaul/t2i-compbench)...")
    ds = load_dataset("sayakpaul/t2i-compbench", "color_val", split="val")
    print(f"  Total prompts available: {len(ds)}")

    random.seed(SEED)
    selected_indices = sorted(random.sample(range(len(ds)), 30))
    prompts = [
        {"prompt_id": f"CV{i + 1:03d}", "text": ds[idx]["text"]}
        for i, idx in enumerate(selected_indices)
    ]

    print(f"\n  Sampled {len(prompts)} prompts (seed={SEED}):")
    for p in prompts:
        print(f"    {p['prompt_id']} → {p['text']}")

    # Save prompts CSV — source of truth for validate script
    pd.DataFrame(prompts).to_csv(PROMPTS_CSV, index=False, encoding="utf-8")
    print(f"\n  Prompts saved to: {PROMPTS_CSV}")

    # ── Generate images ───────────────────────────────────────────────────────
    total     = len(prompts) * len(REPLICATE_MODELS)
    generated = 0
    skipped   = 0
    errors    = 0

    print(f"\nGenerating {len(prompts)} prompts × {len(REPLICATE_MODELS)} models"
          f" = {total} images...\n")

    for model_key, cfg in REPLICATE_MODELS.items():
        model_dir = NODE3_IMGS / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"  --- {model_key} ---")

        for idx, p in enumerate(prompts, 1):
            img_path = model_dir / f"{p['prompt_id']}_seed{SEED}.png"

            if img_path.exists():
                print(f"  [SKIP] [{idx:2d}/{len(prompts)}] {p['prompt_id']}"
                      f" | {model_key} — already exists")
                skipped += 1
                continue

            try:
                output = replicate.run(
                    cfg["id"],
                    input={"prompt": p["text"], **cfg["params"]}
                )

                # Handle all Replicate output formats
                if isinstance(output, list):
                    raw = output[0]
                elif hasattr(output, "__iter__") and not isinstance(output, str):
                    raw = next(iter(output))
                else:
                    raw = output

                if hasattr(raw, "read"):
                    img_bytes = raw.read()
                elif hasattr(raw, "url"):
                    img_bytes = requests.get(raw.url, timeout=30).content
                else:
                    img_bytes = requests.get(str(raw), timeout=30).content

                with open(img_path, "wb") as fh:
                    fh.write(img_bytes)

                print(f"  [GEN ] [{idx:2d}/{len(prompts)}] {p['prompt_id']}"
                      f" | {model_key} → saved")
                generated += 1
                time.sleep(0.5)

            except Exception as e:
                print(f"  [ERR ] [{idx:2d}/{len(prompts)}] {p['prompt_id']}"
                      f" | {model_key}: {e}")
                errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Generation complete.")
    print(f"  Generated : {generated}")
    print(f"  Skipped   : {skipped}  (already existed)")
    print(f"  Errors    : {errors}")
    print(f"  Present   : {generated + skipped} / {total} images")
    print("=" * 65)

    if errors > 0:
        print(f"\n  WARNING: {errors} image(s) failed. Re-run this script to retry.")
        print("  Errored images are not skipped on retry (file was not saved).")

    print("\nNext step:")
    print("  python phase1c_agent_validation/phase1c_validate.py 3")


if __name__ == "__main__":
    main()