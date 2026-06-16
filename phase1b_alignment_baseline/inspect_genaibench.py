from datasets import load_dataset
import pandas as pd

print("Loading GenAI-Bench...")
ds = load_dataset("BaiqiL/GenAI-Bench", split="train")
df = ds.to_pandas()

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== TOTAL ROWS ===")
print(len(df))

print("\n=== SAMPLE OF Tags COLUMN (first 5) ===")
for i, val in enumerate(df["Tags"].head(5)):
    print(f"  Row {i}: {val}")

print("\n=== SAMPLE OF HumanRatings COLUMN (first 3) ===")
for i, val in enumerate(df["HumanRatings"].head(3)):
    print(f"  Row {i}: {val}")

print("\n=== FIRST 3 ROWS (Prompt + Tags only) ===")
print(df[["Index", "Prompt", "Tags"]].head(3).to_string())