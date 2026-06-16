import pandas as pd

URL = "https://docs.google.com/spreadsheets/d/1y7nAbmR4FREi6npB1u-Bo3GFdwdOPYJc617rBOxIRHY/gviz/tq?tqx=out:csv"

print("Loading DrawBench from Google Sheets...")
df = pd.read_csv(URL)
df.columns = df.columns.str.lower()

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== CATEGORY NAMES AND COUNTS ===")
print(df["category"].value_counts().to_string())

print("\n=== FIRST 3 ROWS ===")
print(df.head(3).to_string())