import pandas as pd
import glob
import os

# Find all CSV files matching the Dallas properties pattern (case-insensitive)
all_files = glob.glob("*.csv")
matching_files = [f for f in all_files if "dallas" in f.lower() and "properties" in f.lower()]

if not matching_files:
    print("No matching CSV files found in the current directory.")
    exit()

print(f"Found {len(matching_files)} files:")
for f in matching_files:
    print(f"  - {f}")

# Read and append all CSVs
dfs = []
skipped_files = []

for file in matching_files:
    try:
        df = pd.read_csv(
            file,
            low_memory=False,       # fixes mixed type warnings
            on_bad_lines="skip",    # skips malformed rows instead of crashing
        )
        df["_source_file"] = file
        dfs.append(df)
        print(f"Loaded {len(df):,} rows from {file}")
    except Exception as e:
        print(f"  ⚠️  Skipped {file} — {e}")
        skipped_files.append(file)

if not dfs:
    print("No files could be loaded.")
    exit()

combined = pd.concat(dfs, ignore_index=True)

# Remove duplicate rows if any
before = len(combined)
combined.drop_duplicates(inplace=True)
after = len(combined)
if before != after:
    print(f"\nRemoved {before - after:,} duplicate rows.")

output_file = "Dallas_properties_combined.csv"
combined.to_csv(output_file, index=False)

print(f"\n✅ Done! Combined {len(dfs)} files into '{output_file}'")
print(f"   Total rows: {len(combined):,}")
print(f"   Total columns: {len(combined.columns):,}")

if skipped_files:
    print(f"\n⚠️  {len(skipped_files)} file(s) were skipped entirely:")
    for f in skipped_files:
        print(f"   - {f}")