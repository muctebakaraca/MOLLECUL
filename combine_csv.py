import pandas as pd
import glob
import re

# Find all matching CSV files in the current directory
pattern = "PART*_NEW_Dallas_Properties.csv"
files = glob.glob(pattern)

if not files:
    print(f"No files found matching pattern: {pattern}")
    exit(1)

# Sort numerically by part number (PART1, PART2, ..., PART10, not PART1, PART10, PART11)
files = sorted(files, key=lambda f: int(re.search(r"PART(\d+)", f).group(1)))

print(f"Found {len(files)} file(s) to combine:")
for f in files:
    print(f"  - {f}")
print()

# Read and combine all CSVs (low_memory=False suppresses mixed-type warnings)
dfs = []
for f in files:
    df = pd.read_csv(f, low_memory=False)
    print(f"  {f}: {len(df):,} rows")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# Save the result
output_file = "Combined_Dallas_Properties.csv"
combined.to_csv(output_file, index=False)

print(f"\nDone! Combined {len(dfs)} files → '{output_file}'")
print(f"Total rows: {len(combined):,}")