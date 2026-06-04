import pandas as pd
import json
import glob
import os

# Find all parquet files in the datasets/ folder
parquet_files = glob.glob("datasets/*.parquet", recursive=True)
# Or if they're directly in datasets/:
# parquet_files = glob.glob("datasets/*.parquet")

print(f"Found {len(parquet_files)} parquet files: {parquet_files}")

output_file = "newari_text.jsonl"

with open(output_file, "w", encoding="utf-8") as f_out:
    for parquet_path in sorted(parquet_files):
        print(f"Processing: {parquet_path}")
        df = pd.read_parquet(parquet_path, columns=["sentence"])  # skip loading audio bytes
        
        for sentence in df["sentence"]:
            if sentence and str(sentence).strip():  # skip empty rows
                record = {"text": str(sentence).strip()}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Done! Output saved to: {output_file}")

# Quick preview
with open(output_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 4:
            break
