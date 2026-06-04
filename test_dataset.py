from datasets import load_dataset
import pandas as pd
import numpy as np
import torch


DATASET_NAME = "./datasets"


ds = load_dataset(DATASET_NAME)

print("\nDATASET SPLITS:")
print(ds)

split = list(ds.keys())[0]  # usually 'train'
data = ds[split]

print("\nCOLUMNS:")
print(data.column_names)

print("\nTOTAL SAMPLES:")
print(len(data))

print("\nSAMPLE ROW (RAW):")
sample = data[0]
print(sample)

print("\nFIELD TYPES:")
for k, v in sample.items():
    print(f"\nKEY: {k}")
    print(f"TYPE: {type(v)}")
    print(f"VALUE PREVIEW: {str(v)[:300]}")


print("\nAUDIO FIELD DEEP INSPECTION:")

audio = sample.get("audio", None)

print("audio type:", type(audio))

# try dict-style inspection safely
if isinstance(audio, dict):
    print("audio keys:", audio.keys())
    for k in audio:
        print(f"{k} -> type: {type(audio[k])}")
else:
    print("audio raw:", audio)


print("\nCHECK MULTIPLE SAMPLES FOR CONSISTENCY:")

for i in range(3):
    try:
        row = data[i]
        print(f"\n--- sample {i} ---")
        print("keys:", row.keys())
        print("audio type:", type(row["audio"]))

        if isinstance(row["audio"], dict):
            print("audio keys:", row["audio"].keys())

    except Exception as e:
        print("error at sample", i, e)
