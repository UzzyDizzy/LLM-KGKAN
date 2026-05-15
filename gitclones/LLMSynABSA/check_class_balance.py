# check_class_weights.py

import os
import pandas as pd
from collections import Counter
import torch

DATA_DIR = "data"

FILES = [
    "restaurant.csv",
    "laptop.csv",
    "device.csv",
    "service.csv",
]

ALL_LABELS = [
    "O",
    "B-POS",
    "I-POS",
    "B-NEG",
    "I-NEG",
    "B-NEU",
    "I-NEU",
]

# =========================================================
# Helper
# =========================================================

def polarity_to_tags(pol):
    if pol == "positive":
        return ("B-POS", "I-POS")
    elif pol == "negative":
        return ("B-NEG", "I-NEG")
    else:
        return ("B-NEU", "I-NEU")


# =========================================================
# Count label frequencies
# =========================================================

global_counter = Counter()

for file in FILES:

    path = os.path.join(DATA_DIR, file)

    print(f"\n==============================")
    print(f"Processing: {file}")
    print(f"==============================")

    df = pd.read_csv(path)

    counter = Counter()

    # Count aspect labels
    for _, row in df.iterrows():

        polarity = row["polarity"]

        aspect = str(row["aspect"]).strip()

        # estimate token count
        n_tokens = len(aspect.split())

        b_tag, i_tag = polarity_to_tags(polarity)

        counter[b_tag] += 1

        if n_tokens > 1:
            counter[i_tag] += (n_tokens - 1)

    # Estimate O count from sentence lengths
    total_tokens = 0
    used_tokens = 0

    grouped = df.groupby("text")

    for text, g in grouped:

        sent_tokens = len(str(text).split())
        total_tokens += sent_tokens

        for _, row in g.iterrows():
            aspect = str(row["aspect"]).strip()
            used_tokens += len(aspect.split())

    counter["O"] = total_tokens - used_tokens

    # ensure all labels exist
    for lab in ALL_LABELS:
        if lab not in counter:
            counter[lab] = 0

    total = sum(counter.values())

    print("\nLabel counts:")
    for k in ALL_LABELS:
        print(f"{k:8s}: {counter[k]}")

    print("\nFrequencies:")
    freqs = {}

    for k in ALL_LABELS:
        freqs[k] = counter[k] / total
        print(f"{k:8s}: {freqs[k]:.6f}")

    # inverse-frequency weights
    weights = {}

    for k in ALL_LABELS:
        weights[k] = 1.0 / (freqs[k] + 1e-8)

    # normalize
    s = sum(weights.values())

    for k in weights:
        weights[k] = weights[k] / s * len(weights)

    print("\nNormalized class weights:")
    for k in ALL_LABELS:
        print(f"{k:8s}: {weights[k]:.4f}")

    # tensor in correct order
    weight_tensor = torch.tensor(
        [weights[k] for k in ALL_LABELS],
        dtype=torch.float32
    )

    print("\nPyTorch tensor:")
    print(weight_tensor)

    # accumulate globally
    global_counter.update(counter)

# =========================================================
# GLOBAL weights across all datasets
# =========================================================

print("\n\n############################################")
print("GLOBAL CLASS WEIGHTS")
print("############################################")

total = sum(global_counter.values())

global_freqs = {}

for k in ALL_LABELS:
    global_freqs[k] = global_counter[k] / total

weights = {}

for k in ALL_LABELS:
    weights[k] = 1.0 / (global_freqs[k] + 1e-8)

s = sum(weights.values())

for k in weights:
    weights[k] = weights[k] / s * len(weights)

print("\nGlobal frequencies:")
for k in ALL_LABELS:
    print(f"{k:8s}: {global_freqs[k]:.6f}")

print("\nGlobal normalized weights:")
for k in ALL_LABELS:
    print(f"{k:8s}: {weights[k]:.4f}")

weight_tensor = torch.tensor(
    [weights[k] for k in ALL_LABELS],
    dtype=torch.float32
)

print("\nFINAL tensor for CrossEntropyLoss:")
print(weight_tensor)

print("\nUse this in training:\n")

print("class_weights =", weight_tensor.tolist())