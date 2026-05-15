# config.py — Hyperparameters matching paper Section 4.3
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

MAX_LEN = 96
EPOCHS = 100
EPOCHS_SYNTAX = 50          # "the maximum epoch are set to ... 50"
LR = 5e-5                   # "learning rate set to 5e-5"
LR_SYNTAX = 1e-4            # "gradually tune from {5e-4, 1e-5, 5e-5}"
DROPOUT = 0.2               # "dropout rate of 0.2"
RHO = 0.5                   # ρ balances L_adv vs L_seq (Eq 27)
BATCH_SIZE = 32              # true batch (accumulate to 32)
ACCUM_STEPS = 2              # effective batch = 16*2 = 32
NUM_PROMPTS = 10             # number of soft prompt embeddings

FREEZE_ENCODER = False        # Freeze LLaMA encoder (48GB GPU budget)

import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]

device = "cuda"

# --- Unified label set (paper Section 3.1) ---
LABELS = ["O", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NEU", "I-NEU"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}
ID_MAP = {i: l for l, i in LABEL_MAP.items()}
NUM_LABELS = len(LABELS)

# --- Auxiliary classifiers (Eq 2, 3) ---
ASPECT_LABELS = ["O", "B", "I"]
SENTIMENT_LABELS = ["O", "POS", "NEG", "NEU"]

ASPECT_LABEL_MAP = {l: i for i, l in enumerate(ASPECT_LABELS)}
ASPECT_ID_MAP = {i: l for l, i in ASPECT_LABEL_MAP.items()}

SENTIMENT_LABEL_MAP = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID_MAP = {i: l for l, i in SENTIMENT_LABEL_MAP.items()}