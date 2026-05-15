"""
KETGM Configuration — all hyperparameters from the paper (Table II).
Paths auto-detect project root for both local and Colab environments.
"""
import os
import torch

class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

    # ── URLs ───────────────────────────────────────────────────────────
    CONCEPTNET_URL = (
        "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
        "conceptnet-assertions-5.7.0.csv.gz"
    )
    CONCEPTNET_CSV_GZ = os.path.join(DATA_DIR, "conceptnet-assertions-5.7.0.csv.gz")
    CONCEPTNET_EN_PKL = os.path.join(DATA_DIR, "conceptnet_en.pkl")

    # ── CSV Data (4 domains) ──────────────────────────────────────────
    DOMAIN_CSV = {
        "restaurant": os.path.join(DATA_DIR, "restaurant.csv"),
        "laptop":     os.path.join(DATA_DIR, "laptop.csv"),
        "device":     os.path.join(DATA_DIR, "device.csv"),
        "service":    os.path.join(DATA_DIR, "service.csv"),
    }
    SOURCE_DOMAIN = "restaurant"   # default source domain
    TARGET_DOMAIN = "laptop"       # default target domain

    # ── BERT ───────────────────────────────────────────────────────────
    BERT_MODEL_NAME = "bert-base-uncased"
    BERT_HIDDEN = 768
    BERT_LR = 2e-5
    BERT_PRETRAIN_EPOCHS = 5

    # ── R-GCN ──────────────────────────────────────────────────────────
    RGCN_HIDDEN = 200
    RGCN_NUM_LAYERS = 2
    RGCN_NUM_BASES = 30
    RGCN_EPOCHS = 400
    RGCN_LR = 1e-3
    NEG_RATIO = 1          # negative-to-positive ratio for link prediction

    # ── LDA / Topics ──────────────────────────────────────────────────
    NUM_TOPICS = 6         # m in the paper
    WORDS_PER_TOPIC = 10   # k in the paper
    TOTAL_TOPIC_WORDS = NUM_TOPICS * WORDS_PER_TOPIC  # 60

    # ── ConceptNet ─────────────────────────────────────────────────────
    MAX_HOPS = 2

    # ── Sequence labelling ─────────────────────────────────────────────
    ABSA_TAGS = ["O", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NEU", "I-NEU"]
    NUM_TAGS = len(ABSA_TAGS)
    TAG2IDX = {t: i for i, t in enumerate(ABSA_TAGS)}
    IDX2TAG = {i: t for i, t in enumerate(ABSA_TAGS)}

    # ── Training ───────────────────────────────────────────────────────
    BATCH_SIZE = 32           # Paper Table II
    MAX_SEQ_LEN = 128
    NUM_EPOCHS = 30
    MU = 1.0                  # reconstruction-loss weight (Eq. 10) — Paper Table II
    CLASSIFIER_DROPOUT = 0.8  # Paper Table II
    KETGM_LR = 2e-5           # Paper Table II
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    TRAIN_RATIO = 0.8
    SEED = 42

    # ── Device ─────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories."""
        for d in [cls.DATA_DIR, cls.RAW_DIR, cls.PROCESSED_DIR, cls.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)
