# config.py
from dataclasses import dataclass
import torch
import os

@dataclass
class Config:
    # ---------------- paths ----------------
    root_dir = "./"
    data_dir = "./data"
    raw_dir = "./raw_corpora"
    model_dir = "./saved_models"
    emb_dir = "./embeddings"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    # ---------------- datasets ----------------
    datasets = {
        "restaurant": "restaurant.csv",
        "laptop": "laptop.csv",
        "device": "device.csv",
        "service": "service.csv",
        "airline": "airline.csv",
        "shoes": "shoes.csv",
        "water_purifier": "water_purifier.csv",
        "university_course": "university_course.csv",
        "healthcare": "healthcare.csv"
    }

    transfer_pairs = [
        ("device", "restaurant"),
        ("device", "service"),
        ("laptop", "restaurant"),
        ("laptop", "service"),
        ("restaurant", "device"),
        ("restaurant", "laptop"),
        ("restaurant", "service"),
        ("service", "device"),
        ("service", "laptop"),
        ("service", "restaurant"),
    ]

    # ---------------- labels ----------------
    labels = [
        "O",
        "B-POS", "I-POS",
        "B-NEG", "I-NEG",
        "B-NEU", "I-NEU"
    ]

    label2id = {x:i for i,x in enumerate(labels)}
    id2label = {i:x for i,x in enumerate(labels)}

    # ---------------- model ----------------
    word_dim = 100
    pos_dim = 15
    hidden = 100
    num_layers = 2
    dropout = 0.5

    # ---------------- train ----------------
    batch_size = 32
    lr = 0.001
    epochs = 25
    seeds = [1,2,3,4,5]

    num_workers = 2
    pin_memory = True

    # ---------------- paper hyperparams ----------------
    eta = 1.0
    lambda_adv = 0.1
    gamma = 0.98
    beta = 60
    rho1 = 0.9
    rho2 = 0.4

    max_len = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"