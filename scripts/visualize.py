"""
scripts/visualize.py — All plotting functions for the 5 paper figures.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    RESULTS_DIR, MODEL_DISPLAY_NAMES, STANDARD_PAIRS,
    FEWSHOT_PAIRS, ZEROSHOT_PAIRS, FEWSHOT_K_VALUES,
)
from scripts.evaluate import load_result

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

def fig_fewshot_sensitivity(models, tgt_domains, save_path=None):
    """Figure 3/4: Few-shot sensitivity line plot (k vs Macro-F1)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("husl", len(models))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    for i, model in enumerate(models):
        ks, f1s = [], []
        for k in FEWSHOT_K_VALUES:
            vals = []
            for src in ["L", "R", "D", "S"]:
                for tgt in tgt_domains:
                    r = load_result(model, f"fewshot_k{k}", src, tgt)
                    if r:
                        vals.append(r["macro_f1"])
            if vals:
                ks.append(k)
                f1s.append(np.mean(vals))
        if ks:
            label = MODEL_DISPLAY_NAMES.get(model, model)
            ax.plot(ks, f1s, marker=markers[i % len(markers)],
                    color=colors[i], label=label, linewidth=2, markersize=7)

    ax.set_xlabel("k (shots per class)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Few-Shot Sensitivity Analysis")
    ax.set_xticks(FEWSHOT_K_VALUES)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

def fig_error_distribution(models, pairs, setting, save_path=None):
    """Figure 5: Error distribution bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    tgt_domains = sorted(set(t for _, t in pairs))
    x = np.arange(len(tgt_domains))
    w = 0.8 / max(len(models), 1)
    colors = sns.color_palette("Set2", len(models))

    for i, model in enumerate(models):
        errs = []
        for tgt in tgt_domains:
            vals = []
            for src, t in pairs:
                if t == tgt:
                    r = load_result(model, setting, src, t)
                    if r:
                        vals.append(100 - r["macro_f1"])
            errs.append(np.mean(vals) if vals else 0)
        label = MODEL_DISPLAY_NAMES.get(model, model)
        ax.bar(x + i * w - 0.4 + w/2, errs, w, label=label, color=colors[i % len(colors)])

    ax.set_xlabel("Target Domain")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Distribution Across Target Domains")
    ax.set_xticks(x)
    ax.set_xticklabels(tgt_domains)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

def fig_model_gain(baseline, proposed, pairs, setting, save_path=None):
    """Figure 6: Model gain bar chart (LLM-KGKAN improvement over baseline)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pair_strs = [f"{s}→{t}" for s, t in pairs]
    gains = []
    for src, tgt in pairs:
        rb = load_result(baseline, setting, src, tgt)
        rp = load_result(proposed, setting, src, tgt)
        if rb and rp:
            gains.append(rp["macro_f1"] - rb["macro_f1"])
        else:
            gains.append(0)

    colors = ["#2ecc71" if g >= 0 else "#e74c3c" for g in gains]
    ax.bar(range(len(gains)), gains, color=colors, edgecolor="white")
    ax.set_xticks(range(len(pair_strs)))
    ax.set_xticklabels(pair_strs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Gain (%)")
    bl = MODEL_DISPLAY_NAMES.get(baseline, baseline)
    pr = MODEL_DISPLAY_NAMES.get(proposed, proposed)
    ax.set_title(f"{pr} Gain over {bl}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

def fig_transfer_heatmap(model, pairs, setting, save_path=None):
    """Figure 7: Pairwise transfer heatmap."""
    srcs = sorted(set(s for s, _ in pairs))
    tgts = sorted(set(t for _, t in pairs))
    matrix = np.full((len(srcs), len(tgts)), np.nan)

    for i, src in enumerate(srcs):
        for j, tgt in enumerate(tgts):
            r = load_result(model, setting, src, tgt)
            if r:
                matrix[i, j] = r["macro_f1"]

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.isnan(matrix)
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=tgts, yticklabels=srcs, mask=mask,
                ax=ax, vmin=20, vmax=80, linewidths=0.5)
    ax.set_xlabel("Target Domain")
    ax.set_ylabel("Source Domain")
    label = MODEL_DISPLAY_NAMES.get(model, model)
    ax.set_title(f"{label} — Pairwise Transfer Macro-F1 (%)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
