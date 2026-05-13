# LLM-KGKAN Paper Reproduction — Walkthrough

## Summary

Built a complete reproduction framework for the LLM-KGKAN paper with **13 new files**, **3 modified files**, and a **57-cell `experiments.ipynb`** notebook that reproduces all 16 tables and 5 figures.

## Files Created

### Core Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| [config.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/config.py) | Centralized config: GPU profiles (RTX P6000 active, H200 commented), PEFT/QLoRA settings, all transfer pairs, API budget ($20/$50/$100 tiers), model configs for all 10+ models |
| [data_utils.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/data_utils.py) | Unified data loading: CSV→BIO tag conversion, multi-aspect merging, few-shot sampling, collate functions for BERT/LLM/simple pipelines |
| [evaluate.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/evaluate.py) | BIO-aware Macro-F1 computation, JSON result save/load, DataFrame table builder with graceful `None` for missing results |
| [train_all.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_all.py) | Training orchestrator for all 10 models with checkpoint resumption, early stopping, AMP, error isolation per model |
| [llm_inference.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/llm_inference.py) | API inference with $20 budget cap, per-request caching, rate limiting, skip when API key missing |
| [visualize.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/visualize.py) | 4 plotting functions for paper figures (sensitivity, error dist, gain, heatmap) |
| [generate_notebook.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/generate_notebook.py) | Generates the 57-cell experiments.ipynb programmatically |
| [__init__.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/__init__.py) | Package init |

### Root Files

| File | Purpose |
|------|---------|
| [experiments.ipynb](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/experiments.ipynb) | Main execution notebook (57 cells) — the single entry point for the entire reproduction |
| [requirements2.txt](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/requirements2.txt) | All dependencies for all models |

### Directories Created

```
saved_models/    — checkpoints for each model
results/
  ├── api_cache/       — cached API responses
  ├── standard/        — standard benchmark results
  ├── fewshot/         — few-shot results
  ├── zeroshot/        — zero-shot results
  ├── ablation/        — ablation study results
  └── training_logs/   — per-epoch training logs
```

## Files Modified

| File | Change |
|------|--------|
| [config.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/config.py) | Changed `llm_name` from `mistralai/Mistral-7B-Instruct-v0.2` → `meta-llama/Meta-Llama-3-8B-Instruct` (paper Table 4) |
| [domains.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/domains.py) | Fixed all 9 domain paths (`laptops.csv`→`laptop.csv`, `restaurants.csv`→`restaurant.csv`, etc.), expanded to all source/target domains |
| [gitclones/AHF/config.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/gitclones/AHF/config.py) | Fixed `pin_memory=cfg.pin_memory` → `pin_memory=True` (NameError) |

## Data Validation

All 9 domains verified:

| Domain | Key | Rows |
|--------|-----|------|
| Laptop | L | 2,843 |
| Restaurant | R | 5,929 |
| Device | D | 2,107 |
| Service | S | 2,239 |
| Airline | A | 977 |
| Shoes | SH | 730 |
| Water Purifier | W | 604 |
| University Course | U | 320 |
| Healthcare | H | 340 |
| **Total** | | **16,089** |

## Notebook Structure (57 cells)

### Phase 0 (Cells 1-6): Setup
- Install deps, imports, KG loading, data validation

### Phase 1 (Cells 7-14): Training  
- BERT baselines (BERT-UDA, AHF, TransProto, BGCA, KETGM, DALM)
- Adapted models (KGAN, SenticGCN)
- LLMSynABSA
- LLM-KGKAN
- Few-shot training (all models × 12 pairs)
- Zero-shot evaluation (all models × 8 pairs)

### Phase 2 (Cells 15-18): Evaluation
- LLM API inference (GPT-4o, GPT-4 Turbo, etc. with budget control)
- Few-shot sensitivity (k=2,4,8,16)
- Ablation studies (w/o KG, Syn, ARG, KAN)

### Phase 3 (Cells 19-57): Results
- Tables 1-16 auto-generated from results JSONs
- Figures 3-7 saved as PNG

## Key Design Decisions

1. **Error isolation**: Each model training is wrapped in try/except — one model failing doesn't block others
2. **Checkpoint resumption**: All training checks for existing checkpoints before starting
3. **API budget control**: $20 active limit with $50/$100 tiers commented in config
4. **Missing results = blank**: Tables show `None`/`—` for models without results, fills in when you rerun
5. **KGAN/SenticGCN adaptation**: Wrapped as BERT+extra-layers sequence labelers to produce BIO tags compatible with the paper's Macro-F1 metric
6. **KnowLA excluded**: Per user request (repo to be deleted)

## How to Run

1. Open `experiments.ipynb` on GPU machine
2. Run cells sequentially
3. Phase 1 trains all models (hours)
4. Phase 2 runs API inference (budget-controlled)
5. Phase 3 auto-generates all tables/figures
6. Re-run individual Phase 3 cells anytime to refresh tables with latest results
