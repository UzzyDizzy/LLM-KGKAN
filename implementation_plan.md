# LLM-KGKAN: Full Paper Reproduction Plan

Reproduce all **16 tables** and **5 figures** from the LLM-KGKAN paper in `experiments.ipynb`.

## User Review Required

> [!IMPORTANT]
> **New models requested (KGAN, KinGDOM, SenticGCN, KnowLA) are NOT in the paper's tables.** The paper's tables only include: BERT-UDA, AHF, TransProto, BGCA, KETGM, DALM, LLM-Augment-Syntax-DA, GPT-4o, GPT-4 Turbo, LLaMA-3 8B, Qwen2-72B, and LLM-KGKAN. Since you asked to add these 4 models to Tables 1, 3, 6, 7, 8 — I will train them and add extra rows in those tables. Please confirm this is the intent.

> [!WARNING]
> **Paper values are hardcoded targets.** The paper reports specific numbers (e.g., Table 1 LLM-KGKAN AVG = 58.13). Since we're running with a single seed and PEFT/QLoRA, actual reproduced numbers will likely differ slightly from published results (which were averaged over multiple seeds). The tables will show **our reproduced values**, not the paper's exact numbers. If you want to show paper values alongside reproduced values, let me know.

> [!IMPORTANT]
> **API cost considerations.** Tables 1, 3, 6, 7 require GPT-4o, GPT-4 Turbo inference across ~60+ transfer pairs. With batching, this could cost $50-200+ in API calls. I'll implement batching with configurable `max_concurrent` and `requests_per_minute` limits. Confirm you want to proceed with all API models.

> [!WARNING]
> **LLaMA-3.1-8B-Instruct vs LLaMA-3-8B-Instruct.** The paper Table 1 has both `LLaMA-3.1-8B-Instruct` and separate `LLaMA-3 8B` rows. Table 4 says the backbone is `LLaMA-3-8B-Instruct`. The config currently uses `mistralai/Mistral-7B-Instruct-v0.2`. I'll switch to `meta-llama/Meta-Llama-3-8B-Instruct` as the paper specifies. Confirm this is correct.

> [!IMPORTANT]
> **Table 3 includes GPT-4.1, GPT-5.4, Claude Sonnet, Gemini 2.5 Pro** — these require additional API keys/endpoints not in `.env.local`. Should I skip these or add placeholder rows? I'll need API keys for Claude and Gemini if you want real inference.

## Open Questions

1. **`books.csv` and `clothing.csv`** exist in `data/` but aren't in the paper tables. Should these be used for anything or are they extras?
2. **domains.py paths mismatch**: `domains.py` references `laptops.csv` and `restaurants.csv` but actual files are `laptop.csv` and `restaurant.csv`. I'll fix this.
3. **KinGDOM** operates on document-level sentiment (books/dvd/electronics/kitchen domains), not aspect-level ABSA. Adapting it to BIO token-level ABSA on our 11 datasets requires substantial architectural changes. Should I create a wrapper that does aspect-level classification, or skip KinGDOM if it doesn't fit the BIO formulation?
4. **KGAN** is also designed for aspect-level sentiment classification (3-class: pos/neg/neu per aspect), not BIO sequence labeling. Same question — should I adapt it or evaluate its 3-class aspect sentiment classification and convert to comparable Macro-F1?
5. **SenticGCN** same situation — it does 3-class aspect sentiment classification. 
6. **KnowLA** is for general QA/commonsense tasks, not ABSA. Adapting it to cross-domain ABSA would require significant restructuring.

> [!CAUTION]
> Models KGAN, KinGDOM, SenticGCN, and KnowLA are designed for **different tasks/formulations** than what the paper evaluates (BIO sequence labeling for cross-domain ABSA). To make them comparable, I need to either: (a) adapt their inference to produce BIO tags compatible with the Macro-F1 metric, or (b) wrap them with a conversion layer. This adaptation is non-trivial and may produce suboptimal results since the models weren't designed for this exact task. **Please confirm how you'd like to handle these 4 models.**

---

## Proposed Changes

### Phase 0: Setup & Infrastructure

#### [NEW] [scripts/config.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/config.py)
Centralized configuration for ALL models and experiments:
- GPU profiles: `RTX_P6000` (default) and `H200` (commented out)
- Unified PEFT config: LoRA r=8, alpha=16, dropout=0.05, bfloat16
- All model configs (batch sizes, learning rates, epochs)
- Transfer pair definitions for all 3 settings (standard, few-shot, zero-shot)
- Seed = 42 (single seed throughout)
- Checkpoint/save directory structure
- API batching config (batch_size, rate_limit, retries)

#### [MODIFY] [requirements2.txt](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/requirements2.txt)
All dependencies for all models:
```
torch>=2.1.0, transformers>=4.45.0, accelerate>=0.30.0, peft>=0.12.0,
bitsandbytes>=0.43.0, pandas, numpy, scikit-learn, spacy>=3.7,
tqdm, sentencepiece, protobuf, matplotlib, seaborn, scipy,
openai>=1.0, tiktoken, python-dotenv, nltk, dgl, torch-geometric,
gensim, lda, fire, datasets
```

#### [MODIFY] [domains.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/domains.py)
Fix file paths (`laptops.csv` → `laptop.csv`, `restaurants.csv` → `restaurant.csv`, etc.) and expand to all 9 domains (L, R, D, S, A, SH, W, U, H).

---

### Phase 1: Knowledge Graph Setup + Data Loading + Default Model Training

#### Cell 1: Install dependencies
```python
!pip install -r requirements2.txt
!python -m spacy download en_core_web_sm
```

#### Cell 2: Environment setup & KG preparation
- Load `.env.local` for API keys
- Extract ConceptNet (already compressed), filter English-only triples
- Load SenticNet from `knowledge_graphs/senticnet/senticnet.txt`
- Load WordNet from `knowledge_graphs/wn3.1.dict/`
- Build hybrid KG (ConceptNet + SenticNet + WordNet) as described in paper
- Save processed KG pickle for fast reloading

#### Cell 3: Unified data loading validation
- Load all 11 CSVs, verify schema: `id, text, aspect, category, polarity, from, to`
- Print Table 2 and Table 5 statistics (total/train/test splits)
- Validate train/test split ratios match paper

#### Cells 4-14: Train/finetune each model (1 cell per model)

Each cell follows the pattern:
```python
# Cell N: Train MODEL_NAME
# - Check for existing checkpoint → resume if found
# - Load data in model-specific format
# - Train with PEFT/QLoRA where applicable
# - Save checkpoint + training logs (loss, val_f1 per epoch)
# - Save final model to saved_models/MODEL_NAME/
```

**Cell 4: BERT-UDA** (BERT-base-uncased + domain adaptation)
- Load `bert-base-uncased` from HF
- Apply QLoRA (4-bit quantization + LoRA r=8)
- Train on source domain with UDA objective
- Save: `saved_models/bert_uda/{src}_to_{tgt}.pt`

**Cell 5: AHF** (Adaptive Hybrid Framework)
- Uses BiLSTM + adversarial training + EMA teacher
- Import from `gitclones/AHF/`, use its model.py/train.py
- Adapt data loading to use our unified CSV format
- Save: `saved_models/ahf/{src}_to_{tgt}.pt`

**Cell 6: TransProto** (Retrieve-and-Edit)
- Import from `gitclones/TransProto/`
- Needs BERT cross-domain setup, prototype retrieval
- Adapt data paths to our CSV format
- Save: `saved_models/transproto/{src}_to_{tgt}.pt`

**Cell 7: BGCA** (Bidirectional Generative Cross-domain)
- Import from `gitclones/BGCA/code/`
- Generative framework for cross-domain ABSA
- Save: `saved_models/bgca/{src}_to_{tgt}.pt`

**Cell 8: KETGM** (Knowledge-Enhanced Topic-Guided)
- Import from `gitclones/KETGM/`
- Pre-train BERT syntactic encoder, R-GCN autoencoder
- Train end-to-end KETGM with topic-guided attention
- Save: `saved_models/ketgm/{src}_to_{tgt}.pt`

**Cell 9: DALM** (Domain-Adaptive Language Modeling)
- Import from `gitclones/DALM/absa/`
- GPT2-based with pseudo-labeling
- Save: `saved_models/dalm/{src}_to_{tgt}.pt`

**Cell 10: LLMSynABSA** (LLM-Augment-Syntax-DA)
- Import from `gitclones/LLMSynABSA/`
- LLaMA-3-8B with syntax-aware transformer
- Apply QLoRA, bfloat16
- Save: `saved_models/llmsynabsa/{src}_to_{tgt}.pt`

**Cell 11: KGAN** (Knowledge Graph Attention Network)
- Import from `gitclones/KGAN/`
- Adapt for cross-domain ABSA evaluation
- Save: `saved_models/kgan/{src}_to_{tgt}.pt`

**Cell 12: SenticGCN** (Sentic Graph Convolutional Network)
- Import from `gitclones/Sentic-GCN/`
- Uses senticnet for graph construction
- Save: `saved_models/senticgcn/{src}_to_{tgt}.pt`

**Cell 13: LLM-KGKAN** (Our model)
- Load from root project files (model.py, data.py, etc.)
- Fix config: `llm_name` → `meta-llama/Meta-Llama-3-8B-Instruct`
- Train with QLoRA + bfloat16 on all transfer pairs
- Save: `saved_models/llm_kgkan/{src}_to_{tgt}.pt`
- Also train ablation variants: w/o KG, w/o Syn, w/o ARG, w/o KAN

**Cell 14: LLM-KGKAN Ablation variants**
- w/o KG: Zero out relational stream
- w/o Syn: Skip syntactic encoder, use identity
- w/o ARG: Skip ARG module, use KAN output directly
- w/o KAN: Replace KAN fusion with simple concatenation + MLP
- Save each variant separately

---

### Phase 2: Evaluation & LLM API Inference

#### Cell 15: LLM API inference setup
- OpenAI client with batching (batch_size=20, rate_limit=500 RPM)
- Prompt template for BIO sequence labeling
- Models: GPT-4o, GPT-4 Turbo, GPT-4.1, GPT-5.4
- Cache all API responses to `results/api_cache/`

#### Cell 16: LLM API inference — Standard benchmark
Run GPT-4o, GPT-4 Turbo on all 10 standard transfer pairs (Table 1)

#### Cell 17: LLM API inference — Few-shot & Zero-shot
Run API models on low-resource (A, SH, W) and no-label (U, H) targets

#### Cell 18: Open-source LLM inference
- LLaMA-3.1-8B-Instruct (via HF, QLoRA 4-bit)
- Qwen2.5-7B/14B/72B-Instruct (inference only for 72B)
- Prompt-based BIO tagging, no training

#### Cells 19-28: Evaluate all trained models on all settings
Each cell evaluates one model across all transfer pairs:
- Standard: 10 pairs (L,R,D,S cross-domain, excluding L↔D)
- Few-shot: 12 pairs ({L,R,D,S} → {A,SH,W}) with k=16-shot
- Zero-shot: 8 pairs ({L,R,D,S} → {U,H})
- Save all results to `results/` as JSON with structure:
```json
{
  "model": "MODEL_NAME",
  "setting": "standard|fewshot|zeroshot",
  "src": "L", "tgt": "R",
  "macro_f1": 65.78,
  "per_epoch_val_f1": [...]
}
```

#### Cell 29: Few-shot sensitivity evaluation
Evaluate LLM-KGKAN and baselines under k ∈ {2, 4, 8, 16} shots

#### Cell 30: Fusion ablation (Table 12)
Train LLM-KGKAN with different fusion methods:
- Concat + MLP, Weighted Sum, Gated Fusion, Bilinear, Cross-Attention, KAN

#### Cell 31: KG source ablation (Table 13)
Train LLM-KGKAN with different KG sources:
- No KG, WordNet only, SenticNet only, ConceptNet only, Hybrid

#### Cell 32: Statistical significance (Table 8)
Run paired t-tests comparing LLM-KGKAN vs each baseline

---

### Phase 3: Reproduce Tables & Figures

#### [NEW] [scripts/train_bert_uda.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_bert_uda.py)
Wrapper for BERT-UDA training adapted to our data format.

#### [NEW] [scripts/train_ahf.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_ahf.py)
Wrapper for AHF training adapted to our data format.

#### [NEW] [scripts/train_transproto.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_transproto.py)
Wrapper for TransProto training.

#### [NEW] [scripts/train_bgca.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_bgca.py)
Wrapper for BGCA training.

#### [NEW] [scripts/train_ketgm.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_ketgm.py)
Wrapper for KETGM training.

#### [NEW] [scripts/train_dalm.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_dalm.py)
Wrapper for DALM training.

#### [NEW] [scripts/train_llmsynabsa.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_llmsynabsa.py)
Wrapper for LLMSynABSA training.

#### [NEW] [scripts/train_kgan.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_kgan.py)
Wrapper for KGAN training.

#### [NEW] [scripts/train_senticgcn.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_senticgcn.py)
Wrapper for SenticGCN training.

#### [NEW] [scripts/train_llmkgkan.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/train_llmkgkan.py)
Wrapper for LLM-KGKAN training (full + ablations).

#### [NEW] [scripts/evaluate.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/evaluate.py)
Unified evaluation: loads any model checkpoint, runs on any transfer pair, computes Macro-F1.

#### [NEW] [scripts/llm_inference.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/llm_inference.py)
API-based LLM inference with batching, caching, and retry logic.

#### [NEW] [scripts/visualize.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLM-KGKAN/scripts/visualize.py)
All plotting functions for the 5 figures.

#### Notebook Cells for Tables (Phase 3):

**Cell 33: Table 1** — Standard cross-domain ABSA benchmark (10 transfer pairs)
**Cell 34: Table 2** — Dataset statistics (computed from CSVs)
**Cell 35: Table 3** — Low-resource and no-label target domains
**Cell 36: Table 4** — Hyperparameter table (static, from config)
**Cell 37: Table 5** — Additional dataset statistics
**Cell 38: Table 6** — Detailed few-shot pairwise results (12 pairs × models)
**Cell 39: Table 7** — Detailed zero-shot pairwise results (8 pairs × models)
**Cell 40: Table 8** — Paired t-test p-values
**Cell 41: Table 9** — Few-shot sensitivity (k ∈ {2,4,8,16})
**Cell 42: Table 10** — Target-wise few-shot results for LLM-KGKAN
**Cell 43: Table 11** — Efficiency comparison (params, time, latency)
**Cell 44: Table 12** — Fusion strategy ablation
**Cell 45: Table 13** — KG source ablation
**Cell 46: Table 14** — Qualitative error analysis (static table from paper)
**Cell 47: Table 15** — Remaining failure cases (static table from paper)
**Cell 48: Table 16** — KG relation type distribution

#### Notebook Cells for Figures (Phase 3):

**Cell 49: Figure 3/4** — Few-shot sensitivity line plot
**Cell 50: Figure 5** — Error distribution bar chart across target domains
**Cell 51: Figure 6** — Model gain bar chart across target domains
**Cell 52: Figure 7** — Pairwise transfer heatmap

---

## Directory Structure

```
LLM-KGKAN/
├── scripts/
│   ├── config.py              # Centralized config
│   ├── train_bert_uda.py
│   ├── train_ahf.py
│   ├── train_transproto.py
│   ├── train_bgca.py
│   ├── train_ketgm.py
│   ├── train_dalm.py
│   ├── train_llmsynabsa.py
│   ├── train_kgan.py
│   ├── train_senticgcn.py
│   ├── train_llmkgkan.py
│   ├── evaluate.py
│   ├── llm_inference.py
│   └── visualize.py
├── saved_models/
│   ├── bert_uda/
│   ├── ahf/
│   ├── transproto/
│   ├── bgca/
│   ├── ketgm/
│   ├── dalm/
│   ├── llmsynabsa/
│   ├── kgan/
│   ├── senticgcn/
│   ├── llm_kgkan/
│   │   ├── full/
│   │   ├── wo_kg/
│   │   ├── wo_syn/
│   │   ├── wo_arg/
│   │   └── wo_kan/
│   └── fusion_ablation/
├── results/
│   ├── api_cache/
│   ├── standard/
│   ├── fewshot/
│   ├── zeroshot/
│   ├── ablation/
│   └── training_logs/
├── knowledge_graphs/
│   ├── conceptnet_en.pkl       # Processed English-only ConceptNet
│   ├── senticnet_processed.pkl # Processed SenticNet
│   ├── wordnet_processed.pkl   # Processed WordNet
│   └── hybrid_kg.pkl           # Combined KG
├── experiments.ipynb            # Main notebook (~52 cells)
└── requirements2.txt
```

## Verification Plan

### Automated Tests
1. **Data integrity**: Verify all 11 CSVs load correctly, row counts match Table 2 & 5
2. **Model loading**: Verify each saved checkpoint loads without error
3. **Metric sanity**: Verify Macro-F1 computation on known inputs
4. **KG coverage**: Verify ConceptNet/SenticNet/WordNet load with expected entity counts
5. **Table auto-rectification**: Re-running Phase 3 cells auto-loads latest results

### Manual Verification
1. Compare reproduced Table 1 numbers against paper (expect ±1-3% due to single seed)
2. Visual inspection of all 5 figures against paper figures
3. Verify training logs show convergence (no NaN losses)
4. Check GPU memory usage stays within 96GB VRAM budget
