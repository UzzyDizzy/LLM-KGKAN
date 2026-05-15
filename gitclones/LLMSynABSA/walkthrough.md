# LLMSynABSA — Paper-Faithful Implementation Walkthrough

## Files Changed

| File | Status | Key Changes |
|------|--------|-------------|
| [config.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLMSynABSA/config.py) | Rewritten | Fixed hyperparameters, added dropout/freeze flags |
| [data.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLMSynABSA/data.py) | Rewritten | Aspect grouping per sentence, train/test split, unified labels |
| [utils.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLMSynABSA/utils.py) | Rewritten | Word-to-token alignment for syntax matrix |
| [model.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLMSynABSA/model.py) | Rewritten | 10+ architectural fixes matching paper equations |
| [train.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLMSynABSA/train.py) | Rewritten | Unified loss L_seq, Adam optimizer, adversarial training |
| [evaluate.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLMSynABSA/evaluate.py) | Rewritten | Batched evaluation, per-class Macro-F1 |
| [main.py](file:///c:/Users/KIIT0001/Desktop/gitclones/LLMSynABSA/main.py) | **NEW** | Proper CLI training script (replaces notebook) |

---

## Critical Fixes Applied

### 1. Unified Loss Function (Eq 26-27)
**Before**: `loss = loss_asp + loss_sent + RHO * loss_adv` (separate aspect + sentiment losses)  
**After**: `loss = loss_seq + RHO * loss_adv` where `loss_seq` is a single 7-class CE over `{O, B-POS, I-POS, B-NEG, I-NEG, B-NEU, I-NEU}`

### 2. Feature-Conditioned Soft Prompts (Eq 22)
**Before**: Static `nn.Parameter(torch.randn(1, 10, d))` — same prompt for every input  
**After**: `P = f(H_mean, A_vec, S_vec, G)` — prompts generated from the current input's analyzed features

### 3. Syntax Transformer LayerNorm (Eq 12)
**Before**: Missing `norm1` — only had `norm2(H_att + FFN(H_att))`  
**After**: `H_att = norm1(H + attention_output)` then `F = norm2(H_att + FFN(H_att))`

### 4. GRL Placement in Domain Predictor (Section 3.4)
**Before**: GRL applied after attention+projection (on output G)  
**After**: GRL applied before attention layer (on input HC), matching paper

### 5. Data Grouping (Multi-Aspect Sentences)
**Before**: Each row = 1 sample → sentences with 3 aspects had 3 partial-label samples  
**After**: Group by text → each sentence has ALL its aspects labeled simultaneously

### 6. Syntax Matrix Word-to-Token Alignment
**Before**: Assumed spacy word position == LLaMA token position  
**After**: Explicit mapping from spacy word indices to LLaMA subword token indices

### 7. Other Fixes
- **FFN intermediate**: `d*2` (8192) → 11008 (paper spec)
- **Dropout**: Added 0.2 everywhere (was missing entirely)
- **Temperature hack**: Removed `/ 0.7` on output
- **Optimizer**: AdamW → Adam (paper spec)
- **Class weights**: Removed (not in paper)
- **Per-head θ, φ**: Scalar per head instead of single global scalar
- **M addition**: After score computation, not dividing by √dk
- **Batch size**: Effective 32 via accumulation

---

## How to Run (on A100)

```bash
# Install dependencies
pip install torch transformers accelerate spacy pandas scikit-learn numpy
python -m spacy download en_core_web_sm

# Run R→L adaptation
python main.py --source restaurant --target laptop

# Custom settings
python main.py --source restaurant --target laptop --split_ratio 0.8 --seed 42
```

### Expected Output
```
Loading tokenizer: meta-llama/Meta-Llama-3-8B-Instruct
Loading source domain: data/restaurant.csv
  Total: 3497 | Train: 2798 | Test: 699
Loading target domain: data/laptop.csv
  Total: 1859 | Train(unlabeled): 1487 | Test: 372
Building syntax cache for ~4000 texts...
Initializing model...
  Total params:     8,xxx,xxx,xxx
  Trainable params: ~300,000,000
  Frozen params:    ~7,700,000,000
============================================================
Training: restaurant → laptop (100 epochs)
============================================================
Epoch   1/100 | Loss: x.xxxx | Macro-F1: x.xxxx | ...
```

### Model Checkpoints
Saved to `checkpoints/`:
- `restaurant_to_laptop_best.pt` — best Macro-F1 checkpoint
- `restaurant_to_laptop_final.pt` — final epoch checkpoint

### Target Performance
- Paper R→L Macro-F1: **0.511**
- Expected range: **0.49–0.53** (±0.02 tolerance)

---

## Memory Estimate (A100 48GB)

| Component | Memory |
|-----------|--------|
| LLaMA-3-8B frozen (bf16) | ~16 GB |
| Added components (~300M params) | ~1 GB |
| Gradients + optimizer states | ~4 GB |
| Activations (batch 16, seq 96) | ~8 GB |
| Syntax cache | ~1 GB |
| **Total** | **~30 GB** ✅ |
