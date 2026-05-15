# train.py — Training loop matching paper (Eq 26–27, Section 4.3)
import torch
import torch.nn as nn
from torch import tensor
#from torch.cuda.amp import GradScaler
from config import (
    LR, LR_SYNTAX, ACCUM_STEPS, RHO, EPOCHS, EPOCHS_SYNTAX,
    NUM_LABELS, device,
)
from config import LABEL_MAP
from torch.optim.lr_scheduler import LambdaLR



def build_optimizers(model):
    """
    Paper Section 4.3:
    - Adam optimizer for main params (lr=5e-5)
    - SGD for syntax-aware transformer params (lr=5e-4)
    """
    syntax_names = ["Wq", "Wk", "Wv", "Wo", "theta", "phi",
                    "rel_emb", "norm1", "norm2", "ffn"]
    
    syntax_param_ids = set()
    for name, param in model.named_parameters():
        if param.requires_grad and any(sn in name for sn in syntax_names):
            syntax_param_ids.add(id(param))

    main_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in syntax_param_ids]
    syn_params  = [p for p in model.parameters()
                   if p.requires_grad and id(p) in syntax_param_ids]

    opt_main   = torch.optim.Adam(main_params, lr=LR)
    opt_syntax = torch.optim.AdamW(syn_params, lr=LR_SYNTAX, weight_decay=0.01)
    return opt_main, opt_syntax


def train_one_epoch(model, source_loader, target_loader, optimizers, scheduler, syntax_cache, epoch, total_epochs):
    """
    Train one epoch with source (labeled) + target (unlabeled) data.
    Loss = L_seq + ρ * L_adv (Eq 27)
    """
    model.train()
    opt_main, opt_syntax = optimizers

    # cross-entropy, class weights etc.
    class_weights = torch.tensor([0.35, 0.90, 1.05, 1.00, 1.25, 1.20, 1.50],device=device,dtype=torch.float32,)
    ce_seq = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    #ce_seq = nn.CrossEntropyLoss(ignore_index=-100)
    ce_domain = nn.CrossEntropyLoss()

    total_loss = total_seq_loss = total_adv_loss = 0.0
    num_batches = 0
    target_iter = iter(target_loader)
    total_steps = total_epochs * len(source_loader)

    for step, src_batch in enumerate(source_loader):
        current_step = (epoch - 1) * len(source_loader) + step
        ids = src_batch["input_ids"].to(device)
        mask = src_batch["attention_mask"].to(device)
        labels = src_batch["labels"].to(device)
        asp_labels = src_batch["aspect_labels"].to(device)
        sent_labels = src_batch["sentiment_labels"].to(device)
        texts = src_batch["text"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, asp_logits, sent_logits, domain_logits = model(
                ids, mask, texts, syntax_cache,
                aspect_labels=asp_labels,
                sentiment_labels=sent_labels,
                current_step=current_step,
                total_steps=total_steps,
            )
            loss_seq = ce_seq(out.view(-1, NUM_LABELS), labels.view(-1))
            loss_adv_src = ce_domain(domain_logits.float(), torch.zeros(ids.size(0), device=device, dtype=torch.long))

        # target domain forward (adversarial)
        try:
            tgt_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_batch = next(target_iter)

        tgt_ids = tgt_batch["input_ids"].to(device)
        tgt_mask = tgt_batch["attention_mask"].to(device)
        tgt_texts = tgt_batch["text"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, _, tgt_domain_logits = model(
                tgt_ids, tgt_mask, tgt_texts, syntax_cache,
                current_step=current_step,
                total_steps=total_steps,
            )
            loss_adv_tgt = ce_domain(tgt_domain_logits.float(), torch.ones(tgt_ids.size(0), device=device, dtype=torch.long))

        # combine
        loss_adv = (loss_adv_src + loss_adv_tgt) / 2
        loss_adv = torch.clamp(loss_adv, max=5.0)
        rho = min(RHO, RHO * epoch / 30)

        # stop adversarial training after stabilization
        if epoch > 20:
            rho = 0.0

        loss = (loss_seq + rho * loss_adv) / ACCUM_STEPS
        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            opt_main.step()
            opt_syntax.step()
            if scheduler is not None:
                scheduler.step()
            opt_main.zero_grad(set_to_none=True)
            opt_syntax.zero_grad(set_to_none=True)

        total_loss += loss.item() * ACCUM_STEPS
        total_seq_loss += loss_seq.item()
        total_adv_loss += loss_adv.item()
        num_batches += 1

    return total_loss / num_batches, total_seq_loss / num_batches, total_adv_loss / num_batches