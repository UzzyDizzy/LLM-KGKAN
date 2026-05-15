import sys
sys.path.append(r"c:\Users\KIIT0001\Desktop\gitclones\LLM-KGKAN\gitclones\DALM\absa")

import torch
import data_utils
from models import BERT_CRF

# We need to test the exact condition in models.py
label_list = data_utils.get_labels("absa")
model = BERT_CRF.from_pretrained("bert-base-uncased", num_labels=len(label_list))

# Mock input that mimics the data_utils output
source_input_ids = torch.randint(0, 1000, (16, 100)).long()
source_input_masks = torch.ones(16, 100).long()
source_label_ids = torch.randint(0, 7, (16, 100)).long()

# Set padding
source_input_masks[:, 20:] = 0
source_label_ids[:, 20:] = -1

# Forward pass
try:
    source_output, absa_loss = model(source_input_ids, attention_mask=source_input_masks, labels=source_label_ids)
    print("MOCK LOSS:", absa_loss.item())
except Exception as e:
    print("ERROR:", e)

# Now test BertAdam
from optimization import BertAdam
param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=0.1, t_total=100)

absa_loss.backward()
optimizer.step()
print("UPDATED!")
