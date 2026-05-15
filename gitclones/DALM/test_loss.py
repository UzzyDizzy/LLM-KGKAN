import sys
import os
import torch
import logging

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(r"c:\Users\KIIT0001\Desktop\gitclones\LLM-KGKAN\gitclones\DALM\absa")

import data_utils
from models import BERT_CRF
from transformers import BertTokenizer

label_list = data_utils.get_labels("absa")
model = BERT_CRF.from_pretrained("bert-base-uncased", num_labels=len(label_list))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
source_data_path = r'c:\Users\KIIT0001\Desktop\gitclones\LLM-KGKAN\gitclones\DALM\raw_data\laptop_train.txt'
source_dataset = data_utils.OurDataset(data_path=source_data_path, task="absa", tokenizer=tokenizer, max_len=128)

from torch.utils.data import DataLoader
source_dataloader = DataLoader(source_dataset, batch_size=16, drop_last=True, shuffle=False)

model.eval()

source_iter = iter(source_dataloader)
batch = next(source_iter)

source_input_ids, source_input_masks, source_label_ids = batch["input_id"], batch["input_mask"], batch["label_id"]

source_output, absa_loss = model(source_input_ids, attention_mask=source_input_masks, labels=source_label_ids)

print("ABSA LOSS:", absa_loss.item())

# print out CRF details
emissions = model.classifier(model.dropout(model.bert(source_input_ids, attention_mask=source_input_masks)[0]))
tags = source_label_ids
mask = source_input_masks.bool()

if model.crf.batch_first:
    emissions = emissions.transpose(0, 1)
    tags = tags.transpose(0, 1)
    mask = mask.transpose(0, 1)

numerator = model.crf._compute_score(emissions, tags, mask)
denominator = model.crf._compute_normalizer(emissions, mask)
print("Numerator:", numerator)
print("Denominator:", denominator)
print("Numerator - Denominator:", numerator - denominator)
