#config.py
from dataclasses import dataclass


@dataclass
class LLMKGKANConfig:
    llm_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    hidden_size: int = 4096   # IMPORTANT (llama hidden dim)
    batch_size: int = 16       # GPU constraint
    epochs: int = 10          # paper uses 10
    freeze_backbone: bool = True
    num_labels: int = 7
    num_dep_relations: int = 40
    num_entities: int = 10000
    num_kg_relations: int = 500
    kg_emb_dim: int = 128
    rgcn_layers: int = 2
    dropout: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    mmd_lambda: float = 0.2
    ignore_index: int = -100
    freeze_backbone: bool = True
    use_4bit: bool = False
    use_distmult: bool = False
    lr: float = 2e-4
    weight_decay: float = 1e-2
    max_len: int = 128
    seed: int = 42
    prefix_len: int = 10
    LABEL2ID = {
    	"O": 0,
    	"B-POS": 1, "I-POS": 2,
    	"B-NEG": 3, "I-NEG": 4,
    	"B-NEU": 5, "I-NEU": 6,
	}
