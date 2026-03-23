import logging
import warnings
import struct
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "sarvamai/sarvam-1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WEIGHTS_BIN_PATH = ARTIFACTS_DIR / "sarvam1.bin"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def write(f, tensor):
    f.write(tensor.detach().float().numpy().astype(np.float32).tobytes())


print("downloading and loading model (this will take a while on first run)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)
model = model.float()
model.eval()

m = model.model
L = len(m.layers)

dim = m.embed_tokens.weight.shape[1]
hidden_dim = m.layers[0].mlp.gate_proj.weight.shape[0]
num_heads = model.config.num_attention_heads
n_kv_heads = model.config.num_key_value_heads
vocab_size = model.config.vocab_size
seq_len = model.config.max_position_embeddings

print(
    f"dim={dim} hidden={hidden_dim} layers={L} heads={num_heads} kv={n_kv_heads} vocab={vocab_size} seq={seq_len}"
)

with open(WEIGHTS_BIN_PATH, "wb") as f:
    f.write(
        struct.pack(
            "iiiiiii", dim, hidden_dim, L, num_heads, n_kv_heads, vocab_size, seq_len
        )
    )
    write(f, m.embed_tokens.weight)
    for i in range(L):
        write(f, m.layers[i].input_layernorm.weight)
    for i in range(L):
        write(f, m.layers[i].self_attn.q_proj.weight)
    for i in range(L):
        write(f, m.layers[i].self_attn.k_proj.weight)
    for i in range(L):
        write(f, m.layers[i].self_attn.v_proj.weight)
    for i in range(L):
        write(f, m.layers[i].self_attn.o_proj.weight)
    for i in range(L):
        write(f, m.layers[i].post_attention_layernorm.weight)
    for i in range(L):
        write(f, m.layers[i].mlp.gate_proj.weight)
    for i in range(L):
        write(f, m.layers[i].mlp.down_proj.weight)
    for i in range(L):
        write(f, m.layers[i].mlp.up_proj.weight)
    write(f, m.norm.weight)
    write(f, model.lm_head.weight)

print(f"written to {WEIGHTS_BIN_PATH}")
