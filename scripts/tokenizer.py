import struct
import urllib.request
from pathlib import Path

SPM_URL = "https://huggingface.co/sarvamai/sarvam-1/resolve/main/tokenizer.model"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TOKENIZER_MODEL_PATH = ARTIFACTS_DIR / "tokenizer.model"
TOKENIZER_BIN_PATH = ARTIFACTS_DIR / "tokenizer.bin"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if not TOKENIZER_MODEL_PATH.exists():
    print(f"downloading {SPM_URL} ...")
    urllib.request.urlretrieve(SPM_URL, TOKENIZER_MODEL_PATH)
    print("done")

import sentencepiece as spm

sp = spm.SentencePieceProcessor(str(TOKENIZER_MODEL_PATH))

max_len = max(
    len(sp.id_to_piece(i).encode("utf-8")) for i in range(sp.get_piece_size())
)

with open(TOKENIZER_BIN_PATH, "wb") as f:
    f.write(struct.pack("i", max_len))
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i).encode("utf-8")
        score = sp.get_score(i)
        f.write(struct.pack("f", score))
        f.write(struct.pack("i", len(piece)))
        f.write(piece)

print(f"vocab size: {sp.get_piece_size()}")
print(f"written to {TOKENIZER_BIN_PATH}")
