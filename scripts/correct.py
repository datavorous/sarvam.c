import sys
import logging
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

MODEL_ID = "sarvamai/sarvam-1"
MAX_NEW_TOKENS = 10
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1


def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.no_grad():
        model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            streamer=streamer,
        )


def main():
    tokenizer, model = load()

    if len(sys.argv) > 1:
        generate(tokenizer, model, " ".join(sys.argv[1:]))
        return

    while True:
        try:
            prompt = input("you: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if prompt.lower() in {"quit", "exit", "q"}:
            break
        if prompt:
            generate(tokenizer, model, prompt)
            print()


if __name__ == "__main__":
    main()
