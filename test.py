import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration Constants
BASE_MODEL_ID = "google/gemma-3-270m"
ADAPTER_DIR = "./gemma3-270m-newari-pipeline"  # your training OUTPUT_DIR

def verify_environment():
    """Validates CUDA availability and local adapter artifact path existence."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA execution environment not detected. Aborting inference.")
    if not os.path.exists(ADAPTER_DIR):
        raise FileNotFoundError(f"Adapter directory '{ADAPTER_DIR}' not found.")

def main():
    verify_environment()

    print(f"[Inference] Loading Gemma default tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print(f"[Inference] Initializing base model ({BASE_MODEL_ID})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to("cuda")

    # Resize to match training-time vocab size (262,145)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.vocab_size = len(tokenizer)

    print(f"[Inference] Merging trained LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()  # merge for faster inference
    model.eval()

    print("\n" + "="*60)
    print("NEWARI LLM INFERENCE SHELL (Type 'exit' to terminate)")
    print("="*60 + "\n")

    while True:
        try:
            prompt_input = input("Enter Prompt Context: ")
            if prompt_input.strip().lower() == "exit":
                break
            if not prompt_input.strip():
                continue

            input_tensors = tokenizer(prompt_input, return_tensors="pt").to("cuda")

            with torch.no_grad():
                output_tokens = model.generate(
                    **input_tensors,
                    max_new_tokens=128,
                    do_sample=True,
                    top_k=40,
                    top_p=0.92,
                    temperature=0.4,
                    repetition_penalty=1.15,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

            input_length = input_tensors.input_ids.shape[1]
            generated_tokens = output_tokens[0][input_length:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            print(f"\nPredicted Response:\n{decoded_output}")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting inference loop.")
            break
        except Exception as eval_err:
            print(f"\n[Execution Error] Generation step failed: {eval_err}\n")

if __name__ == "__main__":
    main()