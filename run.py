from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device="cuda"
repo = "./tokenizer_dir"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

prompt = "नेपालको राजधानी "
ids = tok(prompt, return_tensors="pt").input_ids.to(device)
print(prompt)
print(ids)
