---
library_name: transformers
tags:
  - nanochat
  - causal-lm
  - trust-remote-code
---

# local/nanochat-export

Exported from nanochat checkpoints with custom `transformers` remote code.

## Load

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

repo = "local/nanochat-export"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto" if torch.cuda.is_available() else None,
)

prompt = "नेपालको राजधानी "
ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
out = model.generate(ids, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Notes

- This repo uses custom model/tokenizer code (`trust_remote_code=True`).
- Checkpoint source: `sft`
- Model tag: `d15_harl_fulltokens_sdpa_bs32`
- Step: `49164`
