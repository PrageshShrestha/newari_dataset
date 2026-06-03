import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_nanochat import NanochatConfig


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


def _detect_compute_dtype(device):
    if device.type == "cuda":
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(idx)
        if (major, minor) >= (8, 0):
            return torch.bfloat16
    return torch.float32


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def _has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def _apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    # q/k/v are (B, H, T, D)
    t_q = q.size(2)
    t_k = k.size(2)
    left_window = window_size[0]

    # Full causal attention when the window covers full context.
    if (left_window < 0 or left_window >= t_q) and t_q == t_k:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single-token decode path.
    if t_q == 1:
        if left_window >= 0 and left_window < t_k:
            start = max(0, t_k - (left_window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Build explicit causal (+ optional sliding-window) mask.
    device = q.device
    row_idx = (t_k - t_q) + torch.arange(t_q, device=device).unsqueeze(1)
    col_idx = torch.arange(t_k, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if left_window >= 0 and left_window < t_k:
        mask = mask & ((row_idx - col_idx) <= left_window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def _flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    if not causal:
        raise NotImplementedError("Nanochat HF export currently supports only causal attention")
    # SDPA fallback mirroring nanochat.flash_attention semantics.
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size=window_size, enable_gqa=enable_gqa)
    return y.transpose(1, 2)


class NanochatAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if _has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        bsz, seqlen, _ = x.size()
        q = self.c_q(x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(bsz, seqlen, self.n_kv_head, self.head_dim)
            gate = 3.0 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        q = 1.2 * _norm(q)
        k = 1.2 * _norm(k)

        y = _flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(bsz, seqlen, -1)
        return self.c_proj(y)


class NanochatMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class NanochatBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = NanochatAttention(config, layer_idx)
        self.mlp = NanochatMLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(_norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(_norm(x))
        return x


class NanochatBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.padded_vocab_size, config.n_embd),
                "h": nn.ModuleList([NanochatBlock(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if _has_ve(i, config.n_layer)
            }
        )
        self.window_sizes = self._compute_window_sizes(config)
        self.rotary_seq_len = config.sequence_len * 10
        self.register_buffer("cos", torch.empty(1), persistent=False)
        self.register_buffer("sin", torch.empty(1), persistent=False)
        self._refresh_rotary()

    def _precompute_rotary_embeddings(self, seq_len, head_dim, device, dtype):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (100000 ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :].to(dtype=dtype)
        sin = freqs.sin()[None, :, None, :].to(dtype=dtype)
        return cos, sin

    def _refresh_rotary(self, device=None, dtype=None):
        head_dim = self.config.n_embd // self.config.n_head
        if device is None:
            device = self.transformer.wte.weight.device
        if dtype is None:
            dtype = self.transformer.wte.weight.dtype
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=device, dtype=dtype)
        self.cos = cos
        self.sin = sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        lut = {"L": (long_window, 0), "S": (short_window, 0)}
        out = []
        for layer_idx in range(config.n_layer):
            ch = pattern[layer_idx % len(pattern)]
            out.append(lut[ch])
        out[-1] = (long_window, 0)
        return out

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        compute_dtype = _detect_compute_dtype(input_ids.device)
        if self.cos.device != input_ids.device or self.cos.dtype != compute_dtype:
            self._refresh_rotary(device=input_ids.device, dtype=compute_dtype)

        if seqlen > self.cos.size(1):
            raise ValueError(
                f"Sequence length {seqlen} exceeds rotary cache length {self.cos.size(1)}. "
                "Re-export with larger sequence_len if needed."
            )
        cos_sin = self.cos[:, :seqlen], self.sin[:, :seqlen]

        x = self.transformer["wte"](input_ids)
        x = x.to(compute_dtype)
        x = _norm(x)
        if seqlen > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](input_ids).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = _norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)
        return logits


class NanochatForCausalLM(PreTrainedModel):
    config_class = NanochatConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.model = NanochatBackbone(config)

    @property
    def all_tied_weights_keys(self):
        # Compatibility shim for some transformers/accelerate versions that
        # access `model.all_tied_weights_keys` during device_map inference.
        return {k: None for k in getattr(self, "_tied_weights_keys", [])}

    def get_input_embeddings(self):
        return self.model.transformer["wte"]

    def set_input_embeddings(self, value):
        self.model.transformer["wte"] = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_cache=None,
        past_key_values=None,
        return_dict=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )

        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids}
