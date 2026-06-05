import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str) -> torch.Tensor:
        ids = self.sp.encode(str(text), out_type=int)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return self.sp.decode(ids.tolist())

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NextWordDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: SentencePieceTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = []

        if os.path.isdir(data_path):
            files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jsonl')]
        else:
            files = [data_path]

        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get('text', '')
                        if text:
                            self.lines.append(text)
                    except:
                        continue

        if len(self.lines) == 0:
            raise RuntimeError(f"No valid JSONL lines with 'text' field found in {data_path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_len:
            # random crop instead of always taking the front — acts as augmentation
            start = torch.randint(0, len(tokens) - self.max_len + 1, (1,)).item()
            tokens = tokens[start: start + self.max_len]
        return tokens


def collate_lm(batch):
    lengths = torch.tensor([len(seq) for seq in batch], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return padded, lengths


# ---------------------------------------------------------------------------
# Token-level dropout augmentation (applied inside the training loop)
# Randomly replaces input tokens with the <unk>/pad id to force the model
# to rely on context rather than memorizing exact token sequences.
# ---------------------------------------------------------------------------

def token_dropout(input_ids: torch.Tensor, rate: float = 0.1, pad_id: int = 0) -> torch.Tensor:
    if rate <= 0.0:
        return input_ids
    mask = torch.bernoulli(torch.full(input_ids.shape, rate, device=input_ids.device)).bool()
    return input_ids.masked_fill(mask, pad_id)


# ---------------------------------------------------------------------------
# Model — Decoder-only Transformer (GPT-style)
#
# Design choices:
#   - Pre-LayerNorm (GPT-2 style): normalise before attention/FFN, not after.
#     More stable training, less sensitive to lr.
#   - Rotary Position Embeddings (RoPE): better length generalisation than
#     learned absolute positions, no extra parameters.
#   - SwiGLU feed-forward (LLaMA style): gated activation with two projections
#     instead of one; empirically outperforms ReLU/GELU FFN at every scale.
#   - No bias in attention projections (follows PaLM / LLaMA finding).
#   - Tied input/output embeddings: halves the effective parameter count in the
#     lm_head, strong regulariser for small vocab models.
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (no mean-centering, no bias).
    Faster and equally effective as LayerNorm for transformer blocks."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope_freqs(head_dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Returns complex exponentials of shape (max_len, head_dim // 2)."""
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, theta)               # (max_len, head_dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: (B, T, n_heads, head_dim)  →  same shape with RoPE applied."""
    B, T, H, D = x.shape
    xc = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(2)         # (1, T, 1, D//2)
    xc = xc * freqs
    return torch.view_as_real(xc).reshape(B, T, H, D).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # fused QKV projection — no bias (empirically better for small models)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # RoPE frequencies (registered as buffer so they move with .to(device))
        freqs = precompute_rope_freqs(self.head_dim, max_len)
        self.register_buffer('rope_freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                         # each (B, T, H, D)

        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # reshape to (B, H, T, D) for scaled dot-product attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash-attention compatible path (uses torch SDPA when available)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_drop(self.out_proj(y))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: two gate projections, no bias.
    hidden_dim is typically 8/3 * d_model (LLaMA convention), rounded to
    a multiple of 64 for efficiency."""
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_hidden: int,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model, ff_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))    # pre-norm + residual
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderOnlyLM(nn.Module):
    """Compact GPT-style decoder with RoPE, RMSNorm, and SwiGLU.
    Suitable for low-resource language modelling tasks."""
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 4, ff_hidden: int = 0,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        # ff_hidden: default to LLaMA-style 8/3 * d_model rounded to mult of 64
        if ff_hidden <= 0:
            ff_hidden = int(8 / 3 * d_model)
            ff_hidden = ((ff_hidden + 63) // 64) * 64

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_hidden, dropout, max_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying — lm_head reuses embedding weights
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Gradient norm tracking
# ---------------------------------------------------------------------------

def get_gradient_norms(model):
    attn_norms, ffn_norms = [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm = p.grad.norm().item()
        if 'attn' in name or 'qkv' in name or 'out_proj' in name:
            attn_norms.append(norm)
        elif 'ffn' in name or 'proj' in name:
            ffn_norms.append(norm)
    return (
        np.mean(attn_norms) if attn_norms else 0.0,
        np.mean(ffn_norms)  if ffn_norms  else 0.0,
    )


# ---------------------------------------------------------------------------
# Cosine LR scheduler with linear warmup
# More suitable than Noam for small d_model models.
# ---------------------------------------------------------------------------

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        s = self._step
        if s <= self.warmup_steps:
            scale = s / max(1, self.warmup_steps)
        else:
            progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale
        return self.optimizer.param_groups[0]['lr']


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, device, pad_id=0):
    model.eval()
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    for batch_ids, lengths in val_loader:
        batch_ids = batch_ids.to(device)
        input_ids = batch_ids[:, :-1]
        targets   = batch_ids[:, 1:]
        if input_ids.size(1) == 0:
            continue
        logits = model(input_ids)
        loss   = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        mask   = targets != pad_id
        preds  = logits.argmax(dim=-1)
        total_correct += (preds == targets)[mask].sum().item()
        total_tokens  += mask.sum().item()
        total_loss    += loss.item() * mask.sum().item()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, np.exp(avg_loss), total_correct / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch,
                    writer, pad_id=0, label_smoothing=0.1, token_drop_rate=0.1):
    model.train()
    total_loss, total_tokens = 0.0, 0
    grad_attn, grad_ffn, lrs = [], [], []
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    for batch_idx, (batch_ids, lengths) in enumerate(loader):
        batch_ids = batch_ids.to(device)

        # teacher-forcing with optional token dropout on inputs
        input_ids = batch_ids[:, :-1]
        targets   = batch_ids[:, 1:]
        if input_ids.size(1) == 0:
            continue

        input_ids = token_dropout(input_ids, rate=token_drop_rate, pad_id=pad_id)

        logits = model(input_ids)
        loss   = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        a_norm, f_norm = get_gradient_norms(model)
        grad_attn.append(a_norm)
        grad_ffn.append(f_norm)

        optimizer.step()
        lr = scheduler.step() if scheduler else optimizer.param_groups[0]['lr']
        lrs.append(lr)

        mask = targets != pad_id
        total_loss   += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

        if batch_idx % 50 == 0:
            writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(loader) + batch_idx)

    return {
        'train_loss':      total_loss / max(total_tokens, 1),
        'grad_attn_norm':  np.mean(grad_attn) if grad_attn else 0.0,
        'grad_ffn_norm':   np.mean(grad_ffn)  if grad_ffn  else 0.0,
        'learning_rate':   np.mean(lrs)        if lrs       else 0.0,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 20,
             temperature: float = 0.8, top_p: float = 0.9, device: str = 'cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :] / temperature
        # nucleus (top-p) sampling
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_logits[cum_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float('-inf')
        probs = F.softmax(sorted_logits, dim=-1)
        next_token = sorted_idx[0, torch.multinomial(probs[0], 1)]
        if next_token.item() == 0:
            break
        print(tokens , next_token)
        tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    return tokenizer.decode(tokens[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',       type=str, default = "newari_dataset.jsonl")
    parser.add_argument('--tokenizer_model', type=str, default='data_asr_spm.model')
    parser.add_argument('--summary_dir',     type=str, default='./next_word_model')
    parser.add_argument('--batch_size',      type=int, default=32)
    parser.add_argument('--epochs',          type=int, default=100)
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--warmup_steps',    type=int, default=200,
                        help='Linear warmup steps for cosine scheduler')
    # model
    parser.add_argument('--d_model',         type=int, default=128,
                        help='Embedding / hidden dimension')
    parser.add_argument('--num_layers',      type=int, default=4)
    parser.add_argument('--num_heads',       type=int, default=4,
                        help='Attention heads — must divide d_model evenly')
    parser.add_argument('--ff_hidden',       type=int, default=0,
                        help='FFN hidden dim (0 = auto: 8/3 * d_model)')
    parser.add_argument('--max_seq_len',     type=int, default=128)
    # regularisation
    parser.add_argument('--dropout',         type=float, default=0.2)
    parser.add_argument('--weight_decay',    type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--token_drop_rate', type=float, default=0.1,
                        help='Probability of replacing an input token with pad (augmentation)')
    # training
    parser.add_argument('--val_split',       type=float, default=0.1)
    parser.add_argument('--patience',        type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.summary_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.summary_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.summary_dir, 'tensorboard'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer  = SentencePieceTokenizer(args.tokenizer_model)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # ---- dataset ----
    full_dataset = NextWordDataset(args.data_path, tokenizer, max_len=args.max_seq_len)
    filtered = [i for i in range(len(full_dataset)) if len(full_dataset[i]) >= 2]
    full_dataset = Subset(full_dataset, filtered)
    print(f"Total usable sequences (len>=2): {len(full_dataset)}")

    val_size   = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_lm, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_lm, num_workers=2, pin_memory=True)

    # ---- model ----
    model = DecoderOnlyLM(
        vocab_size = vocab_size,
        d_model    = args.d_model,
        n_layers   = args.num_layers,
        n_heads    = args.num_heads,
        ff_hidden  = args.ff_hidden,
        max_len    = args.max_seq_len,
        dropout    = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # separate weight decay: no decay on norms / biases
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,    'weight_decay': args.weight_decay},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler   = CosineWarmupScheduler(optimizer, args.warmup_steps, total_steps)

    # ---- CSV log ----
    csv_path = os.path.join(args.summary_dir, 'training_summary.csv')
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=[
            'epoch', 'train_loss', 'val_loss', 'val_perplexity', 'val_accuracy',
            'grad_attn_norm', 'grad_ffn_norm', 'learning_rate',
        ]).to_csv(csv_path, index=False)

    best_ppl         = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, writer,
            label_smoothing=args.label_smoothing,
            token_drop_rate=args.token_drop_rate,
        )
        val_loss, val_ppl, val_acc = validate(model, val_loader, device)

        writer.add_scalar('Val/Loss',       val_loss, epoch)
        writer.add_scalar('Val/Perplexity', val_ppl,  epoch)
        writer.add_scalar('Val/Accuracy',   val_acc,  epoch)
        for k, v in train_metrics.items():
            writer.add_scalar(f'Train/{k}', v, epoch)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_metrics['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"PPL: {val_ppl:.2f} | "
              f"Acc: {val_acc:.4f} | "
              f"LR: {train_metrics['learning_rate']:.2e}")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pt'))
            patience_counter = 0
            print(" => Saved new best model.")
        else:
            patience_counter += 1

        row = {
            'epoch':          epoch,
            'train_loss':     train_metrics['train_loss'],
            'val_loss':       val_loss,
            'val_perplexity': val_ppl,
            'val_accuracy':   val_acc,
            'grad_attn_norm': train_metrics['grad_attn_norm'],
            'grad_ffn_norm':  train_metrics['grad_ffn_norm'],
            'learning_rate':  train_metrics['learning_rate'],
        }
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        # if patience_counter >= args.patience:
        #     print(f"\nEarly stopping at epoch {epoch+1}.")
        #     break

    writer.close()
    print(f"\nTraining complete. Outputs saved to: {args.summary_dir}")

    # ---- sample generation ----
    print("\n--- Sample generation (nucleus sampling, top-p=0.9) ---")
    prompt = "आः गुलि ई"
    result = generate(model, tokenizer, prompt, max_new_tokens=20,
                      temperature=0.8, top_p=0.9, device=device)
    print(f"Prompt:       {prompt}")
    print(f"Continuation: {result}")


if __name__ == "__main__":
    main()