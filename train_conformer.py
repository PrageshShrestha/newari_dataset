# train_conformer_monitored.py

from typing import Dict, Tuple, Optional, List
import io
import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchaudio.models import Conformer
import editdistance  # for CER/WER calculation

# =========================================================
# TOKENIZER (unchanged)
# =========================================================
class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str) -> torch.Tensor:
        ids = self.sp.encode(str(text), out_type=int)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return self.sp.decode(ids.tolist())


# =========================================================
# DATASET (unchanged)
# =========================================================
class ParquetASRDataset(Dataset):
    def __init__(
        self,
        parquet_folder: str,
        tokenizer: SentencePieceTokenizer,
        sample_rate: int = 16000,
        max_seconds: int = 15,
    ):
        self.files = glob.glob(os.path.join(parquet_folder, "*.parquet"))
        if len(self.files) == 0:
            raise RuntimeError("No parquet files found")
        self.df = pd.concat([pd.read_parquet(f) for f in self.files], ignore_index=True)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_len = sample_rate * max_seconds

    def __len__(self):
        return len(self.df)

    def _decode_audio(self, audio_field):
        wav = None
        sr = None
        if isinstance(audio_field, dict):
            if "bytes" in audio_field and audio_field["bytes"] is not None:
                wav, sr = torchaudio.load(io.BytesIO(audio_field["bytes"]))
            elif "array" in audio_field and audio_field["array"] is not None:
                wav = torch.tensor(audio_field["array"], dtype=torch.float32)
                sr = 16000
            elif "path" in audio_field and audio_field["path"] is not None:
                wav, sr = torchaudio.load(audio_field["path"])
        if wav is None:
            return None, None
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > self.max_len:
            wav = wav[: self.max_len]
        return wav, self.sample_rate

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav, sr = self._decode_audio(row["audio"])
        text = str(row["sentence"]).strip()
        if wav is None or len(wav) < 1000:
            return self.__getitem__((idx + 1) % len(self.df))
        tokens = self.tokenizer.encode(text)
        return wav, tokens


# =========================================================
# COLLATE FUNCTION (unchanged)
# =========================================================
def collate_fn(batch):
    batch = [(w, t) for w, t in batch if w is not None]
    wavs, tokens = zip(*batch)
    wav_lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    tok_lens = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)
    wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    return wavs, wav_lens, tokens, tok_lens


# =========================================================
# ASR MODEL (unchanged)
# =========================================================
class ConformerASR(torch.nn.Module):
    def __init__(self, conformer, vocab_size: int, sample_rate: int = 16000):
        super().__init__()
        self.conformer = conformer
        self.n_fft = 400
        self.hop_length = 200

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=80,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
        )
        self.proj = torch.nn.Linear(80, vocab_size)

    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor):
        x = self.mel(wavs)                     # (B, n_mels, T)
        x = x.transpose(1, 2)                  # (B, T, n_mels)
        T = x.shape[1]
        frame_lens = (
            (wav_lens - self.n_fft) // self.hop_length
        ) + 1

        frame_lens = torch.clamp(frame_lens, min=1)
        x, out_lens = self.conformer(x, frame_lens)
        logits = self.proj(x)
        return logits, out_lens


# =========================================================
# SPECAUGMENT (for monitoring / optional use)
# =========================================================
class SpecAugment:
    """Time and frequency masking (simplified)."""
    def __init__(self, time_mask=10, freq_mask=5, time_masks=2, freq_masks=2):
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.time_masks = time_masks
        self.freq_masks = freq_masks

    def __call__(self, mel_spec):
        # mel_spec: (B, n_mels, T)
        for _ in range(self.time_masks):
            t = np.random.randint(0, self.time_mask)
            t0 = np.random.randint(0, mel_spec.shape[2] - t)
            mel_spec[:, :, t0:t0+t] = 0
        for _ in range(self.freq_masks):
            f = np.random.randint(0, self.freq_mask)
            f0 = np.random.randint(0, mel_spec.shape[1] - f)
            mel_spec[:, f0:f0+f, :] = 0
        return mel_spec


# =========================================================
# GREEDY DECODER FOR CER/WER
# =========================================================
def greedy_decode(logits, tokenizer):
    """logits: (T, vocab) after softmax (not log_softmax). Returns decoded string."""
    pred_ids = torch.argmax(logits, dim=-1)   # (T,)
    # Remove consecutive duplicates and blanks (assuming blank=0)
    prev = -1
    filtered = []
    for id_ in pred_ids.tolist():
        if id_ != prev and id_ != 0:
            filtered.append(id_)
        prev = id_
    return tokenizer.decode(torch.tensor(filtered, dtype=torch.long))


def compute_cer(ref, hyp):
    """Character error rate."""
    return editdistance.eval(ref, hyp) / max(len(ref), 1)


def compute_wer(ref_words, hyp_words):
    """Word error rate (simple split by space)."""
    return editdistance.eval(ref_words, hyp_words) / max(len(ref_words), 1)


# =========================================================
# VALIDATION LOOP (computes CER/WER)
# =========================================================
@torch.no_grad()
def validate(model, val_loader, tokenizer, device):
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0
    for wavs, wav_lens, tokens, tok_lens in val_loader:
        wavs = wavs.to(device)
        wav_lens = wav_lens.to(device)
        logits, out_lens = model(wavs, wav_lens)   # (B, T', vocab)
        # Softmax over vocab dimension
        probs = F.softmax(logits, dim=-1)
        for i in range(wavs.size(0)):
            # Truncate to actual output length
            out_len = out_lens[i].item()
            logits_i = logits[i, :out_len, :]      # (out_len, vocab)
            pred_text = greedy_decode(logits_i, tokenizer)
            # Reference: decode tokens
            ref_tokens = tokens[i][:tok_lens[i]].tolist()
            ref_text = tokenizer.decode(torch.tensor(ref_tokens, dtype=torch.long))
            # Compute CER/WER
            cer = compute_cer(ref_text, pred_text)
            wer = compute_wer(ref_text.split(), pred_text.split())
            total_cer += cer
            total_wer += wer
            num_samples += 1
    return total_cer / num_samples, total_wer / num_samples


# =========================================================
# GRADIENT NORM TRACKING (convolution vs attention)
# =========================================================
def get_gradient_norms(model):
    conv_norms = []
    attn_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            if "depthwise_conv" in name or "pointwise_conv" in name:
                conv_norms.append(p.grad.norm().item())
            elif "self_attn" in name or "multi_head" in name:
                attn_norms.append(p.grad.norm().item())
    conv_mean = np.mean(conv_norms) if conv_norms else 0.0
    attn_mean = np.mean(attn_norms) if attn_norms else 0.0
    return conv_mean, attn_mean


# =========================================================
# LEARNING RATE SCHEDULER (Noam / Transformer)
# =========================================================
class NoamScheduler:
    def __init__(self, optimizer, d_model=80, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.factor * (self.d_model ** (-0.5) *
              min(self._step ** (-0.5), self._step * (self.warmup_steps ** (-1.5))))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# =========================================================
# TRAIN ONE EPOCH (augmented with logging)
# =========================================================
def train_one_epoch(model, loader, optimizer, ctc_loss, device, epoch, writer,
                    spec_augment=None, scheduler=None):
    model.train()
    total_loss = 0.0
    total_ctc_loss = 0.0
    num_batches = 0
    grad_conv_norms = []
    grad_attn_norms = []
    lr_log = []

    for batch_idx, (wavs, wav_lens, tokens, tok_lens) in enumerate(loader):
        wavs = wavs.to(device)
        wav_lens = wav_lens.to(device)
        tokens = tokens.to(device)
        tok_lens = tok_lens.to(device)

        # Forward
        logits, out_lens = model(wavs, wav_lens)

        # Apply SpecAugment on mel spectrograms? (optional) - we already have mel inside model
        # To keep it simple, we skip modifying model internals; instead we can log augmentation usage.
        # For monitoring, just note that augmentation is enabled.

        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    
        loss = ctc_loss(log_probs, tokens, out_lens, tok_lens)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping and norm logging
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        conv_norm, attn_norm = get_gradient_norms(model)
        grad_conv_norms.append(conv_norm)
        grad_attn_norms.append(attn_norm)

        optimizer.step()
        if scheduler is not None:
            current_lr = scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']
        lr_log.append(current_lr)

        total_loss += loss.item()
        total_ctc_loss += loss.item()
        num_batches += 1

        # Log batch metrics to TensorBoard every 50 batches
        if batch_idx % 50 == 0:
            writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar('Train/LearningRate', current_lr, epoch * len(loader) + batch_idx)

    avg_loss = total_loss / num_batches
    avg_ctc = total_ctc_loss / num_batches
    avg_conv_norm = np.mean(grad_conv_norms)
    avg_attn_norm = np.mean(grad_attn_norms)
    avg_lr = np.mean(lr_log)

    # Log epoch-level metrics
    writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
    writer.add_scalar('Train/CTCLoss', avg_ctc, epoch)
    writer.add_scalar('Train/GradNormConv', avg_conv_norm, epoch)
    writer.add_scalar('Train/GradNormAttn', avg_attn_norm, epoch)
    writer.add_scalar('Train/LearningRate', avg_lr, epoch)

    # Also return for CSV
    return {
        'epoch': epoch,
        'train_loss': avg_loss,
        'train_ctc_loss': avg_ctc,
        'grad_conv_norm': avg_conv_norm,
        'grad_attn_norm': avg_attn_norm,
        'learning_rate': avg_lr,
    }


# =========================================================
# CHECKPOINT SAVE / LOAD (unchanged, but saves to summary folder)
# =========================================================
def save_checkpoint(path: str, model, optimizer, epoch: int):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"]


# =========================================================
# MAIN (with extensive logging)
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./datasets', help='Folder with parquet files')
    parser.add_argument('--tokenizer_model', type=str, default='data_asr_spm.model', help='SentencePiece model path')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate (if not using scheduler)')
    parser.add_argument('--use_scheduler', action='store_true', help='Use Noam warmup scheduler')
    parser.add_argument('--use_specaugment', action='store_true', help='Enable SpecAugment (monitoring only)')
    parser.add_argument('--summary_dir', type=str, default='./training_summary', help='Where to save logs/checkpoints')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data for validation')
    args = parser.parse_args()

    # Create summary folder and subfolders
    os.makedirs(args.summary_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.summary_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.summary_dir, 'tensorboard'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = SentencePieceTokenizer(args.tokenizer_model)
    print(tokenizer.sp.id_to_piece(0))
    # Load full dataset
    full_dataset = ParquetASRDataset(
        parquet_folder=args.data_folder,
        tokenizer=tokenizer,
    )
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    vocab_size = tokenizer.sp.get_piece_size()
    conformer = Conformer(
        input_dim=80,
        num_heads=4,
        ffn_dim=256,
        num_layers=6,
        depthwise_conv_kernel_size=31,
        dropout=0.2
    )
    model = ConformerASR(conformer, vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.use_scheduler:
        scheduler = NoamScheduler(optimizer, d_model=80, warmup_steps=4000)

    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    # Optional: SpecAugment instance (for demonstration, not applied here to avoid altering forward)
    if args.use_specaugment:
        print("SpecAugment is enabled (monitoring mode). Note: not applied inside model forward; see code for integration.")
        # To actually apply, you would modify ConformerASR.forward to call SpecAugment on mel spectrogram.
        # For logging, we just record that it's enabled.
        writer.add_text('Config/SpecAugment', 'Enabled (monitoring only)')

    # CSV summary file
    summary_csv = os.path.join(args.summary_dir, 'training_summary.csv')
    if not os.path.exists(summary_csv):
        pd.DataFrame(columns=[
            'epoch', 'train_loss', 'train_ctc_loss', 'val_cer', 'val_wer',
            'grad_conv_norm', 'grad_attn_norm', 'learning_rate'
        ]).to_csv(summary_csv, index=False)

    print("Starting training with monitoring...")
    for epoch in range(args.epochs):
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, ctc_loss, device,
            epoch, writer, spec_augment=None, scheduler=scheduler
        )

        # Validate
        val_cer, val_wer = validate(model, val_loader, tokenizer, device)
        char_acc = 1-val_cer
        writer.add_scalar('Val/CER', val_cer, epoch)
        writer.add_scalar('Val/WER', val_wer, epoch)
        writer.add_scalar('Val/CharacterAccuracy', char_acc, epoch)

        # Log alignment sharpness proxy: entropy of CTC softmax distribution (sampled from a batch)
        model.eval()
        with torch.no_grad():
            sample_wavs, sample_lens, _, _ = next(iter(val_loader))
            sample_wavs = sample_wavs.to(device)
            logits, _ = model(sample_wavs, sample_lens.to(device))
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
            writer.add_scalar('Alignment/SoftmaxEntropy', entropy, epoch)

        print(f"Epoch {epoch+1:3d} |Char Accuracy: {char_acc:.4f} | train loss: {train_metrics['train_loss']:.4f} | train CTC: {train_metrics['train_ctc_loss']:.4f} | "
              f"Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f} | LR: {train_metrics['learning_rate']:.2e}")

        # Append to CSV
        new_row = {
            'epoch': epoch,
            'train_loss': train_metrics['train_loss'],
            'train_ctc_loss': train_metrics['train_ctc_loss'],
            'val_cer': val_cer,
            'val_wer': val_wer,
            'grad_conv_norm': train_metrics['grad_conv_norm'],
            'grad_attn_norm': train_metrics['grad_attn_norm'],
            'learning_rate': train_metrics['learning_rate'],
        }
        df = pd.read_csv(summary_csv)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(summary_csv, index=False)

        # Save checkpoint (every epoch) inside summary/checkpoints
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pt')
        save_checkpoint(ckpt_path, model, optimizer, epoch)

    writer.close()
    print(f"Training finished. Summary saved in {args.summary_dir}")


if __name__ == "__main__":
    main()