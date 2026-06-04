from typing import Dict, Tuple

import os
import glob
import torch
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torchaudio.models import Conformer
import pandas as pd

class SentencePieceTokenizer:
    r"""
    SentencePiece tokenizer wrapper for ASR training.

    Args:
        model_path (str): Path to SentencePiece model file (.model).
    """

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.blank_id = 0

    def encode(self, text: str) -> torch.Tensor:
        r"""
        Encode input text into token IDs.

        Args:
            text (str): Input transcription text.

        Returns:
            torch.Tensor: 1D tensor of token IDs.
        """
        ids = self.sp.encode(text, out_type=int)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor):
        r"""
        Decode token IDs back into text.

        Args:
            ids (torch.Tensor): Token ID sequence.

        Returns:
            str: Decoded text.
        """
        return self.sp.decode(ids.tolist())


class ParquetASRDataset(Dataset):
    def __init__(
        self,
        parquet_folder: str,
        tokenizer: SentencePieceTokenizer,
        sample_rate: int = 16000,
    ):
        self.files = glob.glob(os.path.join(parquet_folder, "*.parquet"))
        self.df = pd.concat([pd.read_parquet(f) for f in self.files], ignore_index=True)

        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def _load_audio(self, audio_field):
        if isinstance(audio_field, dict):
            wav = audio_field.get("array", None)
            if wav is None:
                wav = audio_field.get("audio", None)
        else:
            wav = audio_field
        print(wav)
        print(type(wav))
        if not isinstance(wav, torch.Tensor):
            
            wav = torch.tensor(wav, dtype=torch.float32)

        return wav

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        wav = self._load_audio(row["audio"])
        tokens = self.tokenizer.encode(row["sentence"])

        return wav, tokens


def collate_fn(batch):
    r"""
    Collate function for batching variable-length audio and token sequences.

    Args:
        batch: List of (waveform, tokens) pairs.

    Returns:
        Tuple containing:
            - padded waveforms
            - waveform lengths
            - padded tokens
            - token lengths
    """
    wavs, tokens = zip(*batch)

    wav_lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    tok_lens = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)

    wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

    return wavs, wav_lens, tokens, tok_lens


class ConformerASR(torch.nn.Module):
    r"""
    Conformer-based Automatic Speech Recognition model.

    Architecture:
        MelSpectrogram -> Conformer Encoder -> Linear Projection -> Vocabulary logits

    Args:
        conformer (torch.nn.Module): Conformer encoder.
        vocab_size (int): Size of output vocabulary.
        n_mels (int, optional): Number of mel filterbanks. Default: 80.
        sample_rate (int, optional): Audio sample rate. Default: 16000.
    """

    def __init__(
        self,
        conformer,
        vocab_size: int,
        n_mels: int = 80,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.conformer = conformer

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        self.proj = torch.nn.Linear(n_mels, vocab_size)

    def forward(
        self,
        wavs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of ASR model.

        Args:
            wavs (torch.Tensor): Batch of waveforms (B, T).
            lengths (torch.Tensor): Valid waveform lengths (B,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                logits (B, T', vocab_size),
                output lengths (B,)
        """
        x = self.mel(wavs)
        x = x.transpose(1, 2)

        x, out_lens = self.conformer(x, lengths)
        logits = self.proj(x)

        return logits, out_lens


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss: torch.nn.CTCLoss,
    device: torch.device,
) -> float:
    r"""
    Executes one full training epoch.

    Args:
        model (torch.nn.Module): ASR model.
        loader (DataLoader): Training dataloader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        ctc_loss (torch.nn.CTCLoss): CTC loss function.
        device (torch.device): Computation device.

    Returns:
        float: Total training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for wavs, wav_lens, tokens, tok_lens in loader:
        wavs = wavs.to(device)
        wav_lens = wav_lens.to(device)
        tokens = tokens.to(device)
        tok_lens = tok_lens.to(device)

        logits, out_lens = model(wavs, wav_lens)

        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        loss = ctc_loss(log_probs, tokens, out_lens, tok_lens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def main(conformer: torch.nn.Module):
    r"""
    Training entry point for Conformer-based ASR.

    Args:
        conformer (torch.nn.Module): Initialized Conformer encoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SentencePieceTokenizer("asr_spm.model")

    dataset = ParquetASRDataset(
        parquet_folder="./datasets",
        tokenizer=tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    vocab_size = tokenizer.sp.get_piece_size()

    model = ConformerASR(
        conformer=conformer,
        vocab_size=vocab_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    epochs = 1000

    for epoch in range(epochs):
        loss = train_one_epoch(model, loader, optimizer, ctc_loss, device)
        print(f"epoch {epoch + 1} loss {loss:.4f}")


conformer = Conformer(
    input_dim=80,
    num_heads=4,
    ffn_dim=256,
    num_layers=6,
    depthwise_conv_kernel_size=31,
)

main(conformer)