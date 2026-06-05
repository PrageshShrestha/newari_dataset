import io
import torch
import torchaudio
import sentencepiece as spm

from torchaudio.models import Conformer


# =========================================================
# TOKENIZER
# =========================================================

class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text: str):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)


# =========================================================
# MODEL
# =========================================================

class ConformerASR(torch.nn.Module):
    def __init__(
        self,
        conformer,
        vocab_size: int,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.conformer = conformer

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=80,
            center=False,
        )

        self.proj = torch.nn.Linear(80, vocab_size)

    def forward(self, wavs, wav_lens):

        x = self.mel(wavs)

        x = x.transpose(1, 2)

        frame_lens = torch.full(
            (x.size(0),),
            x.size(1),
            device=x.device,
            dtype=torch.long,
        )

        x, out_lens = self.conformer(
            x,
            frame_lens,
        )

        logits = self.proj(x)

        return logits, out_lens


# =========================================================
# LOAD MODEL
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = SentencePieceTokenizer(
    "asr_spm.model"
)

VOCAB_SIZE = tokenizer.sp.get_piece_size()

conformer = Conformer(
    input_dim=80,
    num_heads=4,
    ffn_dim=256,
    num_layers=6,
    depthwise_conv_kernel_size=31,
)

model = ConformerASR(
    conformer,
    vocab_size=VOCAB_SIZE,
)

checkpoint = torch.load(
    "conformer_asr_ckpt.pt",
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.to(DEVICE)
model.eval()

print(
    f"Loaded checkpoint from epoch {checkpoint['epoch']}"
)


# =========================================================
# AUDIO LOADING
# =========================================================

def load_audio(path):

    wav, sr = torchaudio.load(path)

    if wav.dim() == 2:
        wav = wav.mean(dim=0)

    if sr != 16000:
        wav = torchaudio.functional.resample(
            wav,
            sr,
            16000,
        )

    return wav


# =========================================================
# GREEDY CTC DECODER
# =========================================================

def greedy_decode(logits):

    predicted_ids = torch.argmax(
        logits,
        dim=-1,
    )

    predicted_ids = predicted_ids[0]

    blank_id = 0

    tokens = []
    previous = None

    for idx in predicted_ids.tolist():

        if idx == blank_id:
            previous = None
            continue

        if idx == previous:
            continue

        tokens.append(idx)
        previous = idx

    return tokens


# =========================================================
# TOKEN VISUALIZATION
# =========================================================

def print_token_details(token_ids):

    print("\nToken IDs:")
    print(token_ids)

    print("\nSentencePiece Tokens:")

    for tid in token_ids:
        piece = tokenizer.sp.id_to_piece(tid)
        print(f"{tid:5d} -> {piece}")

    print("\nDecoded Text:")
    print(tokenizer.decode(token_ids))


# =========================================================
# INFERENCE
# =========================================================

def transcribe(audio_path):

    wav = load_audio(audio_path)

    wav = wav.unsqueeze(0)

    wav_lens = torch.tensor(
        [wav.shape[1]],
        dtype=torch.long,
    )

    wav = wav.to(DEVICE)
    wav_lens = wav_lens.to(DEVICE)

    with torch.no_grad():

        logits, _ = model(
            wav,
            wav_lens,
        )

    token_ids = greedy_decode(logits)

    print_token_details(token_ids)

    text = tokenizer.decode(token_ids)

    return text


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    result = transcribe("audio.wav")

    print("\nFinal Transcript:")
    print(result)