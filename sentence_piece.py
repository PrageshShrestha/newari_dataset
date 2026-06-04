import pandas as pd
import glob

parquet_files = glob.glob("./datasets/*.parquet")

all_text = []

for f in parquet_files:
    df = pd.read_parquet(f)
    all_text.extend(df["sentence"].astype(str).tolist())

with open("all_sentence.txt", "w", encoding="utf-8") as w:
    for t in all_text:
        w.write(t.strip() + "\n")
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="all_sentence.txt",
    model_prefix="asr_spm",
    vocab_size=8000,          # adjust: 4k–16k typical
    model_type="bpe",        # BEST for ASR
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=-1,
    eos_id=-1
)