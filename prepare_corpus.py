"""
prepare_corpus.py

Converts a raw text corpus (one sentence per line) into a space-separated
BPE-piece corpus that KenLM's lmplz can train on.

Why this is needed:
- KenLM's lmplz tokenizes its input by whitespace. Our BPE tokenizer's
  pieces are NOT naturally whitespace-separated (e.g. "छगु" -> "▁छ" + "गु"),
  so we must explicitly join pieces with spaces ourselves before handing
  the corpus to lmplz.
- We reuse the EXACT same SentencePiece .model file used for the ASR
  acoustic model, so the autocomplete model speaks the same "vocabulary"
  as the ASR system's outputs. This means you could later combine them
  (e.g. autocomplete suggestions phrased in the same tokens the ASR model
  would produce).

Usage:
    python prepare_corpus.py \
        --input dataset.txt \
        --spm_model data_asr_spm.model \
        --output corpus_pieces.txt
"""

import argparse
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                         help="Raw text file, one sentence per line.")
    parser.add_argument("--spm_model", required=True,
                         help="Path to your existing SentencePiece .model file.")
    parser.add_argument("--output", default="corpus_pieces.txt",
                         help="Where to write the space-separated piece corpus.")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)
    print(f"Loaded tokenizer with vocab size: {sp.get_piece_size()}")

    n_lines = 0
    n_empty_skipped = 0

    with open(args.input, encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:
        for raw_line in f_in:
            line = raw_line.strip()
            if not line:
                n_empty_skipped += 1
                continue
            pieces = sp.encode(line, out_type=str)
            if not pieces:
                n_empty_skipped += 1
                continue
            f_out.write(" ".join(pieces) + "\n")
            n_lines += 1

    print(f"Wrote {n_lines} tokenized lines to {args.output}")
    if n_empty_skipped:
        print(f"Skipped {n_empty_skipped} empty/unencodable lines")


if __name__ == "__main__":
    main()
