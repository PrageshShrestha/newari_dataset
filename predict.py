"""
predict.py

Given a text prefix, returns the top-k most likely next BPE pieces
(and the resulting decoded text preview for each), using a KenLM n-gram
model trained on piece-tokenized Newari text.

Two backends are supported:
  1. Python bindings (`import kenlm`) - used if available. Faster, no
     subprocess overhead.
  2. CLI fallback (`query` binary) - used automatically if `kenlm` isn't
     importable. Slower (one subprocess call per candidate piece) but
     works as long as you built the KenLM binaries, even if the Python
     bindings failed to install.

Usage:
    python predict.py \
        --kenlm_model newari_pieces.binary \
        --spm_model data_asr_spm.model \
        --prefix "छगु ब्व" \
        --top_k 5

    # Interactive mode (no --prefix given):
    python predict.py --kenlm_model newari_pieces.binary --spm_model data_asr_spm.model
"""

import argparse
import subprocess
import shutil
import sys

import sentencepiece as spm

# Special pieces we never want to suggest as a "next word" completion
SKIP_PIECES = {"<pad>", "<unk>", "<s>", "</s>"}


def try_load_kenlm_python(model_path: str):
    """Returns a loaded kenlm.Model, or None if the python bindings aren't available
    or are a different, incompatible package using the same import name."""
    try:
        import kenlm
    except ImportError:
        return None
    if not hasattr(kenlm, "Model"):
        print(
            "Warning: an importable `kenlm` module was found but it has no "
            "`Model` attribute - this is likely a different/incompatible "
            "package installed under the same name, not the real KenLM "
            "bindings. Falling back to the CLI backend.",
            file=sys.stderr,
        )
        return None
    try:
        return kenlm.Model(model_path)
    except Exception as e:
        print(f"Warning: kenlm.Model() failed to load ({e}). Falling back to CLI backend.", file=sys.stderr)
        return None


class PythonBackend:
    """Scores candidates using the kenlm Python bindings directly."""

    def __init__(self, model):
        self.model = model

    def score_candidates(self, context_pieces, candidate_pieces):
        """
        context_pieces: list[str] - the BPE pieces of the prefix so far
        candidate_pieces: list[str] - vocab pieces to score as the next token
        Returns: list[(piece, score)] sorted by score descending.
        kenlm's full_scores / score work on whitespace-joined strings, with
        higher (less negative) log10 prob = more likely.
        """
        context_str = " ".join(context_pieces)
        results = []
        for piece in candidate_pieces:
            full_str = (context_str + " " + piece).strip()
            # bos=False/eos=False: we are scoring a mid-sentence continuation,
            # not a full standalone sentence, so we don't want KenLM to apply
            # sentence-start/end boundary assumptions here.
            score = self.model.score(full_str, bos=False, eos=False)
            results.append((piece, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class CLIBackend:
    """Fallback: scores candidates by shelling out to the `query` binary."""

    def __init__(self, model_path: str, query_bin: str):
        self.model_path = model_path
        self.query_bin = query_bin

    def score_candidates(self, context_pieces, candidate_pieces):
        context_str = " ".join(context_pieces)
        results = []
        for piece in candidate_pieces:
            full_str = (context_str + " " + piece).strip()
            proc = subprocess.run(
                [self.query_bin, self.model_path],
                input=full_str + "\n",
                capture_output=True,
                text=True,
            )
            # `query` prints per-token logprobs ending in a Total line like:
            #   tok1=.. tok2=..   Total: -12.345 OOV: 0
            # Note: "Total:" appears at the END of the same tab-separated
            # line as the per-token scores, NOT on its own line - so we must
            # search for "Total:" anywhere in each line, not check if the
            # line *starts with* it.
            score = None
            for line in proc.stdout.splitlines():
                if "Total:" in line:
                    try:
                        score = float(line.split("Total:")[1].split("OOV:")[0].strip())
                    except (IndexError, ValueError):
                        score = None
            if score is not None:
                results.append((piece, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def get_backend(kenlm_model_path: str):
    model = try_load_kenlm_python(kenlm_model_path)
    if model is not None:
        print("Using kenlm Python bindings.", file=sys.stderr)
        return PythonBackend(model)

    query_bin = shutil.which("query")
    if query_bin is None:
        raise RuntimeError(
            "Neither the `kenlm` Python package nor a `query` binary on PATH "
            "were found. Install the Python bindings (see SETUP.md step 3) "
            "or add kenlm/build/bin to your PATH."
        )
    print(f"Falling back to CLI backend using: {query_bin}", file=sys.stderr)
    return CLIBackend(kenlm_model_path, query_bin)


def predict_next_pieces(prefix: str, sp: spm.SentencePieceProcessor, backend, top_k: int = 5):
    context_pieces = sp.encode(prefix, out_type=str)
    vocab_size = sp.get_piece_size()
    candidate_pieces = [
        sp.id_to_piece(i) for i in range(vocab_size)
        if sp.id_to_piece(i) not in SKIP_PIECES
    ]

    scored = backend.score_candidates(context_pieces, candidate_pieces)

    if not scored:
        print(
            "Warning: scoring produced zero results - the backend may be "
            "failing to parse model output. Check that the kenlm_model path "
            "is correct and try running the underlying `query` binary "
            "manually on a test string to compare its output format.",
            file=sys.stderr,
        )

    top = scored[:top_k]
    out = []
    for piece, score in top:
        candidate_ids = sp.encode(prefix, out_type=int) + [sp.piece_to_id(piece)]
        preview_text = sp.decode(candidate_ids)
        out.append({"piece": piece, "score": score, "preview": preview_text})
    return out


# Pieces (after stripping the SentencePiece word-boundary marker "▁") that
# count as ending a sentence. Checked against the piece's stripped text so
# this matches whether or not the punctuation happens to carry a leading
# word-boundary marker (e.g. "▁।" vs "।" - both are seen in practice).
SENTENCE_ENDERS = {"।", "?", "!", "."}


def complete_sentence(prefix: str, sp: spm.SentencePieceProcessor, backend, max_pieces: int = 20):
    """
    Greedily extends `prefix` one piece at a time (always taking the single
    best-scoring next piece) until either:
      - a sentence-ending piece is produced, or
      - max_pieces pieces have been added (safety cap, in case the model
        never naturally produces a sentence-ender for this prefix).
    Returns (completed_text, pieces_added, stopped_reason).
    """
    current = prefix
    pieces_added = 0
    stopped_reason = "max_pieces_reached"

    for _ in range(max_pieces):
        results = predict_next_pieces(current, sp, backend, top_k=1)
        if not results:
            stopped_reason = "no_candidates_scored"
            break
        chosen = results[0]
        current = chosen["preview"]
        pieces_added += 1

        stripped = chosen["piece"].replace("\u2581", "").strip()  # strip BPE "▁" marker
        if stripped in SENTENCE_ENDERS:
            stopped_reason = "sentence_ender"
            break

    return current, pieces_added, stopped_reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kenlm_model", required=True,
                         help="Path to trained KenLM .binary (or .arpa) file.")
    parser.add_argument("--spm_model", required=True,
                         help="Path to the SentencePiece .model file used for training.")
    parser.add_argument("--prefix", default=None,
                         help="Text prefix to complete. If omitted, runs interactively.")
    parser.add_argument("--top_k", type=int, default=5,
                         help="Number of next-piece suggestions to return (ignored if --complete_sentence is set).")
    parser.add_argument("--complete_sentence", action="store_true",
                         help="Instead of showing top-k next pieces, greedily generate a full "
                              "sentence completion by repeatedly taking the single best next "
                              "piece until a sentence-ender (।, ?, !) or --max_pieces is reached.")
    parser.add_argument("--max_pieces", type=int, default=20,
                         help="Safety cap on how many pieces to add when --complete_sentence is set.")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)

    backend = get_backend(args.kenlm_model)

    def run_once(prefix):
        if args.complete_sentence:
            completed, n_added, reason = complete_sentence(prefix, sp, backend, max_pieces=args.max_pieces)
            print(f"\nPrefix:    {prefix!r}")
            print(f"Completed: {completed!r}")
            print(f"({n_added} piece(s) added, stopped because: {reason})")
            return
        results = predict_next_pieces(prefix, sp, backend, top_k=args.top_k)
        print(f"\nPrefix: {prefix!r}")
        for rank, r in enumerate(results, 1):
            print(f"  {rank}. piece={r['piece']!r:12s} score={r['score']:.3f}  preview={r['preview']!r}")

    if args.prefix is not None:
        run_once(args.prefix)
    else:
        print("Interactive mode. Type a prefix and press Enter (Ctrl+C to quit).")
        while True:
            try:
                prefix = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prefix.strip():
                run_once(prefix)


if __name__ == "__main__":
    main()
