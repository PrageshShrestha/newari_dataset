# Setup & Training: KenLM Autocomplete for Newari (BPE-level)

## 1. Install build dependencies (one-time)

```bash
sudo apt update
sudo apt install -y build-essential cmake libboost-all-dev \
    libbz2-dev liblzma-dev zlib1g-dev libeigen3-dev
```

## 2. Get and build KenLM

```bash
cd ~/Desktop/himalaya
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This builds the binaries you need at `kenlm/build/bin/`:
- `lmplz`   -> trains the n-gram model from text
- `build_binary` -> compresses the trained model for fast loading
- `query`   -> CLI tool to sanity check a trained model (optional)

## 3. (Optional but recommended) Python bindings

From the `kenlm/` root (not `build/`):

```bash
cd ~/Desktop/himalaya/kenlm
pip install . --break-system-packages
```

This lets the prediction script load the model directly in Python via
`import kenlm` instead of shelling out to the `query` binary. If this
install step fails (missing Boost/cmake config), the prediction script
falls back to shelling out to the `query` binary — see predict.py for that path.

## 4. Prepare your corpus into BPE pieces

Using your EXISTING SentencePiece model (the same one used for the ASR
acoustic model):

```bash
cd ~/Desktop/himalaya
python sentence_piece_corpus_prep.py \
    --input dataset.txt \
    --spm_model data_asr_spm.model \
    --output corpus_pieces.txt
```

(This is the `prepare_corpus.py` script — rename/copy as you like.)

## 5. Train the n-gram model

A 4-gram or 5-gram model is a reasonable default for piece-level modeling
(your earlier vocab=500 SentencePiece log showed multi-piece words forming,
so a few pieces of context is meaningful). Start with 4-gram; try 5-gram
if you have enough data and want longer-range context.

```bash
cd ~/Desktop/himalaya
./kenlm/build/bin/lmplz -o 4 \
    --discount_fallback \
    < corpus_pieces.txt \
    > newari_pieces.arpa
```

Notes on flags:
- `-o 4` sets the n-gram order (4-gram). Try `-o 3` first if your corpus
  is very small (your earlier numbers: ~8,467 sentences / ~312k chars at
  vocab=500 is on the small side for a 5-gram to estimate well — start at
  3 or 4 and only go higher if held-out perplexity actually improves).
- `--discount_fallback` avoids a common lmplz failure ("not enough
  observations") on very small/sparse corpora like yours, where some
  n-gram counts may be too sparse for default Kneser-Ney discounting
  to estimate normally.

Then compress to a fast binary format:

```bash
./kenlm/build/bin/build_binary newari_pieces.arpa newari_pieces.binary
```

Use the `.binary` file for actual queries/inference — the `.arpa` file is
human-readable text and much slower to load.

## 6. Sanity check (optional, no Python needed)

```bash
echo "▁छ गु ▁ब्व" | ./kenlm/build/bin/query newari_pieces.binary
```

This prints per-token log-probabilities — if you see very large negative
numbers (e.g. below -15) for every token, something likely went wrong in
corpus prep (check that `corpus_pieces.txt` actually contains your real
sentences and isn't empty/misencoded).

## 7. Run autocomplete predictions

See `predict.py` — point it at `newari_pieces.binary` and your `.model`
file, and it returns top-k next-piece predictions for a given text prefix.

```bash
python predict.py \
    --kenlm_model newari_pieces.binary \
    --spm_model data_asr_spm.model \
    --prefix "छगु ब्व" \
    --top_k 5
```
