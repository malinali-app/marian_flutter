#!/usr/bin/env bash
set -euo pipefail

MODEL="/mnt/c/Users/PierreGancel/Documents/git_malinali/malinali-app/training/fr-pul/models/fr-pul"
CONVERT="/mnt/c/Users/PierreGancel/Documents/git_malinali/marian_flutter/scripts/convert_tokenizer.py"
FIXTURE="/mnt/c/Users/PierreGancel/Documents/git_malinali/marian_flutter/test/fixtures/fr-pul"
# Keep venv on Linux FS — pip on /mnt/c is extremely slow.
VENV="${HOME}/.venvs/marian-convert"

if [[ ! -d "$MODEL" ]]; then
  echo "Model folder missing: $MODEL" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating venv at $VENV"
  python3 -m venv "$VENV"
fi

echo "Installing Python deps..."
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "transformers>=4.40" sentencepiece protobuf
"$VENV/bin/python" -c "import transformers, sentencepiece; print('ok', transformers.__version__)"

echo "Converting tokenizers..."
"$VENV/bin/python" "$CONVERT" "$MODEL"

mkdir -p "$FIXTURE"
for name in config.json model.safetensors tokenizer-enc.json tokenizer-dec.json generation_config.json; do
  if [[ -f "$MODEL/$name" ]]; then
    cp -f "$MODEL/$name" "$FIXTURE/$name"
    echo "copied $name"
  elif [[ "$name" == "generation_config.json" ]]; then
    echo "skip missing $name"
  else
    echo "missing required $name" >&2
    exit 1
  fi
done

ls -lh "$FIXTURE"
echo "DONE"
