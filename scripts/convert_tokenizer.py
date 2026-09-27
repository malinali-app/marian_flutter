#!/usr/bin/env python3
"""Convert Helsinki/Marian SentencePiece tokenizers to fast tokenizer.json files.

Candle's `tokenizers` crate needs JSON configs. Marian ships `source.spm` /
`target.spm` (+ `vocab.json`). This script mirrors Candle's marian-mt converter,
then patches specials from `tokenizer_config.json` so `<unk>` / `<pad>` are
actually skipped on decode (and Pulaar letters keep their HF ids).

Usage:
  python scripts/convert_tokenizer.py /path/to/models/fr-pul

Writes:
  tokenizer-enc.json  (source / French)
  tokenizer-dec.json  (target / Pulaar)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import (
    SpmConverter,
    import_protobuf,
    requires_backends,
)


class MarianConverter(SpmConverter):
    def __init__(self, *args, index: int = 0):
        requires_backends(self, "protobuf")
        super(SpmConverter, self).__init__(*args)

        model_pb2 = import_protobuf()
        m = model_pb2.ModelProto()
        with open(self.original_tokenizer.spm_files[index], "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m

        dir_path = Path(self.original_tokenizer.spm_files[0]).parent
        with open(dir_path / "vocab.json", "r", encoding="utf-8") as f:
            self._vocab = json.load(f)

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                warnings.warn(
                    "SentencePiece byte_fallback is not fully supported by fast "
                    "tokenizers; unknown characters may become <unk>."
                )

    def vocab(self, proto):
        vocab_size = max(self._vocab.values()) + 1
        vocab = [("<NIL>", -100) for _ in range(vocab_size)]
        for piece in proto.pieces:
            try:
                index = self._vocab[piece.piece]
            except KeyError:
                print(f"Ignored missing piece {piece.piece}")
                continue
            vocab[index] = (piece.piece, piece.score)
        return vocab


def _token_entry(idx: int, meta: dict) -> dict:
    return {
        "id": idx,
        "content": meta["content"],
        "single_word": bool(meta.get("single_word", False)),
        "lstrip": bool(meta.get("lstrip", False)),
        "rstrip": bool(meta.get("rstrip", False)),
        "normalized": bool(meta.get("normalized", False)),
        "special": bool(meta.get("special", False)),
    }


def patch_fast_tokenizer(path: Path, model_dir: Path) -> None:
    """Align fast JSON with HF Marian specials / added letters.

    SpmConverter alone leaves `<unk>` unmarked and invents a bogus `<s>`, so
    Candle `decode(..., skip_special_tokens=true)` prints the literal `<unk>`.
    """
    cfg_path = model_dir / "tokenizer_config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"missing {cfg_path} — needed to patch specials")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    decoder = dict(cfg.get("added_tokens_decoder") or {})

    # Letters from training resize (source of truth if config is stale).
    letters_path = model_dir / "added_tokens.json"
    if letters_path.is_file():
        for content, idx in json.loads(letters_path.read_text(encoding="utf-8")).items():
            key = str(int(idx))
            if key not in decoder:
                decoder[key] = {
                    "content": content,
                    "lstrip": False,
                    "normalized": True,
                    "rstrip": False,
                    "single_word": False,
                    "special": False,
                }

    # Hard requirements for Marian.
    defaults = {
        "0": {
            "content": "</s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "1": {
            "content": "<unk>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
        "59415": {
            "content": "<pad>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        },
    }
    for key, meta in defaults.items():
        decoder[key] = meta

    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = data.get("model", {}).get("vocab")
    if not isinstance(vocab, list):
        raise SystemExit(f"{path}: expected model.vocab list, got {type(vocab)}")

    added: list[dict] = []
    for key, meta in sorted(decoder.items(), key=lambda item: int(item[0])):
        idx = int(key)
        content = meta["content"]
        while len(vocab) <= idx:
            vocab.append(["<NIL>", -100.0])
        score = vocab[idx][1] if isinstance(vocab[idx], (list, tuple)) else -100.0
        vocab[idx] = [content, score]
        added.append(_token_entry(idx, meta))

    data["added_tokens"] = added
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    specials = [t["content"] for t in added if t["special"]]
    print(f"patched {path.name}: specials={specials} added={len(added)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Helsinki/Marian model folder (source.spm, target.spm, vocab.json)",
    )
    args = parser.parse_args()
    model_dir: Path = args.model_dir.resolve()
    if not (model_dir / "vocab.json").is_file():
        raise SystemExit(f"vocab.json missing in {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    enc = MarianConverter(tokenizer, index=0).converted()
    enc_path = model_dir / "tokenizer-enc.json"
    enc.save(str(enc_path))
    print(f"wrote {enc_path}")
    patch_fast_tokenizer(enc_path, model_dir)

    dec = MarianConverter(tokenizer, index=1).converted()
    dec_path = model_dir / "tokenizer-dec.json"
    dec.save(str(dec_path))
    print(f"wrote {dec_path}")
    patch_fast_tokenizer(dec_path, model_dir)


if __name__ == "__main__":
    main()
