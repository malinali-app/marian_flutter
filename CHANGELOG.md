# Changelog

## 0.2.0

- better readme
- slight performance increase for quicker inference refining beam-search 
  - Performance (desktop release CPU) ~1.13×

## 0.1.0

- Initial Flutter FFI plugin with flutter_rust_bridge 2.13 + Cargokit.
- Candle MarianMT engine (`load` / `load_from_dir` / `translate`).
- Beam search decode defaults: beams=4, length_penalty=1.2, no_repeat_ngram=3.
- Dart `MarianService` with Flutter asset materialization for mmap paths.
- Android-first packaging (`minSdk 24`).
- Per-beam incremental KV-cache decode (same hypotheses as full re-decode, ~1.5× faster at beams=4 / 48 tokens on desktop CPU).
