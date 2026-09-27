# Changelog

## 0.1.0

- Initial Flutter FFI plugin with flutter_rust_bridge 2.13 + Cargokit.
- Candle MarianMT engine (`load` / `load_from_dir` / `translate`).
- Beam search decode defaults: beams=4, length_penalty=1.2, no_repeat_ngram=3.
- Dart `MarianService` with Flutter asset materialization for mmap paths.
- Android-first packaging (`minSdk 24`).
