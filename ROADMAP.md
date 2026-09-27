# marian_flutter roadmap

## Android performance (post KV-cache beam search)

Per-beam incremental decode is done (same hypotheses as full re-decode; ~1.5× on desktop at beams=4 / 48 tokens). Remaining ideas, ordered by bang-for-buck:

### Low-hanging (same model)

1. **Adaptive `maxNewTokens`**  
   Cap by input length, e.g. `min(24, src_tokens * 2 + 8)`. Cuts worst-case stalls when the model rambles on short inputs.

2. **Greedy for interactive UI (`numBeams: 1`)**  
   Keep beam=2 for quality / demos; use greedy on the translate button. Often ~2× vs beam=2 with small quality loss on street phrases. Biggest remaining knob without retraining.

3. **Rust release profile knobs** (`rust/Cargo.toml`)  
   ```toml
   [profile.release]
   lto = true          # or "thin"
   codegen-units = 1
   ```
   Often a few–15% on CPU kernels after a full Android rebuild.

4. **Ship `arm64-v8a` only** (if dropping 32-bit is acceptable)  
   Smaller APK, one optimized binary; every modern Samsung is arm64.

5. **Keep translate off the UI isolate**  
   Already async via FRB; avoid reloading the model per tap. Boot warmup stays.

### Medium effort (still no retrain)

6. **Cross-attn encoder K/V cache**  
   Candle Marian re-projects full `encoder_xs` every decode step. Caching once per translation is the next algorithmic win — needs a small wrapper/fork around Candle attention.

7. **Faster vocab step**  
   Each step does `log_softmax` + `top_k` over ~50k logits. Restricting log-softmax to a candidate set (e.g. top-100 then beam) saves CPU; slight score change.

### Larger bets

| Idea | Notes |
|------|--------|
| FP16 / int8 weights | Export + Candle path + quality check |
| Smaller / distilled Marian | Retrain in `fula-marian-training` |
| GPU / NPU | Not free with Candle-on-Android today |

**Suggested next slice on device:** greedy for the button + adaptive max tokens + release LTO.

## Packaging / ecosystem

- [ ] Publish `marian_flutter` 0.1.0 to pub.dev (dry-run already clean)
- [ ] Push GitHub remote `malinali-app/marian_flutter`
- [ ] Point `malinali-app` at `marian_flutter: ^0.1.0` once published (path dep until then)

## Why Rust + flutter_rust_bridge (not `marian_dart`)

`marian_dart` was an early CLI experiment: raw `dart:ffi` into a hand-built Windows `marian_candle.dll` with a tiny C ABI (`translate_text`). It was not a Flutter plugin (no Cargokit, no Android NDK build, hardcoded desktop DLL path).

`marian_flutter` exists because mobile needs:

- **Cross-compile** of the Candle engine into `.so` via Cargokit for `arm64-v8a` / `armeabi-v7a`
- **Typed async API** (`MarianService.translate`) without hand-maintained FFI stubs
- **Tokenizers + beam search in Rust** so on-device output matches `demo.py` / training (see README “Why not ONNX-only?”)
- **Asset → mmap paths** for safetensors on device

FRB adds a thin marshaling layer (strings / structs across the isolate boundary). That cost is negligible next to Marian forward passes (tens–thousands of ms). You do *not* lose meaningful translate latency to the bridge; you gain maintainable Android packaging and an API that matches training quality.
