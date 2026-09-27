# marian_flutter

On-device **MarianMT** for Flutter, powered by [Candle](https://github.com/huggingface/candle)
and [flutter_rust_bridge](https://cjycode.com/flutter_rust_bridge/) (v2 + Cargokit).

**Platform focus:** Android first (`arm64-v8a` / `armeabi-v7a` via Cargokit).

## Why not ONNX-only?

This package keeps HuggingFace SentencePiece tokenization (`tokenizers` crate)
and beam search (`num_beams=4`, length penalty, no-repeat n-gram) in Rust so
mobile output matches training/`demo.py` quality.

## Model folder layout

Ship (or first-run download) a directory containing:

| File | Role |
|------|------|
| `config.json` | Marian config (deserialized by Candle) |
| `model.safetensors` | Weights |
| `tokenizer-enc.json` | Source (French) fast tokenizer |
| `tokenizer-dec.json` | Target (Pulaar) fast tokenizer |

Convert Helsinki/Marian SPM files (prefer WSL Python — Windows pip may hit a broken NVIDIA index):

```bash
# from Windows
wsl bash scripts/prepare_test_fixture_wsl.sh
# or: powershell -File scripts/prepare_test_fixture.ps1
# then:
flutter test test/marian_service_test.dart
```

Manual convert only:

```bash
wsl bash -lc '~/.venvs/marian-convert/bin/python scripts/convert_tokenizer.py /mnt/c/.../models/fr-pul'
```

If the checkpoint is still `pytorch_model.bin`, convert to safetensors before shipping
(e.g. `transformers` `save_pretrained` / `safetensors` export).

## App usage

```yaml
dependencies:
  marian_flutter: ^0.1.0

flutter:
  assets:
    - assets/fr-pul/
```

```dart
import 'package:marian_flutter/marian_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await MarianService.initRust();
  runApp(const MyApp());
}

// later:
final marian = await MarianService.loadFromAssets(
  assetFolder: 'assets/fr-pul',
);
final pulaar = await marian.translate('Bonjour Pierre');
```

Decode defaults match street mode:

- `numBeams: 4`
- `maxNewTokens: 48`
- `lengthPenalty: 1.2`
- `noRepeatNgramSize: 3`

## Develop / regenerate bindings

```bash
# After editing rust/src/api/**
cd marian_flutter
flutter_rust_bridge_codegen generate
```

## Android build

Cargokit builds the Rust `cdylib` during the Flutter Android Gradle build.
One-time host setup:

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android
# Android NDK via Android Studio / sdkmanager
```

Then:

```bash
cd example
flutter run
```

## Package layout

```
marian_flutter/
  lib/                 # Dart API (MarianService + generated FRB)
  rust/                # Candle MarianMT engine
  android/             # ffiPlugin + Cargokit Gradle hook
  cargokit/            # FRB Android/iOS/desktop build helpers
  scripts/             # tokenizer conversion
  example/             # smoke-test Flutter app
```
