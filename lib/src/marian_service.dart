import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import 'rust/api/marian.dart';
import 'rust/frb_generated.dart';

export 'rust/api/marian.dart' show MarianTranslator, TranslationConfig;
export 'rust/api/simple.dart' show greet;
export 'rust/frb_generated.dart' show RustLib;

/// Street-mode decode defaults for on-device Candle.
///
/// HuggingFace `demo.py --label street` uses `num_beams=4`. Beam search now
/// keeps a per-beam KV cache (incremental decode), but 4×48 is still heavy on
/// phone CPUs — beams=2 / 24 new tokens stays interactive while retaining the
/// street length penalty.
const TranslationConfig kStreetTranslationConfig = TranslationConfig(
  numBeams: 2,
  maxNewTokens: 24,
  lengthPenalty: 1.2,
  noRepeatNgramSize: 3,
);

/// High-level helper: init FRB, materialize Flutter assets to disk, load Candle.
class MarianService {
  MarianService._(this._translator, this.modelDir);

  final MarianTranslator _translator;

  /// Absolute filesystem path of the materialized model folder.
  final String modelDir;

  static bool _rustReady = false;

  /// Initialize the Rust bridge (safe to call more than once).
  static Future<void> initRust({ExternalLibrary? externalLibrary}) async {
    if (_rustReady) return;
    await RustLib.init(externalLibrary: externalLibrary);
    _rustReady = true;
  }

  /// Copy a Flutter asset folder into app documents and load the model.
  ///
  /// [assetFolder] is the Flutter asset prefix, e.g. `assets/fr-pul`.
  /// Required files inside that folder:
  /// - `config.json`
  /// - `model.safetensors`
  /// - `tokenizer-enc.json`
  /// - `tokenizer-dec.json`
  static Future<MarianService> loadFromAssets({
    required String assetFolder,
    String? cacheSubdir,
    List<String> filenames = const [
      'config.json',
      'model.safetensors',
      'tokenizer-enc.json',
      'tokenizer-dec.json',
    ],
  }) async {
    await initRust();
    final dir = await materializeAssets(
      assetFolder: assetFolder,
      cacheSubdir: cacheSubdir ?? p.basename(assetFolder),
      filenames: filenames,
    );
    final translator = await MarianTranslator.loadFromDir(modelDir: dir.path);
    return MarianService._(translator, dir.path);
  }

  /// Load from an already-on-disk model directory.
  static Future<MarianService> loadFromDirectory(String modelDir) async {
    await initRust();
    final translator =
        await MarianTranslator.loadFromDir(modelDir: modelDir);
    return MarianService._(translator, modelDir);
  }

  Future<bool> get isReady => _translator.isReady();

  Future<String> translate(
    String text, {
    TranslationConfig config = kStreetTranslationConfig,
  }) {
    return _translator.translate(text: text, config: config);
  }

  /// Copy listed assets under [assetFolder] into a writable directory.
  ///
  /// Candle / `tokenizers` need real filesystem paths (mmap), not asset URIs.
  static Future<Directory> materializeAssets({
    required String assetFolder,
    required String cacheSubdir,
    required List<String> filenames,
  }) async {
    final docs = await getApplicationDocumentsDirectory();
    final String baseDir = Platform.isWindows ? 'Malinali_do_not_delete/marian_models' : 'marian_models';
    final out = Directory(p.join(docs.path, baseDir, cacheSubdir));
    if (!await out.exists()) {
      await out.create(recursive: true);
    }

    final prefix = assetFolder.endsWith('/')
        ? assetFolder
        : '$assetFolder/';

    for (final name in filenames) {
      final dest = File(p.join(out.path, name));
      final stamp = File(p.join(out.path, '$name.fp'));
      final data = await rootBundle.load('$prefix$name');
      final bytes =
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      final fp = _assetFingerprint(bytes);
      if (await dest.exists() &&
          await stamp.exists() &&
          (await stamp.readAsString()).trim() == fp) {
        continue;
      }
      await dest.writeAsBytes(bytes, flush: true);
      await stamp.writeAsString(fp, flush: true);
    }
    return out;
  }

  /// Cheap stable fingerprint (length + sampled bytes). Fine-tunes keep the
  /// same safetensors *size*, so a length-only cache check would skip updates.
  static String _assetFingerprint(List<int> bytes) {
    var mix = bytes.length;
    final step = (bytes.length ~/ 2048).clamp(1, bytes.length);
    for (var i = 0; i < bytes.length; i += step) {
      mix = 0x1fffffff & (mix + bytes[i]);
      mix = 0x1fffffff & (mix + ((0x0007ffff & mix) << 10));
      mix ^= mix >> 6;
    }
    if (bytes.isNotEmpty) {
      mix = 0x1fffffff & (mix + bytes.first + bytes.last);
    }
    return '${bytes.length}:$mix';
  }
}
