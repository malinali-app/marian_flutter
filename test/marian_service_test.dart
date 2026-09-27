import 'dart:io';

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:marian_flutter/marian_flutter.dart';
import 'package:path/path.dart' as p;

/// Host FFI smoke test against the converted fr-pul Candle folder.
///
/// Prepare once:
///   powershell -File scripts/prepare_test_fixture.ps1
/// (uses WSL Python for tokenizer conversion)
void main() {
  test('street translation config defaults', () {
    expect(kStreetTranslationConfig.numBeams, 4);
    expect(kStreetTranslationConfig.maxNewTokens, 48);
    expect(kStreetTranslationConfig.lengthPenalty, 1.2);
    expect(kStreetTranslationConfig.noRepeatNgramSize, 3);
  });

  test('load fr-pul fixture and translate a short French sentence', () async {
    final fixture = _fixtureDir();
    for (final name in const [
      'config.json',
      'model.safetensors',
      'tokenizer-enc.json',
      'tokenizer-dec.json',
    ]) {
      final f = File(p.join(fixture.path, name));
      expect(f.existsSync(), isTrue, reason: 'missing ${f.path}');
    }

    final lib = _hostLibrary();
    expect(lib.existsSync(), isTrue, reason: 'build rust release first: ${lib.path}');

    await MarianService.initRust(
      externalLibrary: ExternalLibrary.open(lib.path),
    );
    addTearDown(RustLib.dispose);

    final marian = await MarianService.loadFromDirectory(fixture.path);
    expect(await marian.isReady, isTrue);

    final out = await marian.translate(
      'Bonjour',
      config: const TranslationConfig(
        numBeams: 2,
        maxNewTokens: 16,
        lengthPenalty: 1.2,
        noRepeatNgramSize: 3,
      ),
    );

    // ignore: avoid_print
    print('translate("Bonjour") => "$out"');
    expect(out.trim(), isNotEmpty);
    expect(out.toLowerCase(), isNot(contains('bonjour')));
  }, timeout: const Timeout(Duration(minutes: 5)));
}

Directory _fixtureDir() {
  return Directory(p.join(Directory.current.path, 'test', 'fixtures', 'fr-pul'));
}

File _hostLibrary() {
  final release = p.join(Directory.current.path, 'rust', 'target', 'release');
  if (Platform.isWindows) {
    return File(p.join(release, 'marian_flutter.dll'));
  }
  if (Platform.isMacOS) {
    return File(p.join(release, 'libmarian_flutter.dylib'));
  }
  return File(p.join(release, 'libmarian_flutter.so'));
}
