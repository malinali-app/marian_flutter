import 'package:flutter/material.dart';
import 'package:marian_flutter/marian_flutter.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await MarianService.initRust();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final _controller = TextEditingController(text: 'Bonjour Pierre');
  String _status = 'Bridge ready. Push a model to documents/marian_models/fr-pul.';
  String _output = '';
  MarianService? _marian;
  bool _busy = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    setState(() {
      _busy = true;
      _status = 'Loading…';
    });
    try {
      final docs = await getApplicationDocumentsDirectory();
      final modelDir = p.join(docs.path, 'marian_models', 'fr-pul');
      final service = await MarianService.loadFromDirectory(modelDir);
      setState(() {
        _marian = service;
        _status = 'Loaded $modelDir';
      });
    } catch (e) {
      setState(() => _status = 'Load failed: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _translate() async {
    final marian = _marian;
    if (marian == null) {
      setState(() => _status = 'Load a model first.');
      return;
    }
    setState(() => _busy = true);
    try {
      final out = await marian.translate(_controller.text);
      setState(() {
        _output = out;
        _status = 'OK';
      });
    } catch (e) {
      setState(() => _status = 'Translate failed: $e');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('marian_flutter')),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('Rust greet smoke: ${greet(name: 'Tom')}'),
              const SizedBox(height: 12),
              TextField(
                controller: _controller,
                decoration: const InputDecoration(
                  labelText: 'French',
                  border: OutlineInputBorder(),
                ),
                minLines: 2,
                maxLines: 4,
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: FilledButton(
                      onPressed: _busy ? null : _loadModel,
                      child: const Text('Load model'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: FilledButton(
                      onPressed: _busy ? null : _translate,
                      child: const Text('Translate'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(_status, style: Theme.of(context).textTheme.bodySmall),
              const SizedBox(height: 12),
              Expanded(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.black26),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: SelectableText(
                      _output.isEmpty ? 'Pulaar output…' : _output,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
