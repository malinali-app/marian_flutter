use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result as AnyResult};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::marian::{Config, MTModel};
use tokenizers::Tokenizer;

/// Decode hyperparameters matching Malinali street-mode / HuggingFace Marian defaults.
#[derive(Clone, Debug)]
pub struct TranslationConfig {
    pub num_beams: u32,
    pub max_new_tokens: u32,
    pub length_penalty: f64,
    pub no_repeat_ngram_size: u32,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        // Interactive on-device defaults (see Dart `kStreetTranslationConfig`).
        Self {
            num_beams: 2,
            max_new_tokens: 24,
            length_penalty: 1.2,
            no_repeat_ngram_size: 3,
        }
    }
}

struct TranslatorInner {
    model: MTModel,
    config: Config,
    tokenizer_enc: Tokenizer,
    tokenizer_dec: Tokenizer,
    device: Device,
}

#[derive(Clone)]
struct Beam {
    tokens: Vec<u32>,
    score: f32,
    finished: bool,
}

/// On-device MarianMT translator backed by Candle.
///
/// Expected directory layout for [`MarianTranslator::load_from_dir`]:
/// - `config.json`
/// - `model.safetensors`
/// - `tokenizer-enc.json` (source / French)
/// - `tokenizer-dec.json` (target / Pulaar)
///
/// Convert Helsinki/Marian SentencePiece tokenizers with
/// `scripts/convert_tokenizer.py`.
pub struct MarianTranslator {
    inner: Mutex<TranslatorInner>,
}

impl MarianTranslator {
    /// Load from explicit file paths.
    pub fn load(
        model_path: String,
        config_path: String,
        tokenizer_enc_path: String,
        tokenizer_dec_path: String,
    ) -> Result<Self, String> {
        load_inner(
            PathBuf::from(model_path),
            PathBuf::from(config_path),
            PathBuf::from(tokenizer_enc_path),
            PathBuf::from(tokenizer_dec_path),
        )
        .map_err(|e| format!("{e:#}"))
        .map(|inner| Self {
            inner: Mutex::new(inner),
        })
    }

    /// Load from a Helsinki-style model folder.
    pub fn load_from_dir(model_dir: String) -> Result<Self, String> {
        let dir = PathBuf::from(&model_dir);
        let model_path = first_existing(&dir, &["model.safetensors"])
            .ok_or_else(|| format!("model.safetensors not found in {model_dir}"))?;
        let config_path = first_existing(&dir, &["config.json"])
            .ok_or_else(|| format!("config.json not found in {model_dir}"))?;
        
        let tokenizer_enc_path = first_existing(
            &dir,
            &[
                "tokenizer.json",
                "tokenizer-enc.json",
                "tokenizer_src.json",
                "source.json",
            ],
        )
        .ok_or_else(|| {
            format!("No compatible JSON tokenizer found in {model_dir}. HuggingFace Marian models require a .json tokenizer (like the one provided by Xenova).")
        })?;

        let tokenizer_dec_path = first_existing(
            &dir,
            &[
                "tokenizer.json",
                "tokenizer-dec.json",
                "tokenizer_tgt.json",
                "target.json",
            ],
        )
        .unwrap_or_else(|| tokenizer_enc_path.clone());

        Self::load(
            path_string(&model_path),
            path_string(&config_path),
            path_string(&tokenizer_enc_path),
            path_string(&tokenizer_dec_path),
        )
    }

    pub fn is_ready(&self) -> bool {
        self.inner.lock().is_ok()
    }

    /// Translate `text` with optional decode config (defaults to street-mode).
    pub fn translate(
        &self,
        text: String,
        config: Option<TranslationConfig>,
    ) -> Result<String, String> {
        let cfg = config.unwrap_or_default();
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "translator lock poisoned".to_string())?;
        translate_inner(&mut inner, &text, &cfg).map_err(|e| format!("{e:#}"))
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn first_existing(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names
        .iter()
        .map(|name| dir.join(name))
        .find(|p| p.is_file())
}

fn load_inner(
    model_path: PathBuf,
    config_path: PathBuf,
    tokenizer_enc_path: PathBuf,
    tokenizer_dec_path: PathBuf,
) -> AnyResult<TranslatorInner> {
    let device = Device::Cpu;
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;
    let config: Config = serde_json::from_str(&config_str)
        .with_context(|| format!("parse config {}", config_path.display()))?;

    let tensors = candle_core::safetensors::load(&model_path, &device)
        .with_context(|| format!("load safetensors {}", model_path.display()))?;

    // Helsinki models on HF sometimes name the shared weights incorrectly 
    // when converted to safetensors. MTModel expects "model.shared.weight".
    // We provide keys both with and without the "model." prefix to be safe.
    let mut mapped = std::collections::HashMap::new();
    for (k, v) in tensors {
        mapped.insert(k.clone(), v.clone());
        if k.starts_with("model.") {
            mapped.insert(k.strip_prefix("model.").unwrap().to_string(), v.clone());
        } else {
            mapped.insert(format!("model.{}", k), v.clone());
        }
    }

    // Shared embeddings alias
    if !mapped.contains_key("model.shared.weight") {
        if let Some(v) = mapped.get("model.decoder.embed_tokens.weight").cloned() {
            mapped.insert("model.shared.weight".to_string(), v.clone());
            mapped.insert("shared.weight".to_string(), v);
        } else if let Some(v) = mapped.get("model.encoder.embed_tokens.weight").cloned() {
            mapped.insert("model.shared.weight".to_string(), v.clone());
            mapped.insert("shared.weight".to_string(), v);
        }
    }

    let vb = VarBuilder::from_tensors(mapped, DType::F32, &device);

    let model = MTModel::new(&config, vb).map_err(|e| {
        anyhow!("MTModel::new failed: {e}")
    })?;

    let tokenizer_enc = Tokenizer::from_file(&tokenizer_enc_path)
        .map_err(|e| anyhow!("load encoder tokenizer {}: {e}", tokenizer_enc_path.display()))?;
    let tokenizer_dec = Tokenizer::from_file(&tokenizer_dec_path)
        .map_err(|e| anyhow!("load decoder tokenizer {}: {e}", tokenizer_dec_path.display()))?;

    Ok(TranslatorInner {
        model,
        config,
        tokenizer_enc,
        tokenizer_dec,
        device,
    })
}

fn translate_inner(
    inner: &mut TranslatorInner,
    text: &str,
    cfg: &TranslationConfig,
) -> AnyResult<String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(String::new());
    }

    let mut tokens = inner
        .tokenizer_enc
        .encode(text, true)
        .map_err(|e| anyhow!("tokenize: {e}"))?
        .get_ids()
        .to_vec();
    tokens.push(inner.config.eos_token_id);

    inner.model.reset_kv_cache();
    let encoder_input = Tensor::new(tokens.as_slice(), &inner.device)?.unsqueeze(0)?;
    let encoder_xs = inner.model.encoder().forward(&encoder_input, 0)?;

    let generated = if cfg.num_beams <= 1 {
        greedy_decode(inner, &encoder_xs, cfg)?
    } else {
        beam_search(inner, &encoder_xs, cfg)?
    };

    let mut decode_ids: Vec<u32> = generated
        .into_iter()
        .filter(|id| *id != inner.config.decoder_start_token_id)
        .collect();
    if let Some(pos) = decode_ids
        .iter()
        .position(|id| *id == inner.config.eos_token_id || *id == inner.config.forced_eos_token_id)
    {
        decode_ids.truncate(pos);
    }

    inner
        .tokenizer_dec
        .decode(&decode_ids, true)
        .map_err(|e| anyhow!("detokenize: {e}"))
}

fn greedy_decode(
    inner: &mut TranslatorInner,
    encoder_xs: &Tensor,
    cfg: &TranslationConfig,
) -> AnyResult<Vec<u32>> {
    let mut token_ids = vec![inner.config.decoder_start_token_id];
    let max_new = cfg.max_new_tokens as usize;

    for index in 0..max_new {
        let context_size = if index >= 1 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &inner.device)?.unsqueeze(0)?;
        let logits = inner.model.decode(&input_ids, encoder_xs, start_pos)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let mut scores = logits.to_vec1::<f32>()?;
        apply_no_repeat_ngram(&token_ids, &mut scores, cfg.no_repeat_ngram_size as usize);
        let next = argmax(&scores) as u32;
        token_ids.push(next);
        if next == inner.config.eos_token_id || next == inner.config.forced_eos_token_id {
            break;
        }
    }
    Ok(token_ids)
}

fn beam_search(
    inner: &mut TranslatorInner,
    encoder_xs: &Tensor,
    cfg: &TranslationConfig,
) -> AnyResult<Vec<u32>> {
    let num_beams = cfg.num_beams as usize;
    let max_new = cfg.max_new_tokens as usize;
    let length_penalty = cfg.length_penalty;
    let ngram = cfg.no_repeat_ngram_size as usize;

    let mut beams = vec![Beam {
        tokens: vec![inner.config.decoder_start_token_id],
        score: 0.0,
        finished: false,
    }];
    let mut finished: Vec<Beam> = Vec::new();

    for _step in 0..max_new {
        let mut candidates: Vec<Beam> = Vec::new();

        for beam in &beams {
            if beam.finished {
                candidates.push(beam.clone());
                continue;
            }

            // Full decoder pass per candidate (no per-beam KV cache). Fine for short phrases.
            inner.model.reset_kv_cache();
            let input_ids = Tensor::new(beam.tokens.as_slice(), &inner.device)?.unsqueeze(0)?;
            let logits = inner.model.decode(&input_ids, encoder_xs, 0)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let mut scores = logits.to_vec1::<f32>()?;
            apply_no_repeat_ngram(&beam.tokens, &mut scores, ngram);
            log_softmax_inplace(&mut scores);

            for idx in top_k_indices(&scores, num_beams) {
                let token = idx as u32;
                let mut next_tokens = beam.tokens.clone();
                next_tokens.push(token);
                let done = token == inner.config.eos_token_id
                    || token == inner.config.forced_eos_token_id;
                candidates.push(Beam {
                    tokens: next_tokens,
                    score: beam.score + scores[idx],
                    finished: done,
                });
            }
        }

        candidates.sort_by(|a, b| {
            normalized_score(b.score, b.tokens.len(), length_penalty)
                .partial_cmp(&normalized_score(a.score, a.tokens.len(), length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        beams.clear();
        for cand in candidates {
            if cand.finished {
                finished.push(cand);
            } else if beams.len() < num_beams {
                beams.push(cand);
            }
            if beams.len() >= num_beams && finished.len() >= num_beams {
                break;
            }
        }

        if beams.is_empty() {
            break;
        }
    }

    finished.extend(beams);
    finished.sort_by(|a, b| {
        normalized_score(b.score, b.tokens.len(), length_penalty)
            .partial_cmp(&normalized_score(a.score, a.tokens.len(), length_penalty))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    finished
        .into_iter()
        .next()
        .map(|b| b.tokens)
        .ok_or_else(|| anyhow!("beam search produced no hypotheses"))
}

fn normalized_score(score: f32, len: usize, length_penalty: f64) -> f32 {
    let len = len.max(1) as f64;
    let norm = ((5.0 + len) / 6.0).powf(length_penalty);
    (score as f64 / norm) as f32
}

fn apply_no_repeat_ngram(tokens: &[u32], scores: &mut [f32], ngram_size: usize) {
    if ngram_size == 0 || tokens.len() < ngram_size {
        return;
    }
    let prefix_len = ngram_size - 1;
    let prefix = &tokens[tokens.len() - prefix_len..];
    for start in 0..=tokens.len().saturating_sub(ngram_size) {
        if &tokens[start..start + prefix_len] == prefix {
            let banned = tokens[start + prefix_len] as usize;
            if banned < scores.len() {
                scores[banned] = f32::NEG_INFINITY;
            }
        }
    }
}

fn log_softmax_inplace(scores: &mut [f32]) {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    let log_sum = sum.ln();
    for s in scores.iter_mut() {
        *s = s.ln() - log_sum;
    }
}

fn argmax(scores: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in scores.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len());
    if k == 0 {
        return Vec::new();
    }
    // Marian vocab is ~50k: full sort per beam step is far too slow on CPU.
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}
