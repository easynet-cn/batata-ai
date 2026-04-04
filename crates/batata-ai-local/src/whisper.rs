use std::path::Path;

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use tokenizers::Tokenizer;
use tracing::info;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::multimodal::{AudioData, TranscriptionResult};

/// Whisper model for speech-to-text transcription
pub struct WhisperModel {
    model: m::model::Whisper,
    config: Config,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,
    device: Device,
}

/// Whisper model size variants
#[derive(Debug, Clone, Copy)]
pub enum WhisperSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl WhisperSize {
    fn repo_id(&self) -> &str {
        match self {
            Self::Tiny => "openai/whisper-tiny",
            Self::Base => "openai/whisper-base",
            Self::Small => "openai/whisper-small",
            Self::Medium => "openai/whisper-medium",
            Self::Large => "openai/whisper-large-v3-turbo",
        }
    }

    fn config_filename(&self) -> &str {
        "config.json"
    }
}

impl WhisperModel {
    /// Download and load a Whisper model from HuggingFace Hub
    pub fn download_and_load(size: WhisperSize, device: &Device) -> Result<Self> {
        let repo_id = size.repo_id();
        info!("downloading Whisper model from {repo_id}...");

        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo
            .get(size.config_filename())
            .map_err(|e| BatataError::Inference(format!("config download failed: {e}")))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| BatataError::Inference(format!("tokenizer download failed: {e}")))?;
        let model_path = repo
            .get("model.safetensors")
            .map_err(|e| BatataError::Inference(format!("model download failed: {e}")))?;

        Self::load(&model_path, &config_path, &tokenizer_path, device)
    }

    /// Load Whisper from local files
    pub fn load(
        model_path: &Path,
        config_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        info!("loading Whisper model from {}", model_path.display());

        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| BatataError::Inference(format!("invalid config: {e}")))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_path.to_path_buf()],
                m::DTYPE,
                device,
            )
            .map_err(|e| BatataError::Inference(format!("failed to load weights: {e}")))?
        };

        let model = m::model::Whisper::load(&vb, config.clone())
            .map_err(|e| BatataError::Inference(format!("failed to build model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;

        let mel_filters = vec![0f32; config.num_mel_bins * m::N_FFT];

        info!("Whisper model loaded successfully");

        Ok(Self {
            model,
            config,
            tokenizer,
            mel_filters,
            device: device.clone(),
        })
    }

    /// Transcribe audio data
    pub fn transcribe(&mut self, audio: AudioData) -> Result<TranscriptionResult> {
        let samples = if audio.sample_rate != m::SAMPLE_RATE as u32 {
            // Simple nearest-neighbor resampling
            let ratio = m::SAMPLE_RATE as f32 / audio.sample_rate as f32;
            let new_len = (audio.samples.len() as f32 * ratio) as usize;
            (0..new_len)
                .map(|i| {
                    let src_idx = (i as f32 / ratio) as usize;
                    audio.samples[src_idx.min(audio.samples.len() - 1)]
                })
                .collect::<Vec<_>>()
        } else {
            audio.samples
        };

        let mel = audio::pcm_to_mel(&self.config, &samples, &self.mel_filters);
        let mel_len = mel.len();
        let n_mels = self.config.num_mel_bins;

        let mel = Tensor::from_vec(mel, (1, n_mels, mel_len / n_mels), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        let encoder_output = self
            .model
            .encoder
            .forward(&mel, true)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        // Decode tokens greedily
        let sot_token = self
            .tokenizer
            .token_to_id("<|startoftranscript|>")
            .unwrap_or(50258);
        let transcribe_token = self
            .tokenizer
            .token_to_id("<|transcribe|>")
            .unwrap_or(50359);
        let eot_token = self
            .tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(50257);
        let no_timestamps_token = self
            .tokenizer
            .token_to_id("<|notimestamps|>")
            .unwrap_or(50363);

        let mut tokens = vec![sot_token, transcribe_token, no_timestamps_token];
        let max_tokens = 224;

        for _ in 0..max_tokens {
            let token_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| BatataError::Inference(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let logits = self
                .model
                .decoder
                .forward(&token_tensor, &encoder_output, tokens.len() == 3)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let last_pos = tokens.len() - 1;
            let logits_slice = logits
                .i((.., last_pos..last_pos + 1, ..))
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let logits = self
                .model
                .decoder
                .final_linear(&logits_slice)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let logits = logits
                .squeeze(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?
                .squeeze(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let next_token = logits
                .argmax(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?
                .to_scalar::<u32>()
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            if next_token == eot_token {
                break;
            }

            tokens.push(next_token);
        }

        // Skip special tokens at the start
        let text_tokens: Vec<u32> = tokens
            .into_iter()
            .filter(|&t| t < sot_token)
            .collect();

        let text = self
            .tokenizer
            .decode(&text_tokens, true)
            .map_err(|e| BatataError::Inference(format!("decode failed: {e}")))?;

        self.model.reset_kv_cache();

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            segments: vec![],
            language: None,
        })
    }

    /// Transcribe from a WAV file
    pub fn transcribe_file(&mut self, path: &Path) -> Result<TranscriptionResult> {
        let samples = read_wav_pcm(path)?;
        self.transcribe(AudioData {
            samples,
            sample_rate: m::SAMPLE_RATE as u32,
        })
    }
}

/// Read a WAV file and return f32 PCM samples at 16kHz mono
fn read_wav_pcm(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;

    // Minimal WAV parser — expects PCM format
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(BatataError::Inference("not a valid WAV file".into()));
    }

    // Find "data" chunk
    let mut offset = 12;
    while offset + 8 < bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size =
            u32::from_le_bytes([bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7]])
                as usize;

        if chunk_id == b"fmt " {
            let _audio_format = u16::from_le_bytes([bytes[offset + 8], bytes[offset + 9]]);
            let channels = u16::from_le_bytes([bytes[offset + 10], bytes[offset + 11]]);
            let sample_rate =
                u32::from_le_bytes([bytes[offset + 12], bytes[offset + 13], bytes[offset + 14], bytes[offset + 15]]);
            let bits_per_sample = u16::from_le_bytes([bytes[offset + 22], bytes[offset + 23]]);

            if bits_per_sample != 16 {
                return Err(BatataError::Inference(format!(
                    "unsupported bits per sample: {bits_per_sample}, expected 16"
                )));
            }

            offset += 8 + chunk_size;

            // Find data chunk
            while offset + 8 < bytes.len() {
                let cid = &bytes[offset..offset + 4];
                let csz = u32::from_le_bytes([
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]) as usize;

                if cid == b"data" {
                    let data = &bytes[offset + 8..offset + 8 + csz.min(bytes.len() - offset - 8)];
                    let mut samples = Vec::with_capacity(data.len() / 2);

                    for chunk in data.chunks_exact(2) {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                        samples.push(sample);
                    }

                    // Convert to mono if stereo
                    if channels == 2 {
                        samples = samples
                            .chunks(2)
                            .map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0)
                            .collect();
                    }

                    // Resample to 16kHz if needed
                    if sample_rate != m::SAMPLE_RATE as u32 {
                        let ratio = m::SAMPLE_RATE as f32 / sample_rate as f32;
                        let new_len = (samples.len() as f32 * ratio) as usize;
                        samples = (0..new_len)
                            .map(|i| {
                                let idx = (i as f32 / ratio) as usize;
                                samples[idx.min(samples.len() - 1)]
                            })
                            .collect();
                    }

                    return Ok(samples);
                }

                offset += 8 + csz;
            }
        }

        offset += 8 + chunk_size;
    }

    Err(BatataError::Inference("failed to parse WAV data".into()))
}

/// Download Whisper model files to a local directory
pub fn download_whisper_to(
    size: WhisperSize,
    target_dir: &Path,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| BatataError::Inference(e.to_string()))?;
    let repo = api.model(size.repo_id().to_string());

    let config_cached = repo
        .get("config.json")
        .map_err(|e| BatataError::Inference(format!("download failed: {e}")))?;
    let tokenizer_cached = repo
        .get("tokenizer.json")
        .map_err(|e| BatataError::Inference(format!("download failed: {e}")))?;
    let model_cached = repo
        .get("model.safetensors")
        .map_err(|e| BatataError::Inference(format!("download failed: {e}")))?;

    std::fs::create_dir_all(target_dir)?;

    let config_target = target_dir.join("config.json");
    let tokenizer_target = target_dir.join("tokenizer.json");
    let model_target = target_dir.join("model.safetensors");

    if !config_target.exists() {
        std::fs::copy(&config_cached, &config_target)?;
    }
    if !tokenizer_target.exists() {
        std::fs::copy(&tokenizer_cached, &tokenizer_target)?;
    }
    if !model_target.exists() {
        info!("copying whisper model to {}", model_target.display());
        std::fs::copy(&model_cached, &model_target)?;
    }

    Ok((model_target, config_target, tokenizer_target))
}
