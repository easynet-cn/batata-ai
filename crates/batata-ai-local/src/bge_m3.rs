//! BGE-M3 text encoder — based on XLM-RoBERTa-large.
//!
//! The bge-small-zh-v1.5 style (BERT) loader lives in [`crate::bge`];
//! this module handles the M3 family specifically because it:
//! 1. Uses XLM-RoBERTa, not BERT.
//! 2. Ships weights only as `pytorch_model.bin` (no safetensors).
//! 3. Has 1024 hidden / 8192 position embeddings — heavier all around.
//!
//! Retrieval recipe is the same: run the encoder, take the CLS token
//! (`<s>`, id 0 in the XLM-R vocab), L2-normalise the resulting vector.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaModel};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::info;

use batata_ai_core::error::{BatataError, Result};

/// BGE-M3 encoder. Takes a directory containing `config.json`,
/// `tokenizer.json`, and `pytorch_model.bin`.
pub struct BgeM3Model {
    model: XLMRobertaModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
    max_len: usize,
}

impl BgeM3Model {
    pub fn load_local(dir: &Path, device: &Device) -> Result<Self> {
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        // BGE-M3 ships .bin only; fall back to safetensors if ever added.
        let pth_path = dir.join("pytorch_model.bin");
        let safetensors_path = dir.join("model.safetensors");

        info!("loading BGE-M3 (xlm-roberta) model from {}", dir.display());

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str).map_err(|e| {
            BatataError::Inference(format!("failed to parse BGE-M3 config: {e}"))
        })?;
        let dim = config.hidden_size;
        // HF's XLMRoBERTa sets max_position_embeddings = actual + 2
        // (padding + offset), so the usable context is max_len - 2.
        let max_len = config.max_position_embeddings.saturating_sub(2);

        // Tokenizer: uses the XLMR sentencepiece; the .json file is
        // self-contained so `from_file` works without touching the .model.
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;
        let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(1);
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_id,
                pad_token: "<pad>".into(),
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .map_err(|e| BatataError::Inference(format!("truncation setup failed: {e}")))?;

        // Load weights. `from_pth` reads a raw pickle; the BGE-M3
        // checkpoint is flat (no "roberta." prefix).
        let vb = if safetensors_path.exists() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[safetensors_path],
                    DType::F32,
                    device,
                )
                .map_err(|e| BatataError::Inference(format!("failed to load safetensors: {e}")))?
            }
        } else {
            VarBuilder::from_pth(&pth_path, DType::F32, device)
                .map_err(|e| BatataError::Inference(format!("failed to load .bin: {e}")))?
        };

        // BGE-M3's checkpoint is flat — the encoder/embeddings live at
        // the top level of the state dict (no `roberta.` prefix).
        let model = XLMRobertaModel::new(&config, vb.clone())
            .or_else(|_| XLMRobertaModel::new(&config, vb.pp("roberta")))
            .map_err(|e| BatataError::Inference(format!("failed to build XLMR model: {e}")))?;

        info!("BGE-M3 model loaded (dim={dim}, max_len={max_len})");
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dim,
            max_len,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dim
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Encode a batch of strings to L2-normalised CLS-pooled vectors.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| BatataError::Inference(format!("tokenise batch failed: {e}")))?;

        let batch_size = encodings.len();
        let seq_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        let mut token_ids = Vec::with_capacity(batch_size * seq_len);
        let mut type_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attn_mask = Vec::with_capacity(batch_size * seq_len);
        for enc in &encodings {
            token_ids.extend_from_slice(enc.get_ids());
            type_ids.extend_from_slice(enc.get_type_ids());
            attn_mask.extend_from_slice(enc.get_attention_mask());
        }

        let token_ids = Tensor::from_vec(token_ids, (batch_size, seq_len), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let type_ids = Tensor::from_vec(type_ids, (batch_size, seq_len), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let attn_mask = Tensor::from_vec(attn_mask, (batch_size, seq_len), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        let hidden = self
            .model
            .forward(&token_ids, &attn_mask, &type_ids, None, None, None)
            .map_err(|e| BatataError::Inference(format!("xlm-roberta forward failed: {e}")))?;

        let cls = hidden
            .i((.., 0, ..))
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        let norm = l2_normalize(&cls)
            .map_err(|e| BatataError::Inference(format!("l2 normalise failed: {e}")))?;

        let out: Vec<Vec<f32>> = norm
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec2())
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        Ok(out)
    }
}

fn l2_normalize(t: &Tensor) -> candle_core::Result<Tensor> {
    let squared = t.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let norm = squared.sqrt()?;
    t.broadcast_div(&norm)
}
