//! BGE (BAAI general embeddings) text encoder.
//!
//! BGE models are standard BERT architecture with:
//! 1. CLS-token pooling (take `hidden_states[:, 0, :]`)
//! 2. L2 normalisation on the pooled vector
//!
//! We reuse candle's `bert::BertModel` and wrap it with tokenisation +
//! batching + the pooling/normalise tail.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::info;

use batata_ai_core::error::{BatataError, Result};

/// BGE text encoder — loads from a local directory containing
/// `model.safetensors`, `config.json`, and `tokenizer.json`.
pub struct BgeModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
    max_len: usize,
}

impl BgeModel {
    /// Load from a local directory. The directory must contain
    /// `config.json`, `tokenizer.json`, and `model.safetensors`.
    pub fn load_local(dir: &Path, device: &Device) -> Result<Self> {
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let weights_path = dir.join("model.safetensors");

        info!("loading BGE model from {}", dir.display());

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str).map_err(|e| {
            BatataError::Inference(format!("failed to parse BGE config: {e}"))
        })?;
        let dim = config.hidden_size;
        let max_len = config.max_position_embeddings;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;
        // Enable dynamic padding + truncation to `max_len` so encoding
        // a batch returns equal-length tensors without manual work.
        let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_id,
                pad_token: "[PAD]".into(),
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .map_err(|e| BatataError::Inference(format!("truncation setup failed: {e}")))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path.clone()],
                DTYPE,
                device,
            )
            .map_err(|e| BatataError::Inference(format!("failed to load weights: {e}")))?
        };
        let model = BertModel::load(vb, &config)
            .map_err(|e| BatataError::Inference(format!("failed to build BERT model: {e}")))?;

        info!("BGE model loaded (dim={dim}, max_len={max_len})");
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

    /// Encode a batch of texts into L2-normalised, CLS-pooled vectors.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenise as batch.
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
            .forward(&token_ids, &type_ids, Some(&attn_mask))
            .map_err(|e| BatataError::Inference(format!("BERT forward failed: {e}")))?;

        // CLS pool: take the first token of each sequence → [batch, hidden]
        let cls = hidden
            .i((.., 0, ..))
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        // L2 normalise along the hidden dim.
        let norm = l2_normalize(&cls)
            .map_err(|e| BatataError::Inference(format!("l2 normalise failed: {e}")))?;

        // Pull back to Vec<Vec<f32>>.
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
