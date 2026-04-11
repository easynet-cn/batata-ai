//! BGE cross-encoder reranker (XLM-RoBERTa-base + classification head).
//!
//! Takes (query, passage) pairs as `<s>query</s></s>passage</s>` and
//! returns a single relevance logit per pair. Combine with a coarse
//! first-stage retrieval to boost top-k precision.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaForSequenceClassification};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::info;

use batata_ai_core::error::{BatataError, Result};

pub struct BgeRerankerModel {
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
    max_len: usize,
}

impl BgeRerankerModel {
    pub fn load_local(dir: &Path, device: &Device) -> Result<Self> {
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let safetensors_path = dir.join("model.safetensors");
        let pth_path = dir.join("pytorch_model.bin");

        info!("loading BGE reranker from {}", dir.display());

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| BatataError::Inference(format!("reranker config parse: {e}")))?;
        let max_len = config.max_position_embeddings.saturating_sub(2);

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("tokenizer load: {e}")))?;
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
            .map_err(|e| BatataError::Inference(format!("trunc setup: {e}")))?;

        let vb = if safetensors_path.exists() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, device)
                    .map_err(|e| BatataError::Inference(format!("safetensors load: {e}")))?
            }
        } else {
            VarBuilder::from_pth(&pth_path, DType::F32, device)
                .map_err(|e| BatataError::Inference(format!("pth load: {e}")))?
        };

        // BGE reranker exports the sequence-classification head at the
        // root level (so classifier and roberta live next to each other).
        // candle's XLMRobertaForSequenceClassification::new adds its own
        // `roberta.` and `classifier.` prefixes internally, matching the
        // HF naming.
        let model = XLMRobertaForSequenceClassification::new(1, &config, vb)
            .map_err(|e| BatataError::Inference(format!("build reranker: {e}")))?;

        info!("BGE reranker loaded (max_len={max_len})");
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            max_len,
        })
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Score a batch of (query, passage) pairs. Returns a relevance
    /// logit per pair — higher is more relevant. Not normalised;
    /// applying a sigmoid is optional and only useful for thresholding.
    pub fn score_pairs(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        // Tokenise each (query, passage) as a pair. The tokenizer emits
        // `<s> query </s> </s> passage </s>` with correct type_ids.
        let pairs: Vec<(String, String)> = passages
            .iter()
            .map(|p| (query.to_string(), p.clone()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| BatataError::Inference(format!("tokenise rerank batch: {e}")))?;

        let batch = encodings.len();
        let seq = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        let mut token_ids = Vec::with_capacity(batch * seq);
        let mut type_ids = Vec::with_capacity(batch * seq);
        let mut attn = Vec::with_capacity(batch * seq);
        for enc in &encodings {
            token_ids.extend_from_slice(enc.get_ids());
            type_ids.extend_from_slice(enc.get_type_ids());
            attn.extend_from_slice(enc.get_attention_mask());
        }
        let token_ids = Tensor::from_vec(token_ids, (batch, seq), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let type_ids = Tensor::from_vec(type_ids, (batch, seq), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let attn = Tensor::from_vec(attn, (batch, seq), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        let logits = self
            .model
            .forward(&token_ids, &attn, &type_ids)
            .map_err(|e| BatataError::Inference(format!("reranker forward: {e}")))?;

        // Shape: [batch, 1] → flatten to [batch].
        let logits = logits
            .squeeze(1)
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let scores: Vec<f32> = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec1())
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        Ok(scores)
    }
}
