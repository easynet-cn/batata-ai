use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;

use batata_ai_core::error::{BatataError, Result};

/// Parameters for text generation
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: Some(0.9),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 42,
        }
    }
}

/// Token-by-token text generation loop.
/// Works with any model that implements the `forward` method returning logits.
pub fn generate_tokens(
    logits_processor: &mut LogitsProcessor,
    logits: &Tensor,
    tokens: &mut Vec<u32>,
    params: &GenerationParams,
) -> Result<u32> {
    let logits = logits
        .squeeze(0)
        .map_err(|e| BatataError::Inference(e.to_string()))?;

    let logits = if params.repeat_penalty != 1.0 {
        let start = tokens.len().saturating_sub(params.repeat_last_n);
        candle_transformers::utils::apply_repeat_penalty(
            &logits,
            params.repeat_penalty,
            &tokens[start..],
        )
        .map_err(|e| BatataError::Inference(e.to_string()))?
    } else {
        logits
    };

    let next_token = logits_processor
        .sample(&logits)
        .map_err(|e| BatataError::Inference(e.to_string()))?;

    tokens.push(next_token);
    Ok(next_token)
}
