use std::io::Cursor;
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_phi3::ModelWeights;
use tokenizers::Tokenizer;
use tracing::info;

use batata_ai_core::error::{BatataError, Result};

use crate::inference::GenerationParams;

/// A loaded Phi-3 model ready for inference
pub struct Phi3Model {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
}

impl Phi3Model {
    /// Load a Phi-3 model from a GGUF file and a tokenizer file
    pub fn load(model_path: &Path, tokenizer_path: &Path, device: &Device) -> Result<Self> {
        info!("loading Phi-3 model from {}", model_path.display());

        let mut file = std::fs::File::open(model_path)
            .map_err(|e| BatataError::Inference(format!("failed to open model: {e}")))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| BatataError::Inference(format!("failed to read GGUF: {e}")))?;

        let model = ModelWeights::from_gguf(false, content, &mut file, device)
            .map_err(|e| BatataError::Inference(format!("failed to load model weights: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;

        info!("Phi-3 model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Load from raw bytes (useful for downloaded content)
    pub fn load_from_bytes(
        model_bytes: &[u8],
        tokenizer_bytes: &[u8],
        device: &Device,
    ) -> Result<Self> {
        let mut cursor = Cursor::new(model_bytes);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| BatataError::Inference(format!("failed to read GGUF: {e}")))?;

        let model = ModelWeights::from_gguf(false, content, &mut cursor, device)
            .map_err(|e| BatataError::Inference(format!("failed to load model weights: {e}")))?;

        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Format messages into Phi-3 chat template
    pub fn format_prompt(messages: &[(String, String)]) -> String {
        let mut prompt = String::new();
        for (role, content) in messages {
            match role.as_str() {
                "system" => {
                    prompt.push_str(&format!("<|system|>\n{content}<|end|>\n"));
                }
                "user" => {
                    prompt.push_str(&format!("<|user|>\n{content}<|end|>\n"));
                }
                "assistant" => {
                    prompt.push_str(&format!("<|assistant|>\n{content}<|end|>\n"));
                }
                _ => {}
            }
        }
        prompt.push_str("<|assistant|>\n");
        prompt
    }

    fn resolve_eos_token(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<|end|>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| self.tokenizer.token_to_id("</s>"))
    }

    fn create_logits_processor(params: &GenerationParams) -> LogitsProcessor {
        LogitsProcessor::from_sampling(
            params.seed,
            candle_transformers::generation::Sampling::TopP {
                p: params.top_p.unwrap_or(0.9),
                temperature: params.temperature,
            },
        )
    }

    fn sample_next_token(
        &self,
        logits: &Tensor,
        tokens: &[u32],
        logits_processor: &mut LogitsProcessor,
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

        logits_processor
            .sample(&logits)
            .map_err(|e| BatataError::Inference(e.to_string()))
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| BatataError::Inference(format!("tokenization failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode_token(&self, token: u32) -> Result<String> {
        self.tokenizer
            .decode(&[token], true)
            .map_err(|e| BatataError::Inference(format!("decoding failed: {e}")))
    }

    fn forward_prompt(&mut self, tokens: &[u32]) -> Result<Tensor> {
        let input = Tensor::new(tokens, &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        self.model
            .forward(&input, 0)
            .map_err(|e| BatataError::Inference(e.to_string()))
    }

    fn forward_token(&mut self, token: u32, index_pos: usize) -> Result<Tensor> {
        let input = Tensor::new(&[token], &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        self.model
            .forward(&input, index_pos)
            .map_err(|e| BatataError::Inference(e.to_string()))
    }

    /// Generate text from a list of (role, content) messages
    pub fn chat(
        &mut self,
        messages: &[(String, String)],
        params: &GenerationParams,
    ) -> Result<String> {
        let prompt = Self::format_prompt(messages);
        self.generate(&prompt, params)
    }

    /// Generate text from a raw prompt string (non-streaming)
    pub fn generate(&mut self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let mut tokens = self.encode_prompt(prompt)?;
        let prompt_len = tokens.len();
        let eos_token = self.resolve_eos_token();
        let mut logits_processor = Self::create_logits_processor(params);

        // Process prompt
        let logits = self.forward_prompt(&tokens)?;
        let next_token = self.sample_next_token(&logits, &tokens, &mut logits_processor, params)?;
        tokens.push(next_token);

        let mut generated_tokens = vec![next_token];

        // Generate tokens
        for i in 0..params.max_tokens.saturating_sub(1) {
            if eos_token.is_some_and(|eos| *generated_tokens.last().unwrap() == eos) {
                break;
            }

            let logits =
                self.forward_token(*generated_tokens.last().unwrap(), prompt_len + i)?;
            let next_token =
                self.sample_next_token(&logits, &tokens, &mut logits_processor, params)?;

            tokens.push(next_token);
            generated_tokens.push(next_token);
        }

        // Remove EOS token
        if let Some(eos) = eos_token {
            if generated_tokens.last() == Some(&eos) {
                generated_tokens.pop();
            }
        }

        self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| BatataError::Inference(format!("decoding failed: {e}")))
    }

    /// Generate tokens with streaming — sends each decoded token text via the callback.
    /// Returns the full generated text.
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let mut tokens = self.encode_prompt(prompt)?;
        let prompt_len = tokens.len();
        let eos_token = self.resolve_eos_token();
        let mut logits_processor = Self::create_logits_processor(params);

        // Process prompt
        let logits = self.forward_prompt(&tokens)?;
        let next_token = self.sample_next_token(&logits, &tokens, &mut logits_processor, params)?;
        tokens.push(next_token);

        if eos_token.is_some_and(|eos| next_token == eos) {
            return Ok(String::new());
        }

        let text = self.decode_token(next_token)?;
        on_token(&text);
        let mut full_text = text;

        // Generate remaining tokens
        for i in 0..params.max_tokens.saturating_sub(1) {
            let last_token = *tokens.last().unwrap();

            let logits = self.forward_token(last_token, prompt_len + i)?;
            let next_token =
                self.sample_next_token(&logits, &tokens, &mut logits_processor, params)?;

            if eos_token.is_some_and(|eos| next_token == eos) {
                break;
            }

            tokens.push(next_token);

            let text = self.decode_token(next_token)?;
            on_token(&text);
            full_text.push_str(&text);
        }

        Ok(full_text)
    }

    /// Streaming chat — sends tokens via callback
    pub fn chat_stream<F>(
        &mut self,
        messages: &[(String, String)],
        params: &GenerationParams,
        on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let prompt = Self::format_prompt(messages);
        self.generate_stream(&prompt, params, on_token)
    }
}

/// Download Phi-3-mini GGUF model from Hugging Face Hub
pub fn download_phi3_mini(
    quantization: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf";
    let model_filename = match quantization {
        "q4" | "q4_k_m" => "Phi-3-mini-4k-instruct-q4.gguf",
        "fp16" => "Phi-3-mini-4k-instruct-fp16.gguf",
        _ => "Phi-3-mini-4k-instruct-q4.gguf",
    };

    info!("downloading Phi-3-mini model ({quantization})...");

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| BatataError::Inference(e.to_string()))?;

    let model_repo = api.model(repo_id.to_string());
    let model_path = model_repo
        .get(model_filename)
        .map_err(|e| BatataError::Inference(format!("model download failed: {e}")))?;

    // Tokenizer comes from the non-GGUF repo
    let tokenizer_repo = api.model("microsoft/Phi-3-mini-4k-instruct".to_string());
    let tokenizer_path = tokenizer_repo
        .get("tokenizer.json")
        .map_err(|e| BatataError::Inference(format!("tokenizer download failed: {e}")))?;

    info!("download complete");
    Ok((model_path, tokenizer_path))
}
