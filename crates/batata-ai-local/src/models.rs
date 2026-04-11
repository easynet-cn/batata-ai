use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use tracing::info;

use batata_ai_core::error::{BatataError, Result};

use crate::inference::GenerationParams;

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Phi3,
    Llama,
    Qwen2,
    Qwen3,
    Gemma3,
}

/// Wrapper around different quantized model backends with a uniform interface
enum ModelBackend {
    Phi3(candle_transformers::models::quantized_phi3::ModelWeights),
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    Qwen2(candle_transformers::models::quantized_qwen2::ModelWeights),
    Qwen3(candle_transformers::models::quantized_qwen3::ModelWeights),
    Gemma3(candle_transformers::models::quantized_gemma3::ModelWeights),
}

impl ModelBackend {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Phi3(m) => m.forward(x, index_pos),
            Self::Llama(m) => m.forward(x, index_pos),
            Self::Qwen2(m) => m.forward(x, index_pos),
            Self::Qwen3(m) => m.forward(x, index_pos),
            Self::Gemma3(m) => m.forward(x, index_pos),
        }
    }
}

/// Chat template format for different model families
#[derive(Debug, Clone, Copy)]
pub enum ChatTemplate {
    /// Phi-3: <|system|>\n...<|end|>\n<|user|>\n...<|end|>\n<|assistant|>\n
    Phi3,
    /// Llama-3: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>...
    Llama3,
    /// Qwen: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    ChatML,
    /// Gemma: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
    Gemma,
}

/// Model descriptor for downloading from HuggingFace Hub
#[derive(Debug, Clone)]
pub struct ModelDescriptor {
    pub arch: ModelArch,
    pub name: String,
    pub repo_id: String,
    pub filenames: Vec<String>,
    pub tokenizer_repo: String,
    pub chat_template: ChatTemplate,
    pub eos_tokens: Vec<String>,
}

/// Pre-defined model descriptors
impl ModelDescriptor {
    pub fn phi3_mini_q4() -> Self {
        Self {
            arch: ModelArch::Phi3,
            name: "phi-3-mini".into(),
            repo_id: "microsoft/Phi-3-mini-4k-instruct-gguf".into(),
            filenames: vec!["Phi-3-mini-4k-instruct-q4.gguf".into()],
            tokenizer_repo: "microsoft/Phi-3-mini-4k-instruct".into(),
            chat_template: ChatTemplate::Phi3,
            eos_tokens: vec!["<|end|>".into(), "<|endoftext|>".into()],
        }
    }

    pub fn llama3_8b_q4() -> Self {
        Self {
            arch: ModelArch::Llama,
            name: "llama-3-8b".into(),
            repo_id: "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF".into(),
            filenames: vec!["Meta-Llama-3-8B-Instruct.Q4_K_M.gguf".into()],
            tokenizer_repo: "NousResearch/Meta-Llama-3-8B-Instruct".into(),
            chat_template: ChatTemplate::Llama3,
            eos_tokens: vec![
                "<|eot_id|>".into(),
                "<|end_of_text|>".into(),
            ],
        }
    }

    pub fn qwen2_1_5b_q4() -> Self {
        Self {
            arch: ModelArch::Qwen2,
            name: "qwen2-1.5b".into(),
            repo_id: "Qwen/Qwen2-1.5B-Instruct-GGUF".into(),
            filenames: vec!["qwen2-1_5b-instruct-q4_k_m.gguf".into()],
            tokenizer_repo: "Qwen/Qwen2-1.5B-Instruct".into(),
            chat_template: ChatTemplate::ChatML,
            eos_tokens: vec!["<|im_end|>".into(), "<|endoftext|>".into()],
        }
    }

    /// Qwen3-1.7B (native candle Qwen3 loader).
    ///
    /// Note: The official `Qwen/Qwen3-1.7B-GGUF` repo only ships the Q8_0
    /// quantization, so this descriptor uses Q8_0 (≈1.8 GB).
    pub fn qwen3_1_7b_q4() -> Self {
        Self {
            arch: ModelArch::Qwen3,
            name: "qwen3-1.7b".into(),
            repo_id: "Qwen/Qwen3-1.7B-GGUF".into(),
            filenames: vec!["Qwen3-1.7B-Q8_0.gguf".into()],
            tokenizer_repo: "Qwen/Qwen3-1.7B".into(),
            chat_template: ChatTemplate::ChatML,
            eos_tokens: vec!["<|im_end|>".into(), "<|endoftext|>".into()],
        }
    }

    pub fn qwen3_4b_q4() -> Self {
        Self {
            arch: ModelArch::Qwen3,
            name: "qwen3-4b".into(),
            repo_id: "Qwen/Qwen3-4B-GGUF".into(),
            filenames: vec!["Qwen3-4B-Q4_K_M.gguf".into()],
            tokenizer_repo: "Qwen/Qwen3-4B".into(),
            chat_template: ChatTemplate::ChatML,
            eos_tokens: vec!["<|im_end|>".into(), "<|endoftext|>".into()],
        }
    }

    pub fn gemma3_1b_q4() -> Self {
        Self {
            arch: ModelArch::Gemma3,
            name: "gemma3-1b".into(),
            repo_id: "google/gemma-3-1b-it-qat-q4_0-gguf".into(),
            filenames: vec!["gemma-3-1b-it-q4_0.gguf".into()],
            tokenizer_repo: "google/gemma-3-1b-it".into(),
            chat_template: ChatTemplate::Gemma,
            eos_tokens: vec!["<end_of_turn>".into(), "<eos>".into()],
        }
    }

    pub fn gemma3_4b_q4() -> Self {
        Self {
            arch: ModelArch::Gemma3,
            name: "gemma3-4b".into(),
            repo_id: "google/gemma-3-4b-it-qat-q4_0-gguf".into(),
            filenames: vec!["gemma-3-4b-it-q4_0.gguf".into()],
            tokenizer_repo: "google/gemma-3-4b-it".into(),
            chat_template: ChatTemplate::Gemma,
            eos_tokens: vec!["<end_of_turn>".into(), "<eos>".into()],
        }
    }

    /// Lookup a model descriptor by name
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "phi-3-mini" | "phi3" => Some(Self::phi3_mini_q4()),
            "llama-3-8b" | "llama3" => Some(Self::llama3_8b_q4()),
            "qwen2-1.5b" | "qwen2" => Some(Self::qwen2_1_5b_q4()),
            "qwen3-1.7b" | "qwen3" => Some(Self::qwen3_1_7b_q4()),
            "qwen3-4b" => Some(Self::qwen3_4b_q4()),
            "gemma3-1b" | "gemma3" => Some(Self::gemma3_1b_q4()),
            "gemma3-4b" => Some(Self::gemma3_4b_q4()),
            _ => None,
        }
    }

    /// List all available model names
    pub fn available_models() -> Vec<&'static str> {
        vec![
            "phi-3-mini",
            "llama-3-8b",
            "qwen2-1.5b",
            "qwen3-1.7b",
            "qwen3-4b",
            "gemma3-1b",
            "gemma3-4b",
        ]
    }
}

/// A loaded model ready for inference — supports multiple architectures
pub struct LocalModel {
    backend: ModelBackend,
    tokenizer: Tokenizer,
    device: Device,
    descriptor: ModelDescriptor,
}

impl LocalModel {
    /// Load a model from a GGUF file
    pub fn load(
        descriptor: ModelDescriptor,
        model_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        info!("loading {} model from {}", descriptor.name, model_path.display());

        let mut file = std::fs::File::open(model_path)
            .map_err(|e| BatataError::Inference(format!("failed to open model: {e}")))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| BatataError::Inference(format!("failed to read GGUF: {e}")))?;

        let backend = Self::load_backend(descriptor.arch, content, &mut file, device)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;

        info!("{} model loaded successfully", descriptor.name);

        Ok(Self {
            backend,
            tokenizer,
            device: device.clone(),
            descriptor,
        })
    }

    /// Auto-detect model architecture from GGUF metadata and load.
    ///
    /// Reads the `general.architecture` field from GGUF metadata to determine
    /// the correct model backend. Falls back to the descriptor's arch if metadata
    /// is unavailable.
    ///
    /// This allows loading any GGUF file without manually specifying architecture,
    /// supporting all quantization formats (Q4_0, Q4_K_M, Q5_K_M, Q8_0, etc.)
    /// as long as the model architecture is supported.
    pub fn load_auto(
        model_path: &Path,
        tokenizer_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        info!("auto-detecting model from {}", model_path.display());

        let mut file = std::fs::File::open(model_path)
            .map_err(|e| BatataError::Inference(format!("failed to open model: {e}")))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| BatataError::Inference(format!("failed to read GGUF: {e}")))?;

        // Read architecture from GGUF metadata
        let arch = Self::detect_arch_from_metadata(&content)?;

        info!(arch = ?arch, "detected model architecture from GGUF metadata");

        // Determine chat template and EOS tokens from arch
        let (chat_template, eos_tokens) = match arch {
            ModelArch::Phi3 => (ChatTemplate::Phi3, vec!["<|end|>".into(), "<|endoftext|>".into()]),
            ModelArch::Llama => (ChatTemplate::Llama3, vec!["<|eot_id|>".into(), "<|end_of_text|>".into()]),
            ModelArch::Qwen2 => (ChatTemplate::ChatML, vec!["<|im_end|>".into(), "<|endoftext|>".into()]),
            ModelArch::Qwen3 => (ChatTemplate::ChatML, vec!["<|im_end|>".into(), "<|endoftext|>".into()]),
            ModelArch::Gemma3 => (ChatTemplate::Gemma, vec!["<end_of_turn>".into(), "<eos>".into()]),
        };

        let name = model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let descriptor = ModelDescriptor {
            arch,
            name: name.clone(),
            repo_id: String::new(),
            filenames: vec![],
            tokenizer_repo: String::new(),
            chat_template,
            eos_tokens,
        };

        let backend = Self::load_backend(arch, content, &mut file, device)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;

        info!("{name} model loaded successfully (auto-detected)");

        Ok(Self {
            backend,
            tokenizer,
            device: device.clone(),
            descriptor,
        })
    }

    /// Detect `ModelArch` from GGUF metadata `general.architecture` field.
    fn detect_arch_from_metadata(content: &gguf_file::Content) -> Result<ModelArch> {
        let arch_value = content
            .metadata
            .get("general.architecture")
            .ok_or_else(|| {
                BatataError::Inference(
                    "GGUF file missing 'general.architecture' metadata".to_string(),
                )
            })?;

        let arch_str = match arch_value {
            gguf_file::Value::String(s) => s.as_str(),
            _ => {
                return Err(BatataError::Inference(
                    "general.architecture is not a string".to_string(),
                ))
            }
        };

        match arch_str {
            "phi3" => Ok(ModelArch::Phi3),
            "llama" => Ok(ModelArch::Llama),
            "qwen2" => Ok(ModelArch::Qwen2),
            "qwen3" => Ok(ModelArch::Qwen3),
            "gemma3" | "gemma2" => Ok(ModelArch::Gemma3),
            other => Err(BatataError::Inference(format!(
                "unsupported GGUF architecture '{other}', supported: phi3, llama, qwen2, qwen3, gemma3"
            ))),
        }
    }

    fn load_backend(
        arch: ModelArch,
        content: gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
    ) -> Result<ModelBackend> {
        match arch {
            ModelArch::Phi3 => {
                let m = candle_transformers::models::quantized_phi3::ModelWeights::from_gguf(
                    false, content, file, device,
                )
                .map_err(|e| BatataError::Inference(format!("failed to load Phi-3: {e}")))?;
                Ok(ModelBackend::Phi3(m))
            }
            ModelArch::Llama => {
                let m = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    content, file, device,
                )
                .map_err(|e| BatataError::Inference(format!("failed to load Llama: {e}")))?;
                Ok(ModelBackend::Llama(m))
            }
            ModelArch::Qwen2 => {
                let m = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(
                    content, file, device,
                )
                .map_err(|e| BatataError::Inference(format!("failed to load Qwen2: {e}")))?;
                Ok(ModelBackend::Qwen2(m))
            }
            ModelArch::Qwen3 => {
                let m = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
                    content, file, device,
                )
                .map_err(|e| BatataError::Inference(format!("failed to load Qwen3: {e}")))?;
                Ok(ModelBackend::Qwen3(m))
            }
            ModelArch::Gemma3 => {
                let m = candle_transformers::models::quantized_gemma3::ModelWeights::from_gguf(
                    content, file, device,
                )
                .map_err(|e| BatataError::Inference(format!("failed to load Gemma3: {e}")))?;
                Ok(ModelBackend::Gemma3(m))
            }
        }
    }

    /// Download model from HuggingFace Hub and load it
    pub fn download_and_load(descriptor: ModelDescriptor, device: &Device) -> Result<Self> {
        let (model_path, tokenizer_path) = download_model(&descriptor)?;
        Self::load(descriptor, &model_path, &tokenizer_path, device)
    }

    /// Format messages using the model's chat template
    pub fn format_prompt(&self, messages: &[(String, String)]) -> String {
        format_chat_messages(messages, self.descriptor.chat_template)
    }

    fn resolve_eos_token(&self) -> Option<u32> {
        for token_str in &self.descriptor.eos_tokens {
            if let Some(id) = self.tokenizer.token_to_id(token_str) {
                return Some(id);
            }
        }
        self.tokenizer.token_to_id("</s>")
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

    fn forward_tokens(&mut self, tokens: &[u32], index_pos: usize) -> Result<Tensor> {
        let input = Tensor::new(tokens, &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        self.backend
            .forward(&input, index_pos)
            .map_err(|e| BatataError::Inference(e.to_string()))
    }

    /// Chat with the model (non-streaming)
    pub fn chat(
        &mut self,
        messages: &[(String, String)],
        params: &GenerationParams,
    ) -> Result<String> {
        let prompt = self.format_prompt(messages);
        self.generate(&prompt, params)
    }

    /// Generate text from a raw prompt (non-streaming)
    pub fn generate(&mut self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let mut tokens = self.encode_prompt(prompt)?;
        let prompt_len = tokens.len();
        let eos_token = self.resolve_eos_token();
        let mut logits_processor = Self::create_logits_processor(params);

        let logits = self.forward_tokens(&tokens, 0)?;
        let next_token = self.sample_next_token(&logits, &tokens, &mut logits_processor, params)?;
        tokens.push(next_token);

        let mut generated_tokens = vec![next_token];

        for i in 0..params.max_tokens.saturating_sub(1) {
            if eos_token.is_some_and(|eos| *generated_tokens.last().unwrap() == eos) {
                break;
            }

            let logits =
                self.forward_tokens(&[*generated_tokens.last().unwrap()], prompt_len + i)?;
            let next_token =
                self.sample_next_token(&logits, &tokens, &mut logits_processor, params)?;

            tokens.push(next_token);
            generated_tokens.push(next_token);
        }

        if let Some(eos) = eos_token {
            if generated_tokens.last() == Some(&eos) {
                generated_tokens.pop();
            }
        }

        self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| BatataError::Inference(format!("decoding failed: {e}")))
    }

    /// Generate with streaming — sends each token via callback
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

        let logits = self.forward_tokens(&tokens, 0)?;
        let next_token = self.sample_next_token(&logits, &tokens, &mut logits_processor, params)?;
        tokens.push(next_token);

        if eos_token.is_some_and(|eos| next_token == eos) {
            return Ok(String::new());
        }

        let text = self.decode_token(next_token)?;
        on_token(&text);
        let mut full_text = text;

        for i in 0..params.max_tokens.saturating_sub(1) {
            let last_token = *tokens.last().unwrap();
            let logits = self.forward_tokens(&[last_token], prompt_len + i)?;
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

    /// Streaming chat
    pub fn chat_stream<F>(
        &mut self,
        messages: &[(String, String)],
        params: &GenerationParams,
        on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let prompt = self.format_prompt(messages);
        self.generate_stream(&prompt, params, on_token)
    }
}

/// Format chat messages using the specified template
pub fn format_chat_messages(messages: &[(String, String)], template: ChatTemplate) -> String {
    match template {
        ChatTemplate::Phi3 => {
            let mut prompt = String::new();
            for (role, content) in messages {
                match role.as_str() {
                    "system" => prompt.push_str(&format!("<|system|>\n{content}<|end|>\n")),
                    "user" => prompt.push_str(&format!("<|user|>\n{content}<|end|>\n")),
                    "assistant" => {
                        prompt.push_str(&format!("<|assistant|>\n{content}<|end|>\n"))
                    }
                    _ => {}
                }
            }
            prompt.push_str("<|assistant|>\n");
            prompt
        }
        ChatTemplate::Llama3 => {
            let mut prompt = "<|begin_of_text|>".to_string();
            for (role, content) in messages {
                prompt.push_str(&format!(
                    "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                ));
            }
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            prompt
        }
        ChatTemplate::ChatML => {
            let mut prompt = String::new();
            for (role, content) in messages {
                prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            prompt.push_str("<|im_start|>assistant\n");
            prompt
        }
        ChatTemplate::Gemma => {
            let mut prompt = String::new();
            for (role, content) in messages {
                let gemma_role = match role.as_str() {
                    "assistant" => "model",
                    other => other,
                };
                prompt
                    .push_str(&format!("<start_of_turn>{gemma_role}\n{content}<end_of_turn>\n"));
            }
            prompt.push_str("<start_of_turn>model\n");
            prompt
        }
    }
}

/// Download a model and its tokenizer from HuggingFace Hub.
///
/// Returns `(model_path, tokenizer_path)` pointing to the cached files in `~/.cache/huggingface/`.
pub fn download_model(
    descriptor: &ModelDescriptor,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    info!("downloading {} model...", descriptor.name);

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| BatataError::Inference(e.to_string()))?;

    let model_repo = api.model(descriptor.repo_id.clone());
    let model_path = model_repo
        .get(&descriptor.filenames[0])
        .map_err(|e| BatataError::Inference(format!("model download failed: {e}")))?;

    let tokenizer_repo = api.model(descriptor.tokenizer_repo.clone());
    let tokenizer_path = tokenizer_repo
        .get("tokenizer.json")
        .map_err(|e| BatataError::Inference(format!("tokenizer download failed: {e}")))?;

    info!("download complete");
    Ok((model_path, tokenizer_path))
}

/// Resolve the base directory used to store downloaded models.
///
/// Resolution order:
/// 1. `BATATA_AI_MODELS_DIR` environment variable, if set.
/// 2. `$HOME/work/models` (default layout for this project).
/// 3. `./models` as a last-resort fallback when `$HOME` is not available.
pub fn default_models_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("BATATA_AI_MODELS_DIR") {
        return std::path::PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return std::path::PathBuf::from(home).join("work").join("models");
    }
    std::path::PathBuf::from("./models")
}

/// Resolve the on-disk directory for a specific model name, based on
/// [`default_models_dir`]. The layout is `<base>/<model_name>/`.
pub fn resolve_model_dir(model_name: &str) -> std::path::PathBuf {
    default_models_dir().join(model_name)
}

/// Download a model by name into its resolved directory under
/// [`default_models_dir`]. Equivalent to
/// `download_model_to(model_name, &default_models_dir(), true)`.
pub fn download_model_default(
    model_name: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    download_model_to(model_name, &default_models_dir(), true)
}

/// Download a model by name and copy files to a target directory.
///
/// Returns `(model_path, tokenizer_path)` in the target directory.
///
/// When `nest_by_model` is `true`, files are placed under
/// `target_dir/<model_name>/` so that multiple models can coexist inside the
/// same base directory. When `false`, files are placed directly in
/// `target_dir` (legacy behavior, kept for callers that already manage the
/// layout themselves).
///
/// This is useful for:
/// - Offline deployment (pre-download, then use `LocalProvider::from_local`)
/// - Docker images (download at build time)
/// - Air-gapped environments
pub fn download_model_to(
    model_name: &str,
    target_dir: &std::path::Path,
    nest_by_model: bool,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let descriptor = ModelDescriptor::by_name(model_name).ok_or_else(|| {
        BatataError::ModelNotFound(format!(
            "unknown model '{model_name}', available: {:?}",
            ModelDescriptor::available_models()
        ))
    })?;

    // Download to HF cache first
    let (cached_model, cached_tokenizer) = download_model(&descriptor)?;

    // Resolve final directory: nest under <target_dir>/<model_name>/ when asked
    let final_dir: std::path::PathBuf = if nest_by_model {
        target_dir.join(&descriptor.name)
    } else {
        target_dir.to_path_buf()
    };

    // Ensure target directory exists
    std::fs::create_dir_all(&final_dir)?;

    // Copy model file
    let model_filename = &descriptor.filenames[0];
    let target_model = final_dir.join(model_filename);
    if !target_model.exists() {
        info!("copying model to {}", target_model.display());
        std::fs::copy(&cached_model, &target_model)?;
    }

    // Copy tokenizer
    let target_tokenizer = final_dir.join("tokenizer.json");
    if !target_tokenizer.exists() {
        info!("copying tokenizer to {}", target_tokenizer.display());
        std::fs::copy(&cached_tokenizer, &target_tokenizer)?;
    }

    info!(
        "model files ready in {}",
        final_dir.display()
    );
    Ok((target_model, target_tokenizer))
}
