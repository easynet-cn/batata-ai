use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use batata_ai_core::{
    error::{BatataError, Result},
    message::{ChatRequest, ChatResponse, ChatStream, Role},
    provider::{Provider, ProviderCapabilities},
};

use crate::inference::GenerationParams;
use crate::model::resolve_device;
use crate::models::{LocalModel, ModelDescriptor};

/// Model source: either download from HuggingFace Hub or load from local files.
enum ModelSource {
    /// Auto-download from HF Hub by model name
    Hub { model_name: String },
    /// Load from local filesystem
    Local {
        model_name: String,
        model_path: PathBuf,
        tokenizer_path: PathBuf,
    },
}

/// Local inference provider backed by candle.
/// Supports Phi-3, Llama-3, Qwen2, and Qwen3 via GGUF quantized format.
pub struct LocalProvider {
    model: Arc<Mutex<Option<LocalModel>>>,
    source: ModelSource,
    use_gpu: bool,
    generation_params: GenerationParams,
}

impl LocalProvider {
    /// Create a new local provider that auto-downloads from HuggingFace Hub.
    ///
    /// Available models: `"phi3"`, `"llama3"`, `"qwen2"`, `"qwen3"`
    ///
    /// First call will download the model (~1-5 GB) and cache it in `~/.cache/huggingface/`.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            source: ModelSource::Hub {
                model_name: model_name.into(),
            },
            use_gpu: false,
            generation_params: GenerationParams::default(),
        }
    }

    /// Create a new local provider from local model files (no download needed).
    ///
    /// - `model_name`: one of `"phi3"`, `"llama3"`, `"qwen2"`, `"qwen3"` (determines architecture)
    /// - `model_path`: path to the `.gguf` model file
    /// - `tokenizer_path`: path to the `tokenizer.json` file
    pub fn from_local(
        model_name: impl Into<String>,
        model_path: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            source: ModelSource::Local {
                model_name: model_name.into(),
                model_path: model_path.into(),
                tokenizer_path: tokenizer_path.into(),
            },
            use_gpu: false,
            generation_params: GenerationParams::default(),
        }
    }

    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    pub fn with_generation_params(mut self, params: GenerationParams) -> Self {
        self.generation_params = params;
        self
    }

    /// List all available model names
    pub fn available_models() -> Vec<&'static str> {
        ModelDescriptor::available_models()
    }

    fn model_display_name(&self) -> &str {
        match &self.source {
            ModelSource::Hub { model_name } => model_name,
            ModelSource::Local { model_name, .. } => model_name,
        }
    }

    /// Ensure the model is loaded
    fn ensure_loaded(&self) -> Result<()> {
        let mut guard = self.model.lock().map_err(|e| {
            BatataError::Inference(format!("failed to lock model: {e}"))
        })?;

        if guard.is_none() {
            let device = resolve_device(self.use_gpu)?;
            let loaded = match &self.source {
                ModelSource::Hub { model_name } => {
                    let descriptor =
                        ModelDescriptor::by_name(model_name).ok_or_else(|| {
                            BatataError::ModelNotFound(format!(
                                "unknown model '{model_name}', available: {:?}",
                                ModelDescriptor::available_models()
                            ))
                        })?;
                    LocalModel::download_and_load(descriptor, &device)?
                }
                ModelSource::Local {
                    model_name,
                    model_path,
                    tokenizer_path,
                } => {
                    let descriptor =
                        ModelDescriptor::by_name(model_name).ok_or_else(|| {
                            BatataError::ModelNotFound(format!(
                                "unknown model architecture '{model_name}', available: {:?}",
                                ModelDescriptor::available_models()
                            ))
                        })?;
                    LocalModel::load(descriptor, model_path, tokenizer_path, &device)?
                }
            };
            *guard = Some(loaded);
        }

        Ok(())
    }

    fn build_messages(req: &ChatRequest) -> Vec<(String, String)> {
        req.messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "user",
                };
                (role.to_string(), m.content.clone())
            })
            .collect()
    }

    fn build_params(&self, req: &ChatRequest) -> GenerationParams {
        let mut params = self.generation_params.clone();
        if let Some(temp) = req.temperature {
            params.temperature = temp as f64;
        }
        if let Some(max) = req.max_tokens {
            params.max_tokens = max as usize;
        }
        params
    }
}

#[async_trait]
impl Provider for LocalProvider {
    fn name(&self) -> &str {
        "local"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            chat: true,
            streaming: true,
            embeddings: false,
            function_calling: false,
        }
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        self.ensure_loaded()?;

        let messages = Self::build_messages(&req);
        let params = self.build_params(&req);

        let content = {
            let mut guard = self.model.lock().map_err(|e| {
                BatataError::Inference(format!("failed to lock model: {e}"))
            })?;
            let model = guard.as_mut().ok_or_else(|| {
                BatataError::Inference("model not loaded".to_string())
            })?;
            model.chat(&messages, &params)?
        };

        Ok(ChatResponse {
            content,
            model: self.model_display_name().to_string(),
            usage: None,
        })
    }

    async fn stream_chat(&self, req: ChatRequest) -> Result<ChatStream> {
        self.ensure_loaded()?;

        let messages = Self::build_messages(&req);
        let params = self.build_params(&req);
        let model = Arc::clone(&self.model);

        let (tx, rx) = mpsc::channel::<Result<String>>(32);

        tokio::task::spawn_blocking(move || {
            let mut guard = match model.lock() {
                Ok(g) => g,
                Err(e) => {
                    let _ = tx.blocking_send(Err(BatataError::Inference(format!(
                        "failed to lock model: {e}"
                    ))));
                    return;
                }
            };

            let model = match guard.as_mut() {
                Some(m) => m,
                None => {
                    let _ = tx.blocking_send(Err(BatataError::Inference(
                        "model not loaded".to_string(),
                    )));
                    return;
                }
            };

            let _ = model.chat_stream(&messages, &params, |token_text| {
                let _ = tx.blocking_send(Ok(token_text.to_string()));
            });
        });

        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}
