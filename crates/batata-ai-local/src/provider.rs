use std::path::PathBuf;
use std::sync::Arc;

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
use crate::model_pool::ModelPool;
use crate::models::ModelDescriptor;

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
///
/// Uses a `ModelPool` for multi-model concurrent loading with LRU eviction.
/// Supports Phi-3, Llama-3, Qwen2, Qwen3, and Gemma3 via GGUF quantized format.
pub struct LocalProvider {
    pool: Arc<ModelPool>,
    source: ModelSource,
    generation_params: GenerationParams,
}

impl LocalProvider {
    /// Create a new local provider that auto-downloads from HuggingFace Hub.
    ///
    /// Uses a pool with capacity 1 (single model). Use `with_pool` to share a pool.
    pub fn new(model_name: impl Into<String>) -> Self {
        let device = resolve_device(false).unwrap_or(candle_core::Device::Cpu);
        Self {
            pool: Arc::new(ModelPool::new(1, device)),
            source: ModelSource::Hub {
                model_name: model_name.into(),
            },
            generation_params: GenerationParams::default(),
        }
    }

    /// Create a new local provider from local model files (no download needed).
    pub fn from_local(
        model_name: impl Into<String>,
        model_path: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
    ) -> Self {
        let device = resolve_device(false).unwrap_or(candle_core::Device::Cpu);
        Self {
            pool: Arc::new(ModelPool::new(1, device)),
            source: ModelSource::Local {
                model_name: model_name.into(),
                model_path: model_path.into(),
                tokenizer_path: tokenizer_path.into(),
            },
            generation_params: GenerationParams::default(),
        }
    }

    pub fn with_gpu(self, use_gpu: bool) -> Self {
        let device = resolve_device(use_gpu).unwrap_or(candle_core::Device::Cpu);
        Self {
            pool: Arc::new(ModelPool::new(
                self.pool.loaded_count().max(1),
                device,
            )),
            ..self
        }
    }

    /// Use a shared `ModelPool` (for multi-model scenarios).
    ///
    /// This allows multiple `LocalProvider` instances to share the same pool,
    /// enabling concurrent loading of different models with LRU eviction.
    pub fn with_pool(mut self, pool: Arc<ModelPool>) -> Self {
        self.pool = pool;
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

    /// Get the underlying model pool (for sharing with other providers).
    pub fn pool(&self) -> &Arc<ModelPool> {
        &self.pool
    }

    fn model_name(&self) -> &str {
        match &self.source {
            ModelSource::Hub { model_name } => model_name,
            ModelSource::Local { model_name, .. } => model_name,
        }
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
        let messages = Self::build_messages(&req);
        let params = self.build_params(&req);
        let model_name = self.model_name().to_string();

        let content = match &self.source {
            ModelSource::Hub { model_name } => {
                let name = model_name.clone();
                self.pool
                    .with_model(&name, |model| model.chat(&messages, &params))?
            }
            ModelSource::Local {
                model_name,
                model_path,
                tokenizer_path,
            } => self.pool.load_local(
                model_name,
                model_path,
                tokenizer_path,
                |model| model.chat(&messages, &params),
            )?,
        };

        Ok(ChatResponse {
            content,
            model: model_name,
            usage: None,
        })
    }

    async fn stream_chat(&self, req: ChatRequest) -> Result<ChatStream> {
        let messages = Self::build_messages(&req);
        let params = self.build_params(&req);
        let pool = Arc::clone(&self.pool);
        let model_name = self.model_name().to_string();
        let source_is_hub = matches!(&self.source, ModelSource::Hub { .. });
        let model_path = match &self.source {
            ModelSource::Local {
                model_path,
                tokenizer_path,
                ..
            } => Some((model_path.clone(), tokenizer_path.clone())),
            _ => None,
        };

        let (tx, rx) = mpsc::channel::<Result<String>>(32);

        tokio::task::spawn_blocking(move || {
            let result = if source_is_hub {
                pool.with_model(&model_name, |model| {
                    model.chat_stream(&messages, &params, |token_text| {
                        let _ = tx.blocking_send(Ok(token_text.to_string()));
                    })
                })
            } else if let Some((mp, tp)) = &model_path {
                pool.load_local(&model_name, mp, tp, |model| {
                    model.chat_stream(&messages, &params, |token_text| {
                        let _ = tx.blocking_send(Ok(token_text.to_string()));
                    })
                })
            } else {
                Err(BatataError::Inference("invalid model source".to_string()))
            };

            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e));
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}
