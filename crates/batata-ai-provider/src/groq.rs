use async_trait::async_trait;
use reqwest::header::HeaderMap;

use batata_ai_core::{
    error::Result,
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

use crate::openai_compat::{self, OpenAiCompatConfig};

// -- Popular model constants ──────────────────────────────────────

pub const LLAMA_3_3_70B: &str = "llama-3.3-70b-versatile";
pub const LLAMA_3_1_8B: &str = "llama-3.1-8b-instant";
pub const LLAMA_GUARD_3_8B: &str = "llama-guard-3-8b";
pub const MIXTRAL_8X7B: &str = "mixtral-8x7b-32768";
pub const GEMMA_2_9B: &str = "gemma2-9b-it";
pub const QWEN_QWQ_32B: &str = "qwen-qwq-32b";
pub const DEEPSEEK_R1_70B: &str = "deepseek-r1-distill-llama-70b";
pub const META_LLAMA_4_SCOUT: &str = "meta-llama/llama-4-scout-17b-16e-instruct";

/// List popular models available on Groq.
pub fn popular_models() -> Vec<&'static str> {
    vec![
        LLAMA_3_3_70B,
        LLAMA_3_1_8B,
        MIXTRAL_8X7B,
        GEMMA_2_9B,
        QWEN_QWQ_32B,
        DEEPSEEK_R1_70B,
        META_LLAMA_4_SCOUT,
    ]
}

// -- Provider ─────────────────────────────────────────────────────

pub struct GroqProvider {
    config: OpenAiCompatConfig,
}

impl GroqProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: OpenAiCompatConfig {
                base_url: "https://api.groq.com/openai/v1".to_string(),
                api_key: api_key.into(),
                default_model: LLAMA_3_3_70B.to_string(),
                extra_headers: HeaderMap::new(),
            },
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for GroqProvider {
    fn name(&self) -> &str {
        "groq"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            chat: true,
            streaming: true,
            embeddings: false,
            function_calling: true,
        }
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        openai_compat::chat(&self.config, req).await
    }

    async fn stream_chat(&self, req: ChatRequest) -> Result<ChatStream> {
        openai_compat::stream_chat(&self.config, req).await
    }
}
