use async_trait::async_trait;
use reqwest::header::HeaderMap;

use batata_ai_core::{
    error::Result,
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

use crate::openai_compat::{self, OpenAiCompatConfig};

// -- Popular model constants ──────────────────────────────────────

pub const LLAMA_3_3_70B: &str = "meta-llama/Llama-3.3-70B-Instruct-Turbo";
pub const LLAMA_3_1_405B: &str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo";
pub const LLAMA_3_1_8B: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo";
pub const QWEN_2_5_72B: &str = "Qwen/Qwen2.5-72B-Instruct-Turbo";
pub const QWEN_2_5_CODER_32B: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
pub const DEEPSEEK_V3: &str = "deepseek-ai/DeepSeek-V3";
pub const DEEPSEEK_R1: &str = "deepseek-ai/DeepSeek-R1";
pub const MISTRAL_SMALL: &str = "mistralai/Mistral-Small-24B-Instruct-2501";
pub const GEMMA_2_27B: &str = "google/gemma-2-27b-it";

/// List popular models available on Together AI.
pub fn popular_models() -> Vec<&'static str> {
    vec![
        LLAMA_3_3_70B,
        LLAMA_3_1_405B,
        LLAMA_3_1_8B,
        QWEN_2_5_72B,
        QWEN_2_5_CODER_32B,
        DEEPSEEK_V3,
        DEEPSEEK_R1,
        MISTRAL_SMALL,
        GEMMA_2_27B,
    ]
}

// -- Provider ─────────────────────────────────────────────────────

pub struct TogetherProvider {
    config: OpenAiCompatConfig,
}

impl TogetherProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: OpenAiCompatConfig {
                base_url: "https://api.together.xyz/v1".to_string(),
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
impl Provider for TogetherProvider {
    fn name(&self) -> &str {
        "together"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            chat: true,
            streaming: true,
            embeddings: true,
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
