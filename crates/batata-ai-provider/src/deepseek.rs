use async_trait::async_trait;
use reqwest::header::HeaderMap;

use batata_ai_core::{
    error::Result,
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

use crate::openai_compat::{self, OpenAiCompatConfig};

pub struct DeepSeekProvider {
    config: OpenAiCompatConfig,
}

impl DeepSeekProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: OpenAiCompatConfig {
                base_url: "https://api.deepseek.com/v1".to_string(),
                api_key: api_key.into(),
                default_model: "deepseek-chat".to_string(),
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
impl Provider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
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
