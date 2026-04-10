use async_trait::async_trait;
use reqwest::header::HeaderMap;

use batata_ai_core::{
    error::Result,
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

use crate::openai_compat::{self, OpenAiCompatConfig};

// -- Popular model constants ──────────────────────────────────────

pub const MISTRAL_LARGE: &str = "mistral-large-latest";
pub const MISTRAL_SMALL: &str = "mistral-small-latest";
pub const MISTRAL_MEDIUM: &str = "mistral-medium-latest";
pub const CODESTRAL: &str = "codestral-latest";
pub const MINISTRAL_8B: &str = "ministral-8b-latest";
pub const MISTRAL_NEMO: &str = "open-mistral-nemo";
pub const MIXTRAL_8X7B: &str = "open-mixtral-8x7b";
pub const MIXTRAL_8X22B: &str = "open-mixtral-8x22b";
pub const PIXTRAL_LARGE: &str = "pixtral-large-latest";

/// List popular models available on Mistral AI.
pub fn popular_models() -> Vec<&'static str> {
    vec![
        MISTRAL_LARGE,
        MISTRAL_SMALL,
        CODESTRAL,
        MINISTRAL_8B,
        MISTRAL_NEMO,
        MIXTRAL_8X7B,
        MIXTRAL_8X22B,
        PIXTRAL_LARGE,
    ]
}

// -- Provider ─────────────────────────────────────────────────────

pub struct MistralProvider {
    config: OpenAiCompatConfig,
}

impl MistralProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: OpenAiCompatConfig {
                base_url: "https://api.mistral.ai/v1".to_string(),
                api_key: api_key.into(),
                default_model: MISTRAL_SMALL.to_string(),
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
impl Provider for MistralProvider {
    fn name(&self) -> &str {
        "mistral"
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
