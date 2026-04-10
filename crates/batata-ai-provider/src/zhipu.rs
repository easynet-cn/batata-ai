use async_trait::async_trait;
use reqwest::header::HeaderMap;

use batata_ai_core::{
    error::Result,
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

use crate::openai_compat::{self, OpenAiCompatConfig};

// -- Popular model constants ──────────────────────────────────────

pub const GLM_4_PLUS: &str = "glm-4-plus";
pub const GLM_4_AIR: &str = "glm-4-air";
pub const GLM_4_AIRX: &str = "glm-4-airx";
pub const GLM_4_LONG: &str = "glm-4-long";
pub const GLM_4_FLASH: &str = "glm-4-flash";
pub const GLM_4_FLASHX: &str = "glm-4-flashx";
pub const GLM_4V_PLUS: &str = "glm-4v-plus";
pub const CODEGEEX_4: &str = "codegeex-4";

/// List popular models available on Zhipu AI.
pub fn popular_models() -> Vec<&'static str> {
    vec![
        GLM_4_PLUS,
        GLM_4_AIR,
        GLM_4_AIRX,
        GLM_4_LONG,
        GLM_4_FLASH,
        GLM_4_FLASHX,
        GLM_4V_PLUS,
        CODEGEEX_4,
    ]
}

// -- Provider ─────────────────────────────────────────────────────

pub struct ZhipuProvider {
    config: OpenAiCompatConfig,
}

impl ZhipuProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: OpenAiCompatConfig {
                base_url: "https://open.bigmodel.cn/api/paas/v4".to_string(),
                api_key: api_key.into(),
                default_model: GLM_4_FLASH.to_string(),
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
impl Provider for ZhipuProvider {
    fn name(&self) -> &str {
        "zhipu"
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
