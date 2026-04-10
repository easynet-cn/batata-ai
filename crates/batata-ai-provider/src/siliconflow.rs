use async_trait::async_trait;
use reqwest::header::HeaderMap;

use batata_ai_core::{
    error::Result,
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

use crate::openai_compat::{self, OpenAiCompatConfig};

// -- Popular model constants ──────────────────────────────────────

pub const DEEPSEEK_V3: &str = "deepseek-ai/DeepSeek-V3";
pub const DEEPSEEK_R1: &str = "deepseek-ai/DeepSeek-R1";
pub const QWEN_2_5_72B: &str = "Qwen/Qwen2.5-72B-Instruct";
pub const QWEN_2_5_CODER_32B: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
pub const QWEN_2_5_14B: &str = "Qwen/Qwen2.5-14B-Instruct";
pub const QWEN_2_5_7B: &str = "Qwen/Qwen2.5-7B-Instruct";
pub const GLM_4_9B: &str = "THUDM/glm-4-9b-chat";
pub const YI_1_5_34B: &str = "01-ai/Yi-1.5-34B-Chat-16K";
pub const INTERNLM_2_5_20B: &str = "internlm/internlm2_5-20b-chat";

/// List popular models available on SiliconFlow.
pub fn popular_models() -> Vec<&'static str> {
    vec![
        DEEPSEEK_V3,
        DEEPSEEK_R1,
        QWEN_2_5_72B,
        QWEN_2_5_CODER_32B,
        QWEN_2_5_14B,
        QWEN_2_5_7B,
        GLM_4_9B,
        YI_1_5_34B,
        INTERNLM_2_5_20B,
    ]
}

// -- Provider ─────────────────────────────────────────────────────

pub struct SiliconFlowProvider {
    config: OpenAiCompatConfig,
}

impl SiliconFlowProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            config: OpenAiCompatConfig {
                base_url: "https://api.siliconflow.cn/v1".to_string(),
                api_key: api_key.into(),
                default_model: DEEPSEEK_V3.to_string(),
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
impl Provider for SiliconFlowProvider {
    fn name(&self) -> &str {
        "siliconflow"
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
