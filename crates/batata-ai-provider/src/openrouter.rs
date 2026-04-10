use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};

use batata_ai_core::{
    error::Result,
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

use crate::openai_compat::{self, OpenAiCompatConfig};

// -- Free model constants ─────────────────────────────────────────

pub const GEMMA_3_4B_FREE: &str = "google/gemma-3-4b-it:free";
pub const GEMMA_3_12B_FREE: &str = "google/gemma-3-12b-it:free";
pub const GEMMA_3_27B_FREE: &str = "google/gemma-3-27b-it:free";
pub const GEMMA_3N_E2B_FREE: &str = "google/gemma-3n-e2b-it:free";
pub const GEMMA_3N_E4B_FREE: &str = "google/gemma-3n-e4b-it:free";
pub const LLAMA_3_2_3B_FREE: &str = "meta-llama/llama-3.2-3b-instruct:free";
pub const LLAMA_3_3_70B_FREE: &str = "meta-llama/llama-3.3-70b-instruct:free";
pub const QWEN3_CODER_FREE: &str = "qwen/qwen3-coder:free";
pub const QWEN3_NEXT_80B_FREE: &str = "qwen/qwen3-next-80b-a3b-instruct:free";
pub const QWEN3_6_PLUS_FREE: &str = "qwen/qwen3.6-plus:free";
pub const GPT_OSS_120B_FREE: &str = "openai/gpt-oss-120b:free";
pub const GPT_OSS_20B_FREE: &str = "openai/gpt-oss-20b:free";
pub const NEMOTRON_NANO_9B_FREE: &str = "nvidia/nemotron-nano-9b-v2:free";
pub const NEMOTRON_SUPER_120B_FREE: &str = "nvidia/nemotron-3-super-120b-a12b:free";
pub const HERMES_3_405B_FREE: &str = "nousresearch/hermes-3-llama-3.1-405b:free";
pub const DOLPHIN_MISTRAL_24B_FREE: &str =
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free";
pub const LFM_1_2B_FREE: &str = "liquid/lfm-2.5-1.2b-instruct:free";
pub const MINIMAX_M2_5_FREE: &str = "minimax/minimax-m2.5:free";
pub const STEP_3_5_FLASH_FREE: &str = "stepfun/step-3.5-flash:free";
pub const GLM_4_5_AIR_FREE: &str = "z-ai/glm-4.5-air:free";
pub const OPENROUTER_FREE: &str = "openrouter/free";

/// List all available free model IDs on OpenRouter.
pub fn free_models() -> Vec<&'static str> {
    vec![
        GEMMA_3_4B_FREE,
        GEMMA_3_12B_FREE,
        GEMMA_3_27B_FREE,
        GEMMA_3N_E2B_FREE,
        GEMMA_3N_E4B_FREE,
        LLAMA_3_2_3B_FREE,
        LLAMA_3_3_70B_FREE,
        QWEN3_CODER_FREE,
        QWEN3_NEXT_80B_FREE,
        QWEN3_6_PLUS_FREE,
        GPT_OSS_120B_FREE,
        GPT_OSS_20B_FREE,
        NEMOTRON_NANO_9B_FREE,
        NEMOTRON_SUPER_120B_FREE,
        HERMES_3_405B_FREE,
        DOLPHIN_MISTRAL_24B_FREE,
        LFM_1_2B_FREE,
        MINIMAX_M2_5_FREE,
        STEP_3_5_FLASH_FREE,
        GLM_4_5_AIR_FREE,
        OPENROUTER_FREE,
    ]
}

// -- Provider ─────────────────────────────────────────────────────

pub struct OpenRouterProvider {
    config: OpenAiCompatConfig,
}

impl OpenRouterProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let mut extra_headers = HeaderMap::new();
        extra_headers.insert(
            "HTTP-Referer",
            HeaderValue::from_static("https://github.com/easynet-cn/batata-ai"),
        );
        extra_headers.insert("X-Title", HeaderValue::from_static("batata-ai"));

        Self {
            config: OpenAiCompatConfig {
                base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key: api_key.into(),
                default_model: QWEN3_6_PLUS_FREE.to_string(),
                extra_headers,
            },
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
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
