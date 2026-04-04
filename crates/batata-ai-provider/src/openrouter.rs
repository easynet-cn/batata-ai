use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::Prompt;

use batata_ai_core::{
    error::{BatataError, Result},
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

// ── Free model constants ─────────────────────────────────────────

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

/// List all available free model IDs on OpenRouter
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

// ── Provider ─────────────────────────────────────────────────────

pub struct OpenRouterProvider {
    client: rig::providers::openrouter::Client,
    default_model: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = rig::providers::openrouter::Client::new(&api_key.into())
            .map_err(|e| BatataError::Provider(e.to_string()))?;
        Ok(Self {
            client,
            default_model: QWEN3_6_PLUS_FREE.to_string(),
        })
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
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
        let model = req.model.as_deref().unwrap_or(&self.default_model);
        let agent = self.client.agent(model).build();

        let last_user_msg = req
            .messages
            .iter()
            .rfind(|m| matches!(m.role, batata_ai_core::message::Role::User))
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let response: String = agent
            .prompt(last_user_msg)
            .await
            .map_err(|e| BatataError::Provider(e.to_string()))?;

        Ok(ChatResponse {
            content: response,
            model: model.to_string(),
            usage: None,
        })
    }

    async fn stream_chat(&self, req: ChatRequest) -> Result<ChatStream> {
        let response = self.chat(req).await?;
        let stream = futures::stream::once(async move { Ok(response.content) });
        Ok(Box::pin(stream))
    }
}
