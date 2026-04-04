use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::Prompt;

use batata_ai_core::{
    error::{BatataError, Result},
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

pub struct AnthropicProvider {
    client: rig::providers::anthropic::Client,
    default_model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let client = rig::providers::anthropic::Client::new(&api_key.into())
            .map_err(|e| BatataError::Provider(e.to_string()))?;
        Ok(Self {
            client,
            default_model: "claude-sonnet-4-6-20260404".to_string(),
        })
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
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
