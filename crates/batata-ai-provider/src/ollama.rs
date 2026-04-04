use async_trait::async_trait;
use batata_ai_core::{
    error::{BatataError, Result},
    message::{ChatRequest, ChatResponse, ChatStream},
    provider::{Provider, ProviderCapabilities},
};

pub struct OllamaProvider {
    base_url: String,
    default_model: String,
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            default_model: "llama3".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            chat: true,
            streaming: true,
            embeddings: true,
            function_calling: false,
        }
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        let model = req
            .model
            .as_deref()
            .unwrap_or(&self.default_model)
            .to_string();

        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
        });

        if let Some(temp) = req.temperature {
            body["options"] = serde_json::json!({ "temperature": temp });
        }

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| BatataError::Provider(e.to_string()))?;

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| BatataError::Provider(e.to_string()))?;

        let content = data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(ChatResponse {
            content,
            model,
            usage: None,
        })
    }

    async fn stream_chat(&self, req: ChatRequest) -> Result<ChatStream> {
        // TODO: Implement Ollama streaming
        let response = self.chat(req).await?;
        let stream = futures::stream::once(async move { Ok(response.content) });
        Ok(Box::pin(stream))
    }
}
