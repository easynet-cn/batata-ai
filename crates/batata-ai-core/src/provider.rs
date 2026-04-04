use async_trait::async_trait;

use crate::error::Result;
use crate::message::{ChatRequest, ChatResponse, ChatStream};

#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    pub chat: bool,
    pub streaming: bool,
    pub embeddings: bool,
    pub function_calling: bool,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;

    fn capabilities(&self) -> ProviderCapabilities;

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse>;

    async fn stream_chat(&self, req: ChatRequest) -> Result<ChatStream>;

    async fn embeddings(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let _ = input;
        Err(crate::error::BatataError::Provider(
            "embeddings not supported by this provider".into(),
        ))
    }
}

/// Registry that manages multiple providers
pub struct ProviderRegistry {
    providers: std::collections::HashMap<String, Box<dyn Provider>>,
    default_provider: Option<String>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: std::collections::HashMap::new(),
            default_provider: None,
        }
    }

    pub fn register(&mut self, provider: Box<dyn Provider>) {
        let name = provider.name().to_string();
        self.providers.insert(name, provider);
    }

    pub fn set_default(&mut self, name: impl Into<String>) {
        self.default_provider = Some(name.into());
    }

    pub fn get(&self, name: &str) -> Option<&dyn Provider> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    pub fn default_provider(&self) -> Option<&dyn Provider> {
        self.default_provider
            .as_ref()
            .and_then(|name| self.get(name))
    }

    pub fn list(&self) -> Vec<&str> {
        self.providers.keys().map(|k| k.as_str()).collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}
