use async_trait::async_trait;
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub event_type: EventType,
    pub tenant_id: Option<String>,
    pub payload: serde_json::Value,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    ChatCompleted,
    ChatFailed,
    ConversationCreated,
    ConversationDeleted,
    ModelEnabled,
    ModelDisabled,
    ProviderHealthChanged,
    QuotaExceeded,
    ApiKeyCreated,
    ApiKeyExpired,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChatCompleted => write!(f, "chat.completed"),
            Self::ChatFailed => write!(f, "chat.failed"),
            Self::ConversationCreated => write!(f, "conversation.created"),
            Self::ConversationDeleted => write!(f, "conversation.deleted"),
            Self::ModelEnabled => write!(f, "model.enabled"),
            Self::ModelDisabled => write!(f, "model.disabled"),
            Self::ProviderHealthChanged => write!(f, "provider.health_changed"),
            Self::QuotaExceeded => write!(f, "quota.exceeded"),
            Self::ApiKeyCreated => write!(f, "api_key.created"),
            Self::ApiKeyExpired => write!(f, "api_key.expired"),
        }
    }
}

/// Event subscriber trait.
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Handle an event. Implementations should not block.
    async fn handle(&self, event: &Event) -> Result<()>;
}

/// Event bus for publishing and subscribing to events.
pub struct EventBus {
    handlers: Vec<Box<dyn EventHandler>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self { handlers: vec![] }
    }

    pub fn subscribe(mut self, handler: Box<dyn EventHandler>) -> Self {
        self.handlers.push(handler);
        self
    }

    /// Publish an event to all subscribers (fire-and-forget).
    pub async fn publish(&self, event: &Event) {
        for handler in &self.handlers {
            if let Err(e) = handler.handle(event).await {
                tracing::warn!(
                    event_type = %event.event_type,
                    error = %e,
                    "event handler failed"
                );
            }
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
