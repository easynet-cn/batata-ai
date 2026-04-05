use async_trait::async_trait;
use batata_ai_core::error::Result;
use batata_ai_core::event::{Event, EventHandler};

/// Webhook event handler that POSTs events to a configured URL.
pub struct WebhookHandler {
    url: String,
    secret: Option<String>,
    client: reqwest::Client,
}

impl WebhookHandler {
    pub fn new(url: impl Into<String>, secret: Option<String>) -> Self {
        Self {
            url: url.into(),
            secret,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EventHandler for WebhookHandler {
    async fn handle(&self, event: &Event) -> Result<()> {
        let mut request = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("X-Event-Type", event.event_type.to_string());

        // Add HMAC signature if secret is configured
        if let Some(secret) = &self.secret {
            let payload = serde_json::to_string(event)
                .map_err(|e| batata_ai_core::error::BatataError::Storage(e.to_string()))?;
            // Simple signature: SHA-256 HMAC of payload
            use hmac::{Hmac, Mac};
            use sha2::Sha256;
            let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
                .map_err(|e| batata_ai_core::error::BatataError::Storage(e.to_string()))?;
            mac.update(payload.as_bytes());
            let signature = hex::encode(mac.finalize().into_bytes());
            request = request
                .header("X-Signature-256", format!("sha256={}", signature))
                .body(payload);
        } else {
            request = request.json(event);
        }

        request
            .send()
            .await
            .map_err(|e| batata_ai_core::error::BatataError::Storage(e.to_string()))?;

        Ok(())
    }
}

/// Logging event handler that logs events via tracing.
pub struct LoggingEventHandler;

#[async_trait]
impl EventHandler for LoggingEventHandler {
    async fn handle(&self, event: &Event) -> Result<()> {
        tracing::info!(
            event_id = %event.id,
            event_type = %event.event_type,
            tenant_id = ?event.tenant_id,
            "event published"
        );
        Ok(())
    }
}
