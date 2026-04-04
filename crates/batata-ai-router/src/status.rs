use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use batata_ai_core::error::Result;
use batata_ai_core::routing::{ProviderStatus, StatusStore};

/// Composite key for provider+model pair.
fn status_key(provider_id: &str, model_identifier: &str) -> String {
    format!("{}:{}", provider_id, model_identifier)
}

/// In-memory status store backed by a concurrent HashMap.
pub struct InMemoryStatusStore {
    data: Arc<RwLock<HashMap<String, ProviderStatus>>>,
}

impl InMemoryStatusStore {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryStatusStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StatusStore for InMemoryStatusStore {
    async fn get(
        &self,
        provider_id: &str,
        model_identifier: &str,
    ) -> Result<Option<ProviderStatus>> {
        let data = self.data.read().await;
        Ok(data.get(&status_key(provider_id, model_identifier)).cloned())
    }

    async fn set(&self, status: &ProviderStatus, _ttl_secs: Option<u64>) -> Result<()> {
        let key = status_key(&status.provider_id, &status.model_identifier);
        let mut data = self.data.write().await;
        data.insert(key, status.clone());
        Ok(())
    }

    async fn remove(&self, provider_id: &str, model_identifier: &str) -> Result<()> {
        let mut data = self.data.write().await;
        data.remove(&status_key(provider_id, model_identifier));
        Ok(())
    }

    async fn list_unhealthy(&self) -> Result<Vec<ProviderStatus>> {
        let data = self.data.read().await;
        Ok(data.values().filter(|s| !s.healthy).cloned().collect())
    }

    async fn list_all(&self) -> Result<Vec<ProviderStatus>> {
        let data = self.data.read().await;
        Ok(data.values().cloned().collect())
    }
}
