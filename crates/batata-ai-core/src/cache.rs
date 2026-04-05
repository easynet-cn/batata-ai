use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::message::{ChatRequest, ChatResponse};

/// Cache entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub response: ChatResponse,
    pub hit_count: u64,
    pub created_at: chrono::NaiveDateTime,
    pub expires_at: Option<chrono::NaiveDateTime>,
}

/// Cache key generation strategy.
pub trait CacheKeyStrategy: Send + Sync {
    /// Generate a cache key from a chat request.
    fn generate_key(&self, request: &ChatRequest) -> String;
}

/// Default strategy: hash model + messages content.
pub struct DefaultCacheKeyStrategy;

impl CacheKeyStrategy for DefaultCacheKeyStrategy {
    fn generate_key(&self, request: &ChatRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // Hash model
        if let Some(model) = &request.model {
            model.hash(&mut hasher);
        }
        // Hash messages
        for msg in &request.messages {
            msg.content.hash(&mut hasher);
        }
        // Hash temperature (as bits to avoid float hashing issues)
        if let Some(temp) = request.temperature {
            temp.to_bits().hash(&mut hasher);
        }
        format!("cache:{:x}", hasher.finish())
    }
}

/// Cache store trait — backend abstraction (memory, Redis, etc.)
#[async_trait]
pub trait CacheStore: Send + Sync {
    /// Get a cached response.
    async fn get(&self, key: &str) -> Result<Option<CacheEntry>>;

    /// Store a response in cache.
    async fn set(&self, key: &str, response: &ChatResponse, ttl_secs: Option<u64>) -> Result<()>;

    /// Remove a cache entry.
    async fn remove(&self, key: &str) -> Result<()>;

    /// Clear all cache entries.
    async fn clear(&self) -> Result<()>;

    /// Get cache statistics.
    async fn stats(&self) -> Result<CacheStats>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: u64,
    pub total_hits: u64,
    pub total_misses: u64,
    pub hit_rate: f64,
}
