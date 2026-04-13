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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ChatRequest, Message};

    fn make_request(model: Option<&str>, messages: Vec<Message>, temperature: Option<f32>) -> ChatRequest {
        ChatRequest {
            model: model.map(|s| s.to_string()),
            messages,
            temperature,
            max_tokens: None,
        }
    }

    #[test]
    fn same_request_produces_same_key() {
        let strategy = DefaultCacheKeyStrategy;
        let req = make_request(
            Some("gpt-4"),
            vec![Message::user("hello")],
            Some(0.7),
        );
        let key1 = strategy.generate_key(&req);
        let key2 = strategy.generate_key(&req);
        assert_eq!(key1, key2);
    }

    #[test]
    fn different_models_produce_different_keys() {
        let strategy = DefaultCacheKeyStrategy;
        let req1 = make_request(Some("gpt-4"), vec![Message::user("hello")], None);
        let req2 = make_request(Some("gpt-3.5"), vec![Message::user("hello")], None);
        assert_ne!(strategy.generate_key(&req1), strategy.generate_key(&req2));
    }

    #[test]
    fn different_messages_produce_different_keys() {
        let strategy = DefaultCacheKeyStrategy;
        let req1 = make_request(Some("gpt-4"), vec![Message::user("hello")], None);
        let req2 = make_request(Some("gpt-4"), vec![Message::user("world")], None);
        assert_ne!(strategy.generate_key(&req1), strategy.generate_key(&req2));
    }

    #[test]
    fn different_temperatures_produce_different_keys() {
        let strategy = DefaultCacheKeyStrategy;
        let req1 = make_request(Some("gpt-4"), vec![Message::user("hello")], Some(0.5));
        let req2 = make_request(Some("gpt-4"), vec![Message::user("hello")], Some(0.9));
        assert_ne!(strategy.generate_key(&req1), strategy.generate_key(&req2));
    }
}
