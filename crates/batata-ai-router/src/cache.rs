use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::sync::RwLock;

use batata_ai_core::cache::{CacheEntry, CacheStats, CacheStore};
use batata_ai_core::error::Result;
use batata_ai_core::message::ChatResponse;

/// In-memory cache implementation backed by a `HashMap` behind a `RwLock`.
pub struct InMemoryCache {
    store: RwLock<HashMap<String, CacheEntry>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl InMemoryCache {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CacheStore for InMemoryCache {
    async fn get(&self, key: &str) -> Result<Option<CacheEntry>> {
        let now = chrono::Utc::now().naive_utc();
        let mut store = self.store.write().await;

        if let Some(entry) = store.get_mut(key) {
            // Lazy TTL eviction: remove expired entries on access
            if let Some(expires_at) = entry.expires_at {
                if now > expires_at {
                    store.remove(key);
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    return Ok(None);
                }
            }
            entry.hit_count += 1;
            let result = entry.clone();
            self.hits.fetch_add(1, Ordering::Relaxed);
            Ok(Some(result))
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }

    async fn set(&self, key: &str, response: &ChatResponse, ttl_secs: Option<u64>) -> Result<()> {
        let now = chrono::Utc::now().naive_utc();
        let expires_at =
            ttl_secs.map(|secs| now + chrono::Duration::seconds(secs as i64));

        let entry = CacheEntry {
            key: key.to_string(),
            response: response.clone(),
            hit_count: 0,
            created_at: now,
            expires_at,
        };

        let mut store = self.store.write().await;
        store.insert(key.to_string(), entry);
        Ok(())
    }

    async fn remove(&self, key: &str) -> Result<()> {
        let mut store = self.store.write().await;
        store.remove(key);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut store = self.store.write().await;
        store.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats> {
        let store = self.store.read().await;
        let total_hits = self.hits.load(Ordering::Relaxed);
        let total_misses = self.misses.load(Ordering::Relaxed);
        let total = total_hits + total_misses;
        let hit_rate = if total > 0 {
            total_hits as f64 / total as f64
        } else {
            0.0
        };

        Ok(CacheStats {
            total_entries: store.len() as u64,
            total_hits,
            total_misses,
            hit_rate,
        })
    }
}
