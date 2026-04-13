use async_trait::async_trait;
use redis::AsyncCommands;

use batata_ai_core::cache::{CacheEntry, CacheStats, CacheStore};
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::message::ChatResponse;

const STATS_HITS_KEY: &str = "cache:stats:hits";
const STATS_MISSES_KEY: &str = "cache:stats:misses";
const ENTRIES_SET_KEY: &str = "cache:entries";

/// Redis-backed cache implementation.
pub struct RedisCache {
    client: redis::Client,
}

impl RedisCache {
    /// Create a new `RedisCache` from a Redis connection URL
    /// (e.g. `redis://127.0.0.1:6379`).
    pub fn new(url: &str) -> Result<Self> {
        let client =
            redis::Client::open(url).map_err(|e| BatataError::Config(e.to_string()))?;
        Ok(Self { client })
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))
    }
}

#[async_trait]
impl CacheStore for RedisCache {
    async fn get(&self, key: &str) -> Result<Option<CacheEntry>> {
        let mut conn = self.conn().await?;

        let raw: Option<String> = conn
            .get(key)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        let Some(data) = raw else {
            // Record a miss.
            let _: () = conn
                .incr(STATS_MISSES_KEY, 1u64)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
            return Ok(None);
        };

        let mut entry: CacheEntry = serde_json::from_str(&data)?;

        // Increment hit_count on the entry and persist it back.
        entry.hit_count += 1;
        let updated = serde_json::to_string(&entry)?;

        // Preserve existing TTL when updating the value.
        let ttl: i64 = conn
            .ttl(key)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        if ttl > 0 {
            let _: () = conn
                .set_ex(key, &updated, ttl as u64)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
        } else {
            let _: () = conn
                .set(key, &updated)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
        }

        // Record a hit.
        let _: () = conn
            .incr(STATS_HITS_KEY, 1u64)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        Ok(Some(entry))
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

        let data = serde_json::to_string(&entry)?;
        let mut conn = self.conn().await?;

        if let Some(ttl) = ttl_secs {
            let _: () = conn
                .set_ex(key, &data, ttl)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
        } else {
            let _: () = conn
                .set(key, &data)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
        }

        // Track the key in a set so we can enumerate entries later.
        let _: () = conn
            .sadd(ENTRIES_SET_KEY, key)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn remove(&self, key: &str) -> Result<()> {
        let mut conn = self.conn().await?;

        let _: () = conn
            .del(key)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        let _: () = conn
            .srem(ENTRIES_SET_KEY, key)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut conn = self.conn().await?;

        // Retrieve all tracked entry keys and delete them.
        let keys: Vec<String> = conn
            .smembers(ENTRIES_SET_KEY)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        if !keys.is_empty() {
            let _: () = conn
                .del(&keys)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
        }

        // Remove the tracking set and stats counters.
        let _: () = conn
            .del(&[ENTRIES_SET_KEY, STATS_HITS_KEY, STATS_MISSES_KEY])
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats> {
        let mut conn = self.conn().await?;

        let total_hits: u64 = conn
            .get(STATS_HITS_KEY)
            .await
            .unwrap_or(0);

        let total_misses: u64 = conn
            .get(STATS_MISSES_KEY)
            .await
            .unwrap_or(0);

        let total_entries: u64 = conn
            .scard(ENTRIES_SET_KEY)
            .await
            .unwrap_or(0);

        let total = total_hits + total_misses;
        let hit_rate = if total > 0 {
            total_hits as f64 / total as f64
        } else {
            0.0
        };

        Ok(CacheStats {
            total_entries,
            total_hits,
            total_misses,
            hit_rate,
        })
    }
}
