//! ModelPool — manages multiple loaded local models with LRU eviction.
//!
//! Key features:
//! - Multiple models loaded concurrently, keyed by model name
//! - LRU eviction when pool exceeds capacity
//! - Thread-safe via `Mutex`
//! - Configurable max capacity (number of models)

use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::Device;
use tracing::info;

use batata_ai_core::error::{BatataError, Result};

use crate::models::{LocalModel, ModelDescriptor};

/// Entry in the model pool, tracking last access time for LRU.
struct PoolEntry {
    model: LocalModel,
    last_accessed: std::time::Instant,
}

/// A pool of loaded local models with LRU eviction.
pub struct ModelPool {
    entries: Mutex<HashMap<String, PoolEntry>>,
    max_capacity: usize,
    device: Device,
}

impl ModelPool {
    /// Create a new model pool.
    ///
    /// - `max_capacity`: maximum number of models to keep loaded simultaneously.
    ///   When exceeded, the least recently used model is evicted.
    /// - `device`: candle device (CPU/CUDA/Metal) for all models in the pool.
    pub fn new(max_capacity: usize, device: Device) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_capacity: max_capacity.max(1),
            device,
        }
    }

    /// Get or load a model by name, executing `f` with a mutable reference to it.
    ///
    /// If the model is already loaded, updates its LRU timestamp and returns it.
    /// If not loaded, loads it (evicting LRU if at capacity) and then returns it.
    pub fn with_model<F, R>(&self, model_name: &str, f: F) -> Result<R>
    where
        F: FnOnce(&mut LocalModel) -> Result<R>,
    {
        let mut entries = self.entries.lock().map_err(|e| {
            BatataError::Inference(format!("failed to lock model pool: {e}"))
        })?;

        // If already loaded, update access time and use it
        if let Some(entry) = entries.get_mut(model_name) {
            entry.last_accessed = std::time::Instant::now();
            return f(&mut entry.model);
        }

        // Need to load — evict LRU if at capacity
        if entries.len() >= self.max_capacity {
            let lru_key = entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone());

            if let Some(key) = lru_key {
                info!(evicted = %key, "evicting LRU model from pool");
                entries.remove(&key);
            }
        }

        // Load the model
        let descriptor = ModelDescriptor::by_name(model_name).ok_or_else(|| {
            BatataError::ModelNotFound(format!(
                "unknown model '{model_name}', available: {:?}",
                ModelDescriptor::available_models()
            ))
        })?;

        info!(model = %model_name, "loading model into pool");
        let model = LocalModel::download_and_load(descriptor, &self.device)?;

        let entry = entries.entry(model_name.to_string()).or_insert(PoolEntry {
            model,
            last_accessed: std::time::Instant::now(),
        });

        f(&mut entry.model)
    }

    /// Load a model from local files into the pool.
    pub fn load_local<F, R>(
        &self,
        model_name: &str,
        model_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        f: F,
    ) -> Result<R>
    where
        F: FnOnce(&mut LocalModel) -> Result<R>,
    {
        let mut entries = self.entries.lock().map_err(|e| {
            BatataError::Inference(format!("failed to lock model pool: {e}"))
        })?;

        if let Some(entry) = entries.get_mut(model_name) {
            entry.last_accessed = std::time::Instant::now();
            return f(&mut entry.model);
        }

        if entries.len() >= self.max_capacity {
            let lru_key = entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone());

            if let Some(key) = lru_key {
                info!(evicted = %key, "evicting LRU model from pool");
                entries.remove(&key);
            }
        }

        let descriptor = ModelDescriptor::by_name(model_name).ok_or_else(|| {
            BatataError::ModelNotFound(format!(
                "unknown model architecture '{model_name}'"
            ))
        })?;

        info!(model = %model_name, "loading local model into pool");
        let model = LocalModel::load(descriptor, model_path, tokenizer_path, &self.device)?;

        let entry = entries.entry(model_name.to_string()).or_insert(PoolEntry {
            model,
            last_accessed: std::time::Instant::now(),
        });

        f(&mut entry.model)
    }

    /// Number of currently loaded models.
    pub fn loaded_count(&self) -> usize {
        self.entries
            .lock()
            .map(|e| e.len())
            .unwrap_or(0)
    }

    /// List names of currently loaded models.
    pub fn loaded_models(&self) -> Vec<String> {
        self.entries
            .lock()
            .map(|e| e.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Evict a specific model from the pool.
    pub fn evict(&self, model_name: &str) -> bool {
        self.entries
            .lock()
            .map(|mut e| e.remove(model_name).is_some())
            .unwrap_or(false)
    }

    /// Evict all models from the pool.
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }
}
