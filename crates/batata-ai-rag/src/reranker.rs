//! Reranker implementations.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle_core::Device;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::rag::Reranker;
use batata_ai_local::reranker::BgeRerankerModel;

/// Cross-encoder reranker backed by BGE-reranker-base (XLM-R + classification head).
pub struct BgeReranker {
    model: Arc<Mutex<BgeRerankerModel>>,
}

impl BgeReranker {
    pub fn new(model: BgeRerankerModel) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
        }
    }

    pub fn load_local(dir: &Path, device: &Device) -> Result<Self> {
        let model = BgeRerankerModel::load_local(dir, device)?;
        Ok(Self::new(model))
    }
}

#[async_trait]
impl Reranker for BgeReranker {
    async fn rerank(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        let query = query.to_string();
        let passages = passages.to_vec();
        let model = Arc::clone(&self.model);
        tokio::task::spawn_blocking(move || {
            let guard = model
                .lock()
                .map_err(|e| BatataError::Inference(format!("reranker poisoned: {e}")))?;
            guard.score_pairs(&query, &passages)
        })
        .await
        .map_err(|e| BatataError::Inference(format!("rerank join failed: {e}")))?
    }
}
