use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::rag::{RagChunk, RagHit, RagSearchQuery, VectorStore};

/// Simple in-memory vector store keyed by `kb_id`.
///
/// Similarity is computed with a brute-force cosine scan. Suitable for tests
/// and small knowledge bases (< ~100k chunks). Switch to a sqlite-vec or
/// pgvector backend for production workloads.
#[derive(Default)]
pub struct InMemoryVectorStore {
    inner: Mutex<HashMap<String, Vec<RagChunk>>>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self, kb_id: &str) -> usize {
        self.inner
            .lock()
            .ok()
            .and_then(|m| m.get(kb_id).map(|v| v.len()))
            .unwrap_or(0)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn upsert(&self, chunks: Vec<RagChunk>) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| BatataError::Storage(format!("vector store poisoned: {e}")))?;
        for ch in chunks {
            inner.entry(ch.kb_id.clone()).or_default().push(ch);
        }
        Ok(())
    }

    async fn delete_by_doc(&self, kb_id: &str, doc_id: &str) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| BatataError::Storage(format!("vector store poisoned: {e}")))?;
        if let Some(vec) = inner.get_mut(kb_id) {
            vec.retain(|c| c.doc_id != doc_id);
        }
        Ok(())
    }

    async fn delete_by_kb(&self, kb_id: &str) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| BatataError::Storage(format!("vector store poisoned: {e}")))?;
        inner.remove(kb_id);
        Ok(())
    }

    async fn search(&self, q: RagSearchQuery) -> Result<Vec<RagHit>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| BatataError::Storage(format!("vector store poisoned: {e}")))?;
        let Some(chunks) = inner.get(&q.kb_id) else {
            return Ok(Vec::new());
        };
        let mut hits: Vec<RagHit> = chunks
            .iter()
            .map(|c| RagHit {
                score: cosine(&q.embedding, &c.embedding),
                chunk: c.clone(),
            })
            .filter(|h| q.min_score.map_or(true, |m| h.score >= m))
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(q.top_k);
        Ok(hits)
    }
}
