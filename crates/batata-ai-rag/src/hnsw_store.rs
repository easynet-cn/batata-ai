//! In-memory HNSW vector store.
//!
//! Uses `instant-distance` for approximate nearest-neighbour search.
//! Each knowledge base gets its own bucket; inside a bucket, chunks are
//! appended into a flat `Vec` and the HNSW index is rebuilt lazily on
//! first search after an `upsert` or `delete_by_doc`.
//!
//! This is the fast-path store for single-process deployments where
//! brute-force cosine over sea-orm's `SeaOrmVectorStore` becomes the
//! bottleneck (typically >100k chunks per KB). It is *not* persistent —
//! the index lives and dies with the server process.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use instant_distance::{Builder, HnswMap, Point, Search};

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::rag::{RagChunk, RagHit, RagSearchQuery, VectorStore};

/// A unit-normalised embedding. Distance is `1 - cosine_similarity`,
/// which makes higher similarity map to lower HNSW distance (required
/// by the library's monotonic-decreasing assumption).
#[derive(Clone)]
struct EmbeddingPoint {
    v: Vec<f32>,
    norm: f32,
}

impl EmbeddingPoint {
    fn new(v: Vec<f32>) -> Self {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        Self { v, norm }
    }
}

impl Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        if self.v.len() != other.v.len() {
            return 1.0;
        }
        let dot: f32 = self
            .v
            .iter()
            .zip(other.v.iter())
            .map(|(a, b)| a * b)
            .sum();
        let sim = dot / (self.norm * other.norm);
        (1.0 - sim).clamp(0.0, 2.0)
    }
}

struct KbBucket {
    chunks: Vec<RagChunk>,
    index: Option<HnswMap<EmbeddingPoint, usize>>,
    dirty: bool,
}

impl KbBucket {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            index: None,
            dirty: false,
        }
    }

    fn rebuild(&mut self) {
        if self.chunks.is_empty() {
            self.index = None;
            self.dirty = false;
            return;
        }
        let points: Vec<EmbeddingPoint> = self
            .chunks
            .iter()
            .map(|c| EmbeddingPoint::new(c.embedding.clone()))
            .collect();
        let values: Vec<usize> = (0..self.chunks.len()).collect();
        let map = Builder::default().build(points, values);
        self.index = Some(map);
        self.dirty = false;
    }
}

/// HNSW-backed vector store. See module docs for semantics.
pub struct HnswVectorStore {
    inner: Mutex<HashMap<String, KbBucket>>,
}

impl HnswVectorStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for HnswVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

fn poison<E: std::fmt::Display>(e: E) -> BatataError {
    BatataError::Storage(format!("hnsw store poisoned: {e}"))
}

#[async_trait]
impl VectorStore for HnswVectorStore {
    async fn upsert(&self, chunks: Vec<RagChunk>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(poison)?;
        for ch in chunks {
            let bucket = inner.entry(ch.kb_id.clone()).or_insert_with(KbBucket::new);
            bucket.chunks.push(ch);
            bucket.dirty = true;
        }
        Ok(())
    }

    async fn delete_by_doc(&self, kb_id: &str, doc_id: &str) -> Result<()> {
        let mut inner = self.inner.lock().map_err(poison)?;
        if let Some(bucket) = inner.get_mut(kb_id) {
            let before = bucket.chunks.len();
            bucket.chunks.retain(|c| c.doc_id != doc_id);
            if bucket.chunks.len() != before {
                bucket.dirty = true;
            }
        }
        Ok(())
    }

    async fn delete_by_kb(&self, kb_id: &str) -> Result<()> {
        let mut inner = self.inner.lock().map_err(poison)?;
        inner.remove(kb_id);
        Ok(())
    }

    async fn search(&self, q: RagSearchQuery) -> Result<Vec<RagHit>> {
        let mut inner = self.inner.lock().map_err(poison)?;
        let Some(bucket) = inner.get_mut(&q.kb_id) else {
            return Ok(Vec::new());
        };
        if bucket.dirty || bucket.index.is_none() {
            bucket.rebuild();
        }
        let Some(index) = bucket.index.as_ref() else {
            return Ok(Vec::new());
        };

        let query_point = EmbeddingPoint::new(q.embedding);
        let mut search = Search::default();
        let iter = index.search(&query_point, &mut search);

        let mut hits = Vec::with_capacity(q.top_k);
        for item in iter.take(q.top_k.max(1)) {
            let idx = *item.value;
            let chunk = bucket.chunks[idx].clone();
            // Convert HNSW distance back to cosine similarity for the caller.
            let score = (1.0 - item.distance).clamp(0.0, 1.0);
            if q.min_score.map_or(true, |m| score >= m) {
                hits.push(RagHit { chunk, score });
            }
        }
        hits.truncate(q.top_k);
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, kb: &str, doc: &str, v: Vec<f32>, text: &str) -> RagChunk {
        RagChunk {
            id: id.into(),
            doc_id: doc.into(),
            kb_id: kb.into(),
            ord: 0,
            text: text.into(),
            embedding: v,
            metadata: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn hnsw_finds_nearest() {
        let store = HnswVectorStore::new();
        store
            .upsert(vec![
                chunk("a", "kb", "d1", vec![1.0, 0.0, 0.0], "x-axis"),
                chunk("b", "kb", "d2", vec![0.0, 1.0, 0.0], "y-axis"),
                chunk("c", "kb", "d3", vec![0.0, 0.0, 1.0], "z-axis"),
            ])
            .await
            .unwrap();
        let hits = store
            .search(RagSearchQuery {
                kb_id: "kb".into(),
                embedding: vec![0.9, 0.1, 0.0],
                top_k: 1,
                min_score: None,
                filter: serde_json::Value::Null,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.id, "a");
    }

    #[tokio::test]
    async fn delete_by_doc_drops_and_rebuilds() {
        let store = HnswVectorStore::new();
        store
            .upsert(vec![
                chunk("a", "kb", "d1", vec![1.0, 0.0], "first"),
                chunk("b", "kb", "d2", vec![0.0, 1.0], "second"),
            ])
            .await
            .unwrap();
        store.delete_by_doc("kb", "d1").await.unwrap();
        let hits = store
            .search(RagSearchQuery {
                kb_id: "kb".into(),
                embedding: vec![1.0, 0.0],
                top_k: 5,
                min_score: None,
                filter: serde_json::Value::Null,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.id, "b");
    }
}
