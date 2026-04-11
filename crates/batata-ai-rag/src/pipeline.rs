use std::sync::Arc;

use tracing::debug;
use uuid::Uuid;

use batata_ai_core::error::Result;
use batata_ai_core::rag::{
    Chunker, Embedder, RagChunk, RagHit, RagSearchQuery, Reranker, VectorStore,
};

/// Orchestrates the chunker → embedder → vector store flow with an
/// optional cross-encoder reranker in front of the final top-k cut.
pub struct IngestPipeline {
    pub embedder: Arc<dyn Embedder>,
    pub chunker: Arc<dyn Chunker>,
    pub store: Arc<dyn VectorStore>,
    /// Optional cross-encoder for two-stage retrieval.
    pub reranker: Option<Arc<dyn Reranker>>,
    /// How many candidates to fetch from the vector store before
    /// reranking. Ignored if `reranker` is `None`. Defaults to `top_k * 4`.
    pub rerank_candidates: usize,
}

impl IngestPipeline {
    pub fn new(
        embedder: Arc<dyn Embedder>,
        chunker: Arc<dyn Chunker>,
        store: Arc<dyn VectorStore>,
    ) -> Self {
        Self {
            embedder,
            chunker,
            store,
            reranker: None,
            rerank_candidates: 0,
        }
    }

    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>, candidates: usize) -> Self {
        self.reranker = Some(reranker);
        self.rerank_candidates = candidates;
        self
    }

    /// Chunk raw text, embed, upsert into the store. Returns the new doc id.
    pub async fn ingest_text(
        &self,
        kb_id: &str,
        source_uri: &str,
        text: &str,
    ) -> Result<String> {
        let doc_id = Uuid::new_v4().to_string();
        let parts = self.chunker.chunk(text);
        if parts.is_empty() {
            debug!(%doc_id, "ingest_text: empty input, skipping");
            return Ok(doc_id);
        }
        debug!(%doc_id, kb_id, chunks = parts.len(), "ingest_text: embedding");
        let vectors = self.embedder.embed(&parts).await?;

        let chunks: Vec<RagChunk> = parts
            .into_iter()
            .zip(vectors)
            .enumerate()
            .map(|(i, (text, embedding))| RagChunk {
                id: Uuid::new_v4().to_string(),
                doc_id: doc_id.clone(),
                kb_id: kb_id.to_string(),
                ord: i as u32,
                text,
                embedding,
                metadata: serde_json::json!({ "source_uri": source_uri }),
            })
            .collect();

        self.store.upsert(chunks).await?;
        Ok(doc_id)
    }

    /// Embed a query and return the top-k most similar chunks. When a
    /// reranker is configured, fetch `max(top_k, rerank_candidates)`
    /// from the store first, then let the cross-encoder decide the
    /// final order and cut to `top_k`.
    pub async fn search(
        &self,
        kb_id: &str,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RagHit>> {
        let mut emb = self.embedder.embed(&[query.to_string()]).await?;
        let embedding = emb.pop().unwrap_or_default();

        // Fetch more candidates when a reranker is configured.
        let fetch_k = if self.reranker.is_some() {
            self.rerank_candidates.max(top_k)
        } else {
            top_k
        };

        let q = RagSearchQuery {
            kb_id: kb_id.to_string(),
            embedding,
            top_k: fetch_k,
            min_score: None,
            filter: serde_json::Value::Null,
        };
        let coarse = self.store.search(q).await?;

        let Some(reranker) = self.reranker.as_ref() else {
            return Ok(coarse);
        };
        if coarse.is_empty() {
            return Ok(coarse);
        }

        let passages: Vec<String> = coarse.iter().map(|h| h.chunk.text.clone()).collect();
        let scores = reranker.rerank(query, &passages).await?;

        // Pair each hit with its reranker score, sort desc, take top_k.
        let mut rescored: Vec<RagHit> = coarse
            .into_iter()
            .zip(scores.into_iter())
            .map(|(mut hit, score)| {
                hit.score = score;
                hit
            })
            .collect();
        rescored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rescored.truncate(top_k);
        Ok(rescored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use batata_ai_core::rag::Embedder;

    use crate::{FixedWindowChunker, InMemoryVectorStore};

    /// Deterministic hash-based embedder — no model load, runs in CI.
    struct BagOfBytesEmbedder;

    #[async_trait]
    impl Embedder for BagOfBytesEmbedder {
        fn dimensions(&self) -> usize {
            32
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0f32; 32];
                    for b in t.to_lowercase().bytes() {
                        v[(b as usize) % 32] += 1.0;
                    }
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                    for x in &mut v {
                        *x /= norm;
                    }
                    v
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn end_to_end_ingest_and_search() {
        let pipeline = IngestPipeline::new(
            Arc::new(BagOfBytesEmbedder),
            Arc::new(FixedWindowChunker::new(40, 8)),
            Arc::new(InMemoryVectorStore::new()),
        );

        pipeline
            .ingest_text(
                "kb_test",
                "doc://cat",
                "A small cat is sleeping peacefully on a warm mat near the window.",
            )
            .await
            .unwrap();
        pipeline
            .ingest_text(
                "kb_test",
                "doc://rust",
                "Rust is a memory-safe systems programming language used for reliable software.",
            )
            .await
            .unwrap();

        let hits = pipeline.search("kb_test", "sleeping cat", 2).await.unwrap();
        assert!(!hits.is_empty(), "expected at least one hit");
        let top = &hits[0].chunk.text.to_lowercase();
        assert!(
            top.contains("cat") || top.contains("mat") || top.contains("sleep"),
            "top hit unexpected: {top}"
        );
    }

    #[tokio::test]
    async fn delete_by_doc_removes_chunks() {
        let store = Arc::new(InMemoryVectorStore::new());
        let pipeline = IngestPipeline::new(
            Arc::new(BagOfBytesEmbedder),
            Arc::new(FixedWindowChunker::new(20, 4)),
            store.clone(),
        );
        let doc_id = pipeline
            .ingest_text("kb_del", "doc://x", "hello world hello world hello world")
            .await
            .unwrap();
        assert!(store.len("kb_del") > 0);
        store.delete_by_doc("kb_del", &doc_id).await.unwrap();
        assert_eq!(store.len("kb_del"), 0);
    }
}
