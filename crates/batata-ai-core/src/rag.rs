//! RAG (Retrieval-Augmented Generation) abstractions.
//!
//! The concrete implementations live in `batata-ai-rag`; this module only
//! defines the contracts so that repository, API, and router layers can
//! depend on `batata-ai-core` without pulling the embedding / vector store
//! stack.

use async_trait::async_trait;
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// A knowledge base is a logical container for documents + chunks, scoped
/// optionally to a tenant. Dimensions and chunker parameters are pinned at
/// creation so that swapping embedders mid-lifetime is a hard error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBase {
    pub id: String,
    pub tenant_id: Option<String>,
    pub name: String,
    pub description: Option<String>,
    pub embedder: String,
    pub dim: i32,
    pub chunk_window: i32,
    pub chunk_overlap: i32,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    #[serde(default)]
    pub deleted_at: Option<NaiveDateTime>,
}

/// A document ingested into a knowledge base. Used for listing and
/// managing the lifecycle of ingested content. `chunk_count` is a
/// denormalised counter maintained by the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KbDocument {
    pub id: String,
    pub kb_id: String,
    pub tenant_id: Option<String>,
    pub source_uri: String,
    pub title: Option<String>,
    pub mime: Option<String>,
    pub status: String,
    pub error: Option<String>,
    pub chunk_count: i32,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    #[serde(default)]
    pub deleted_at: Option<NaiveDateTime>,
}

#[async_trait]
pub trait KnowledgeBaseRepository: Send + Sync {
    async fn create(&self, entity: &KnowledgeBase) -> Result<KnowledgeBase>;
    async fn find_by_id(&self, id: &str) -> Result<Option<KnowledgeBase>>;
    /// If `tenant_id` is `None`, only platform-level KBs (tenant_id NULL)
    /// are returned. If `Some`, both platform-level and that tenant's KBs
    /// are returned.
    async fn find_by_tenant(&self, tenant_id: Option<&str>) -> Result<Vec<KnowledgeBase>>;
    async fn update(&self, entity: &KnowledgeBase) -> Result<KnowledgeBase>;
    async fn soft_delete(&self, id: &str) -> Result<bool>;
}

#[async_trait]
pub trait KbDocumentRepository: Send + Sync {
    async fn create(&self, entity: &KbDocument) -> Result<KbDocument>;
    async fn update(&self, entity: &KbDocument) -> Result<KbDocument>;
    async fn find_by_id(&self, id: &str) -> Result<Option<KbDocument>>;
    async fn find_by_kb(&self, kb_id: &str) -> Result<Vec<KbDocument>>;
    async fn soft_delete(&self, id: &str) -> Result<bool>;
    /// Soft-delete every non-deleted document under a KB. Returns the
    /// number of rows affected.
    async fn soft_delete_by_kb(&self, kb_id: &str) -> Result<u64>;
}

/// Metadata about a document ingested into a knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagDocument {
    pub id: String,
    pub kb_id: String,
    pub source_uri: String,
    pub title: Option<String>,
    pub mime: Option<String>,
    pub created_at: NaiveDateTime,
}

/// A single text chunk derived from a document, together with its embedding.
///
/// The `metadata` field is an opaque JSON blob — callers may store filters
/// (e.g. `{"page": 3, "lang": "zh"}`) there and match on them at query time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagChunk {
    pub id: String,
    pub doc_id: String,
    pub kb_id: String,
    pub ord: u32,
    pub text: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A search hit with its similarity score (cosine, higher = more similar).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagHit {
    pub chunk: RagChunk,
    pub score: f32,
}

/// Query parameters for a vector similarity search.
#[derive(Debug, Clone)]
pub struct RagSearchQuery {
    pub kb_id: String,
    pub embedding: Vec<f32>,
    pub top_k: usize,
    pub min_score: Option<f32>,
    pub filter: serde_json::Value,
}

/// Turns text into fixed-dimension vectors.
#[async_trait]
pub trait Embedder: Send + Sync {
    fn dimensions(&self) -> usize;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Cross-encoder reranker. Called after coarse retrieval — takes the
/// candidate passages against a query and returns relevance scores,
/// higher = more relevant. Not normalised (raw logits).
#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, passages: &[String]) -> Result<Vec<f32>>;
}

/// Splits a raw document body into chunks ready for embedding.
pub trait Chunker: Send + Sync {
    fn chunk(&self, text: &str) -> Vec<String>;
}

/// Persistence + similarity search for embedded chunks.
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, chunks: Vec<RagChunk>) -> Result<()>;
    async fn delete_by_doc(&self, kb_id: &str, doc_id: &str) -> Result<()>;
    /// Hard-delete every chunk belonging to a KB. Used for cascade
    /// cleanup when a knowledge base is soft-deleted — keeping orphaned
    /// chunks would cause `search` to leak data from a "deleted" KB.
    async fn delete_by_kb(&self, kb_id: &str) -> Result<()>;
    async fn search(&self, query: RagSearchQuery) -> Result<Vec<RagHit>>;
}

/// Reads raw text content from a source URI (filesystem path, HTTP URL, …).
#[async_trait]
pub trait DocumentLoader: Send + Sync {
    fn supported_mimes(&self) -> &[&str];
    async fn load(&self, source_uri: &str) -> Result<String>;
}
