//! Retrieval-Augmented Generation building blocks.
//!
//! M1 scope: in-memory vector store, fixed-window chunker, a CLIP-based
//! embedder wrapper, and a pipeline that orchestrates ingest + search.
//!
//! Later milestones replace `InMemoryVectorStore` with sqlite-vec / pgvector
//! and wire the pipeline into the HTTP API and `/v1/chat/completions`.

pub mod builder;
pub mod chunker;
pub mod embedder;
pub mod hnsw_store;
pub mod loader;
pub mod pipeline;
pub mod reranker;
pub mod store;

pub use builder::build_pipeline;
pub use chunker::{FixedWindowChunker, RecursiveChunker};
pub use embedder::{BgeEmbedder, BgeM3Embedder, ClipEmbedder};
pub use hnsw_store::HnswVectorStore;
pub use loader::{load_bytes, load_file, load_uri, DocumentFormat};
pub use pipeline::IngestPipeline;
pub use reranker::BgeReranker;
pub use store::InMemoryVectorStore;
