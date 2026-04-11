//! Build an [`IngestPipeline`] from a [`RagConfig`].
//!
//! Keeps all "how to materialise a concrete pipeline" knowledge inside the
//! rag crate so that the server/main binary only passes around config.

use std::sync::Arc;

use batata_ai_core::config::{ChunkerKind, RagConfig, RagDevice, RagEmbedderKind, RagStoreKind};
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::rag::{Reranker, VectorStore};
use batata_ai_local::model::resolve_device;
use batata_ai_local::models::default_models_dir;
use batata_ai_storage::SeaOrmVectorStore;
use sea_orm::DatabaseConnection;

use crate::{
    BgeEmbedder, BgeM3Embedder, BgeReranker, ClipEmbedder, FixedWindowChunker, HnswVectorStore,
    InMemoryVectorStore, IngestPipeline, RecursiveChunker,
};

/// Try to build a pipeline from `cfg`. Returns `Ok(None)` when RAG is
/// disabled in config — callers should treat this as "no pipeline, leave
/// `AppState.rag_pipeline = None`".
///
/// `db` is required when `cfg.store == RagStoreKind::Database`; passing
/// `None` in that case returns an error.
pub fn build_pipeline(
    cfg: &RagConfig,
    db: Option<&DatabaseConnection>,
) -> Result<Option<Arc<IngestPipeline>>> {
    if !cfg.enabled {
        return Ok(None);
    }

    let use_gpu = matches!(cfg.device, RagDevice::Gpu);
    let device = resolve_device(use_gpu)
        .map_err(|e| BatataError::Config(format!("rag: failed to resolve device: {e}")))?;

    let embedder: Arc<dyn batata_ai_core::rag::Embedder> = match cfg.embedder {
        RagEmbedderKind::Clip => Arc::new(ClipEmbedder::load_default(&device)?),
        RagEmbedderKind::Bge => {
            let dir = default_models_dir().join(&cfg.bge_model);
            if !dir.exists() {
                return Err(BatataError::Config(format!(
                    "rag.embedder = 'bge' but model directory not found: {}. \
                     Run `cargo run --example download_multimodal -- {}` first.",
                    dir.display(),
                    &cfg.bge_model
                )));
            }
            // Dispatch on the model family: bge-m3 is XLM-RoBERTa while
            // bge-small/base/large-zh are standard BERT. The dir name is
            // the cheapest reliable signal.
            if cfg.bge_model.to_lowercase().contains("m3") {
                Arc::new(BgeM3Embedder::load_local(&dir, &device)?)
            } else {
                Arc::new(BgeEmbedder::load_local(&dir, &device)?)
            }
        }
    };

    let chunker: Arc<dyn batata_ai_core::rag::Chunker> = match cfg.chunker.kind {
        ChunkerKind::FixedWindow => Arc::new(FixedWindowChunker::new(
            cfg.chunker.window,
            cfg.chunker.overlap,
        )),
        ChunkerKind::Recursive => Arc::new(RecursiveChunker::new(cfg.chunker.window)),
    };

    let store: Arc<dyn VectorStore> = match cfg.store {
        RagStoreKind::InMemory => Arc::new(InMemoryVectorStore::new()),
        RagStoreKind::Database => {
            let db = db.ok_or_else(|| {
                BatataError::Config(
                    "rag.store = 'database' requires a DB connection to be passed to build_pipeline"
                        .into(),
                )
            })?;
            Arc::new(SeaOrmVectorStore::new(db.clone()))
        }
        RagStoreKind::Hnsw => Arc::new(HnswVectorStore::new()),
    };

    let mut pipeline = IngestPipeline::new(embedder, chunker, store);

    if cfg.reranker.enabled {
        let dir = default_models_dir().join(&cfg.reranker.model);
        if !dir.exists() {
            return Err(BatataError::Config(format!(
                "rag.reranker.enabled = true but model directory not found: {}. \
                 Run `cargo run --example download_multimodal -- {}` first.",
                dir.display(),
                &cfg.reranker.model
            )));
        }
        let reranker: Arc<dyn Reranker> = Arc::new(BgeReranker::load_local(&dir, &device)?);
        pipeline = pipeline.with_reranker(reranker, cfg.reranker.candidates);
    }

    Ok(Some(Arc::new(pipeline)))
}
