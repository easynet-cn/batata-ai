use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle_core::Device;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::rag::Embedder;
use batata_ai_local::bge::BgeModel;
use batata_ai_local::bge_m3::BgeM3Model;
use batata_ai_local::clip::ClipModel;

/// Run a CPU-bound closure on tokio's blocking thread pool so the
/// reactor isn't starved while an embedder chews through a batch.
async fn run_blocking<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| BatataError::Inference(format!("blocking task join failed: {e}")))?
}

/// An [`Embedder`] backed by the existing CLIP text encoder.
///
/// M1 uses CLIP because it's already on disk and wired into the project;
/// quality is noticeably worse than dedicated text encoders (BGE, E5, Jina)
/// and CLIP has a hard 77-token context limit. M4 adds a proper BGE backend.
pub struct ClipEmbedder {
    model: Arc<Mutex<ClipModel>>,
    dim: usize,
}

impl ClipEmbedder {
    pub fn new(model: ClipModel) -> Self {
        let dim = model.dimensions();
        Self {
            model: Arc::new(Mutex::new(model)),
            dim,
        }
    }

    /// Load CLIP from HuggingFace cache (or download if missing).
    pub fn load_default(device: &Device) -> Result<Self> {
        let model = ClipModel::download_and_load(device)?;
        Ok(Self::new(model))
    }
}

#[async_trait]
impl Embedder for ClipEmbedder {
    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let texts = texts.to_vec();
        let model = Arc::clone(&self.model);
        run_blocking(move || {
            let guard = model
                .lock()
                .map_err(|e| BatataError::Inference(format!("clip embedder poisoned: {e}")))?;
            guard.embed_texts(&texts)
        })
        .await
    }
}

/// An [`Embedder`] backed by a BGE BERT-style text encoder loaded from a
/// local directory (`<models_dir>/<name>/`).
///
/// BGE is the recommended production embedder — higher quality than CLIP
/// for text retrieval and supports longer context (BGE-small-zh: 512
/// tokens; BGE-m3: 8192 tokens).
pub struct BgeEmbedder {
    model: Arc<Mutex<BgeModel>>,
    dim: usize,
}

impl BgeEmbedder {
    pub fn new(model: BgeModel) -> Self {
        let dim = model.dimensions();
        Self {
            model: Arc::new(Mutex::new(model)),
            dim,
        }
    }

    /// Load a BGE model from a local directory (must contain
    /// `config.json`, `tokenizer.json`, `model.safetensors`).
    pub fn load_local(dir: &Path, device: &Device) -> Result<Self> {
        let model = BgeModel::load_local(dir, device)?;
        Ok(Self::new(model))
    }
}

#[async_trait]
impl Embedder for BgeEmbedder {
    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let texts = texts.to_vec();
        let model = Arc::clone(&self.model);
        run_blocking(move || {
            let guard = model
                .lock()
                .map_err(|e| BatataError::Inference(format!("bge embedder poisoned: {e}")))?;
            guard.embed(&texts)
        })
        .await
    }
}

/// An [`Embedder`] backed by BGE-M3 (XLM-RoBERTa-large). 1024-dim vectors,
/// 8k-token context, multilingual. Heavier than `BgeEmbedder`
/// (~2.3 GB weights) but better quality and longer chunks.
pub struct BgeM3Embedder {
    model: Arc<Mutex<BgeM3Model>>,
    dim: usize,
}

impl BgeM3Embedder {
    pub fn new(model: BgeM3Model) -> Self {
        let dim = model.dimensions();
        Self {
            model: Arc::new(Mutex::new(model)),
            dim,
        }
    }

    pub fn load_local(dir: &Path, device: &Device) -> Result<Self> {
        let model = BgeM3Model::load_local(dir, device)?;
        Ok(Self::new(model))
    }
}

#[async_trait]
impl Embedder for BgeM3Embedder {
    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let texts = texts.to_vec();
        let model = Arc::clone(&self.model);
        run_blocking(move || {
            let guard = model
                .lock()
                .map_err(|e| BatataError::Inference(format!("bge-m3 embedder poisoned: {e}")))?;
            guard.embed(&texts)
        })
        .await
    }
}
