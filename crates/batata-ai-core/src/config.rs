use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{BatataError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatataConfig {
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub local: LocalConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub rag: RagConfig,
}

impl Default for BatataConfig {
    fn default() -> Self {
        Self {
            providers: HashMap::new(),
            local: LocalConfig::default(),
            mcp: McpConfig::default(),
            rag: RagConfig::default(),
        }
    }
}

impl BatataConfig {
    /// Parse a TOML config file. Returns `Default::default()` if `path` is
    /// `None`; returns an error if the file exists but cannot be parsed.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let raw = std::fs::read_to_string(path).map_err(|e| {
            BatataError::Config(format!(
                "failed to read config file {}: {e}",
                path.display()
            ))
        })?;
        toml::from_str::<Self>(&raw).map_err(|e| {
            BatataError::Config(format!("failed to parse {}: {e}", path.display()))
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub provider_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LocalConfig {
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub use_gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_mcp_port")]
    pub port: u16,
}

fn default_models_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("batata-ai")
        .join("models")
}

fn default_mcp_port() -> u16 {
    3000
}

/// RAG subsystem configuration.
///
/// When `enabled = false`, the API layer constructs no pipeline and any
/// RAG-specific endpoint returns `503 Service Unavailable`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub embedder: RagEmbedderKind,
    #[serde(default)]
    pub device: RagDevice,
    #[serde(default)]
    pub chunker: ChunkerConfig,
    /// Where to store chunks. `InMemory` is volatile, `Database` uses the
    /// main SeaORM connection pool.
    #[serde(default)]
    pub store: RagStoreKind,
    /// Directory name under `$BATATA_AI_MODELS_DIR` containing the embedder
    /// weights. Only applies to disk-loaded embedders (BGE, …).
    /// Defaults to `"bge-small-zh-v1.5"`.
    #[serde(default = "default_bge_model_name")]
    pub bge_model: String,
    /// Optional cross-encoder reranker config.
    #[serde(default)]
    pub reranker: RerankerConfig,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            embedder: RagEmbedderKind::default(),
            device: RagDevice::default(),
            chunker: ChunkerConfig::default(),
            store: RagStoreKind::default(),
            bge_model: default_bge_model_name(),
            reranker: RerankerConfig::default(),
        }
    }
}

fn default_bge_model_name() -> String {
    "bge-small-zh-v1.5".to_string()
}

/// Cross-encoder reranker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Directory name under models dir. Defaults to `bge-reranker-base`.
    #[serde(default = "default_reranker_model")]
    pub model: String,
    /// How many candidates to pull from the vector store before reranking.
    #[serde(default = "default_rerank_candidates")]
    pub candidates: usize,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: default_reranker_model(),
            candidates: default_rerank_candidates(),
        }
    }
}

fn default_reranker_model() -> String {
    "bge-reranker-base".to_string()
}

fn default_rerank_candidates() -> usize {
    20
}

/// Backend for chunk persistence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RagStoreKind {
    /// Brute-force cosine over an in-memory HashMap. Suitable for tests
    /// and smoke; O(N) per query.
    #[default]
    InMemory,
    /// Persists to the main SeaORM database connection (`kb_chunks` table).
    /// Brute-force cosine on load.
    Database,
    /// In-memory HNSW index via `instant-distance`. Approximate,
    /// sub-linear query time. Not persistent across restarts.
    Hnsw,
}

/// Supported embedder backends. Extend here as new impls land.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RagEmbedderKind {
    /// CLIP text encoder (512 dim, 77-token context). Cross-modal fallback.
    #[default]
    Clip,
    /// BGE BERT encoder — loads from `<models_dir>/<rag.bge_model>/`.
    /// Use for text-only retrieval; better quality + longer context
    /// (`bge-small-zh-v1.5`: 512 tokens, `bge-m3`: 8192 tokens).
    Bge,
}

/// Compute device for the embedder.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RagDevice {
    #[default]
    Cpu,
    /// Prefer GPU (CUDA/Metal); falls back to CPU if unavailable.
    Gpu,
}

/// Chunker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkerConfig {
    #[serde(default)]
    pub kind: ChunkerKind,
    /// Character window size.
    #[serde(default = "default_chunk_window")]
    pub window: usize,
    /// Character overlap between adjacent windows (ignored by recursive).
    #[serde(default = "default_chunk_overlap")]
    pub overlap: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            kind: ChunkerKind::default(),
            window: default_chunk_window(),
            overlap: default_chunk_overlap(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ChunkerKind {
    /// Fixed character window with overlap. Simple and cheap.
    #[default]
    FixedWindow,
    /// Recursive structure-aware splitter — respects paragraph / line
    /// breaks first, then sentence / word, then raw characters.
    /// Recommended for Markdown, HTML, and code.
    Recursive,
}

fn default_chunk_window() -> usize {
    512
}

fn default_chunk_overlap() -> usize {
    64
}
