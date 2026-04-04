use std::path::{Path, PathBuf};

use batata_ai_core::error::{BatataError, Result};
use candle_core::Device;
use tracing::info;

/// Supported model formats
#[derive(Debug, Clone)]
pub enum ModelFormat {
    Safetensors,
    Gguf,
}

/// Metadata about a loaded or available model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub format: ModelFormat,
    pub size_bytes: u64,
}

/// Resolves the compute device (CPU, CUDA, or Metal)
pub fn resolve_device(use_gpu: bool) -> Result<Device> {
    if use_gpu {
        #[cfg(feature = "cuda")]
        {
            info!("using CUDA device");
            return Device::new_cuda(0).map_err(|e| BatataError::Inference(e.to_string()));
        }

        #[cfg(feature = "metal")]
        {
            info!("using Metal device");
            return Device::new_metal(0).map_err(|e| BatataError::Inference(e.to_string()));
        }

        #[allow(unreachable_code)]
        {
            tracing::warn!("GPU requested but no GPU feature enabled, falling back to CPU");
            Ok(Device::Cpu)
        }
    } else {
        info!("using CPU device");
        Ok(Device::Cpu)
    }
}

/// Download a model from Hugging Face Hub
pub fn download_from_hub(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| BatataError::Inference(e.to_string()))?;

    let repo = if let Some(rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.to_string(),
            hf_hub::RepoType::Model,
            rev.to_string(),
        ))
    } else {
        api.model(repo_id.to_string())
    };

    let path = repo
        .get(filename)
        .map_err(|e| BatataError::Inference(format!("failed to download {repo_id}/{filename}: {e}")))?;

    Ok(path)
}

/// List locally cached models in the given directory
pub fn list_local_models(models_dir: &Path) -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();

    if !models_dir.exists() {
        return Ok(models);
    }

    let entries = std::fs::read_dir(models_dir)?;
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let format = match ext {
                "safetensors" => Some(ModelFormat::Safetensors),
                "gguf" => Some(ModelFormat::Gguf),
                _ => None,
            };

            if let Some(format) = format {
                let metadata = std::fs::metadata(&path)?;
                models.push(ModelInfo {
                    name: path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                    path,
                    format,
                    size_bytes: metadata.len(),
                });
            }
        }
    }

    Ok(models)
}
