//! Pre-download multimodal model files (Whisper / CLIP / BLIP) into
//! `$BATATA_AI_MODELS_DIR` (default `~/work/models`), one subdirectory per
//! model. Mirrors what `batata_ai_local::{whisper,clip,blip}` fetch at runtime,
//! but keeps a clean on-disk layout for offline / air-gapped use.

use std::path::{Path, PathBuf};

use batata_ai::local::models::default_models_dir;

struct MultimodalSpec {
    /// Directory name under the models base dir.
    dir_name: &'static str,
    /// HuggingFace repo id.
    repo_id: &'static str,
    /// Files to pull from the repo.
    files: &'static [&'static str],
}

/// Entries cover every repo that `batata-ai-local` references today.
const SPECS: &[MultimodalSpec] = &[
    // Whisper — speech-to-text
    MultimodalSpec {
        dir_name: "whisper-tiny",
        repo_id: "openai/whisper-tiny",
        files: &["config.json", "tokenizer.json", "model.safetensors"],
    },
    MultimodalSpec {
        dir_name: "whisper-base",
        repo_id: "openai/whisper-base",
        files: &["config.json", "tokenizer.json", "model.safetensors"],
    },
    MultimodalSpec {
        dir_name: "whisper-small",
        repo_id: "openai/whisper-small",
        files: &["config.json", "tokenizer.json", "model.safetensors"],
    },
    MultimodalSpec {
        dir_name: "whisper-medium",
        repo_id: "openai/whisper-medium",
        files: &["config.json", "tokenizer.json", "model.safetensors"],
    },
    MultimodalSpec {
        dir_name: "whisper-large-v3-turbo",
        repo_id: "openai/whisper-large-v3-turbo",
        files: &["config.json", "tokenizer.json", "model.safetensors"],
    },
    // CLIP — image/text joint embedding.
    // Note: upstream repo ships weights as `pytorch_model.bin` only
    // (no `model.safetensors`). We pull the .bin plus config/preprocessor
    // so the layout is complete for offline use.
    MultimodalSpec {
        dir_name: "clip-vit-base-patch32",
        repo_id: "openai/clip-vit-base-patch32",
        files: &[
            "pytorch_model.bin",
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
        ],
    },
    // BLIP — image captioning
    MultimodalSpec {
        dir_name: "blip-image-captioning-large",
        repo_id: "Salesforce/blip-image-captioning-large",
        files: &["model.safetensors", "tokenizer.json"],
    },
    // BGE — Chinese / general-purpose text embeddings. 512-dim,
    // dense vectors produced by CLS-pooling + L2 norm.
    MultimodalSpec {
        dir_name: "bge-small-zh-v1.5",
        repo_id: "BAAI/bge-small-zh-v1.5",
        files: &[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ],
    },
    // BGE-M3 — multilingual (100+ languages), 1024-dim, 8192-token ctx.
    // Heavier (~2.3 GB). Dense + sparse + ColBERT retrieval modes; we
    // use only the dense path.
    MultimodalSpec {
        dir_name: "bge-m3",
        repo_id: "BAAI/bge-m3",
        files: &[
            "pytorch_model.bin",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    },
    // BGE reranker — cross-encoder. Takes (query, passage) pairs and
    // emits a single relevance logit. Based on XLM-RoBERTa-base.
    MultimodalSpec {
        dir_name: "bge-reranker-base",
        repo_id: "BAAI/bge-reranker-base",
        files: &[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    },
];

fn copy_if_missing(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if dst.exists() {
        return Ok(());
    }
    std::fs::copy(src, dst)?;
    Ok(())
}

fn download_spec(spec: &MultimodalSpec, base_dir: &Path) -> anyhow::Result<PathBuf> {
    let target_dir = base_dir.join(spec.dir_name);
    std::fs::create_dir_all(&target_dir)?;

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(spec.repo_id.to_string());

    for file in spec.files {
        let cached = repo
            .get(file)
            .map_err(|e| anyhow::anyhow!("{}: fetch {file} failed: {e}", spec.repo_id))?;
        let dst = target_dir.join(file);
        copy_if_missing(&cached, &dst)?;
        println!("  ✓ {}", dst.display());
    }
    Ok(target_dir)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let base_dir = default_models_dir();
    println!("Base models dir: {}", base_dir.display());
    println!();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let filter: Option<&str> = args.first().map(|s| s.as_str());

    let mut ok = 0usize;
    let mut fail = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();

    for spec in SPECS {
        if let Some(f) = filter {
            if !spec.dir_name.contains(f) {
                continue;
            }
        }
        println!("== {} ({})", spec.dir_name, spec.repo_id);
        match download_spec(spec, &base_dir) {
            Ok(path) => {
                println!("  → {}", path.display());
                ok += 1;
            }
            Err(e) => {
                println!("  ✗ {e}");
                failures.push((spec.dir_name.to_string(), e.to_string()));
                fail += 1;
            }
        }
        println!();
    }

    println!("Done. ok={ok} fail={fail}");
    if !failures.is_empty() {
        println!("Failures:");
        for (name, err) in &failures {
            println!("  - {name}: {err}");
        }
    }
    Ok(())
}
