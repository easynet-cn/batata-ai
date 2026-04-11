use std::path::Path;

use batata_ai_local::models::{ModelDescriptor, download_model_to};
use batata_ai_local::stable_diffusion::download_sd_v1_5;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let target = Path::new("/Volumes/Elements SE/models");
    if !target.exists() {
        anyhow::bail!("target not mounted: {}", target.display());
    }

    let names: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<&str> = if names.is_empty() {
        ModelDescriptor::available_models()
    } else {
        names.iter().map(String::as_str).collect()
    };

    for name in names {
        if name == "sd-v1.5" || name == "stable-diffusion-v1-5" {
            let dir = target.join("sd-v1.5");
            if dir.join("unet.safetensors").exists() {
                tracing::info!("skip sd-v1.5: already present at {}", dir.display());
                continue;
            }
            tracing::info!("downloading sd-v1.5 -> {}", target.display());
            let paths = download_sd_v1_5(target)?;
            tracing::info!(
                "done sd-v1.5: tokenizer={} clip={} unet={} vae={}",
                paths.tokenizer.display(),
                paths.clip.display(),
                paths.unet.display(),
                paths.vae.display()
            );
            continue;
        }

        let dir = target.join(name);
        if dir.exists() && dir.join("tokenizer.json").exists() {
            tracing::info!("skip {name}: already present at {}", dir.display());
            continue;
        }
        tracing::info!("downloading {name} -> {}", target.display());
        let (model_path, tok_path) = download_model_to(name, target, true)?;
        tracing::info!(
            "done {name}: {} + {}",
            model_path.display(),
            tok_path.display()
        );
    }

    Ok(())
}
