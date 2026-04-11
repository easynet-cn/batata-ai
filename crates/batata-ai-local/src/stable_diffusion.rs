//! Stable Diffusion v1.5 text-to-image inference via candle.
//!
//! Weights are loaded from HuggingFace repos:
//! - `runwayml/stable-diffusion-v1-5` (text_encoder / unet / vae, safetensors)
//! - `openai/clip-vit-large-patch14` (tokenizer.json)
//!
//! On CPU, a single 512x512 image at 20 steps takes minutes — not suitable
//! for interactive use, but the plumbing is identical to GPU/Metal paths.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_transformers::models::stable_diffusion::{
    self, vae::AutoEncoderKL, StableDiffusionConfig,
};
use tokenizers::Tokenizer;
use tracing::info;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::multimodal::{GeneratedImage, ImageGenParams, TextToImage};

const VAE_SCALE_V1_5: f64 = 0.18215;
const UNCOND_PROMPT: &str = "";

/// Locally resolved weight files for Stable Diffusion v1.5.
#[derive(Debug, Clone)]
pub struct StableDiffusionPaths {
    pub tokenizer: PathBuf,
    pub clip: PathBuf,
    pub unet: PathBuf,
    pub vae: PathBuf,
}

impl StableDiffusionPaths {
    /// Expected layout under `<base>/sd-v1.5/`:
    ///   tokenizer.json
    ///   text_encoder.safetensors
    ///   unet.safetensors
    ///   vae.safetensors
    pub fn under(base: &Path) -> Self {
        let dir = base.join("sd-v1.5");
        Self {
            tokenizer: dir.join("tokenizer.json"),
            clip: dir.join("text_encoder.safetensors"),
            unet: dir.join("unet.safetensors"),
            vae: dir.join("vae.safetensors"),
        }
    }
}

/// Stable Diffusion v1.5 pipeline (text → image).
pub struct StableDiffusionModel {
    config: StableDiffusionConfig,
    tokenizer: Tokenizer,
    clip: stable_diffusion::clip::ClipTextTransformer,
    unet: stable_diffusion::unet_2d::UNet2DConditionModel,
    vae: AutoEncoderKL,
    device: Device,
    dtype: DType,
}

impl StableDiffusionModel {
    /// Load v1.5 weights from local files. All inference runs in f32 for
    /// CPU compatibility (f16 requires CUDA/Metal).
    pub fn load(paths: &StableDiffusionPaths, device: &Device) -> Result<Self> {
        info!("loading Stable Diffusion v1.5 from {:?}", paths);

        let dtype = DType::F32;
        let config = StableDiffusionConfig::v1_5(None, None, None);

        let tokenizer = Tokenizer::from_file(&paths.tokenizer)
            .map_err(|e| BatataError::Inference(format!("SD tokenizer load failed: {e}")))?;

        let clip = stable_diffusion::build_clip_transformer(
            &config.clip,
            &paths.clip,
            device,
            DType::F32,
        )
        .map_err(|e| BatataError::Inference(format!("SD clip build failed: {e}")))?;

        let unet = config
            .build_unet(&paths.unet, device, 4, false, dtype)
            .map_err(|e| BatataError::Inference(format!("SD unet build failed: {e}")))?;

        let vae = config
            .build_vae(&paths.vae, device, dtype)
            .map_err(|e| BatataError::Inference(format!("SD vae build failed: {e}")))?;

        info!("Stable Diffusion v1.5 loaded");

        Ok(Self {
            config,
            tokenizer,
            clip,
            unet,
            vae,
            device: device.clone(),
            dtype,
        })
    }

    /// Download weights from HuggingFace and cache them under
    /// `<base>/sd-v1.5/`, then load.
    pub fn download_and_load(base: &Path, device: &Device) -> Result<Self> {
        let paths = download_sd_v1_5(base)?;
        Self::load(&paths, device)
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let max_len = self.config.clip.max_position_embeddings;
        let pad_id = match &self.config.clip.pad_with {
            Some(pad) => *self
                .tokenizer
                .get_vocab(true)
                .get(pad.as_str())
                .ok_or_else(|| BatataError::Inference("SD pad token missing".into()))?,
            None => *self
                .tokenizer
                .get_vocab(true)
                .get("<|endoftext|>")
                .ok_or_else(|| BatataError::Inference("SD EOS token missing".into()))?,
        };

        let mut ids = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| BatataError::Inference(format!("SD tokenize failed: {e}")))?
            .get_ids()
            .to_vec();
        if ids.len() > max_len {
            ids.truncate(max_len);
        }
        while ids.len() < max_len {
            ids.push(pad_id);
        }

        let tokens = Tensor::new(ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        self.clip
            .forward(&tokens)
            .map_err(|e| BatataError::Inference(format!("SD clip forward failed: {e}")))
    }

    /// Run denoising and return an RGB image (width × height × 3, u8).
    fn generate_internal(&self, prompt: &str, params: &ImageGenParams) -> Result<GeneratedImage> {
        // Build a fresh config honoring requested dimensions (multiples of 8).
        let width = params.width as usize;
        let height = params.height as usize;
        if width % 8 != 0 || height % 8 != 0 {
            return Err(BatataError::Inference(
                "SD width/height must be divisible by 8".into(),
            ));
        }
        let config = StableDiffusionConfig::v1_5(None, Some(height), Some(width));

        let guidance_scale = params.guidance_scale as f64;
        let use_cfg = guidance_scale > 1.0;

        let cond = self.encode_prompt(prompt)?;
        let text_embeddings = if use_cfg {
            let neg = params.negative_prompt.as_deref().unwrap_or(UNCOND_PROMPT);
            let uncond = self.encode_prompt(neg)?;
            Tensor::cat(&[&uncond, &cond], 0)
                .map_err(|e| BatataError::Inference(e.to_string()))?
        } else {
            cond
        }
        .to_dtype(self.dtype)
        .map_err(|e| BatataError::Inference(e.to_string()))?;

        let mut scheduler = config
            .build_scheduler(params.steps as usize)
            .map_err(|e| BatataError::Inference(format!("SD scheduler build failed: {e}")))?;

        let bsize = 1usize;
        let latent_shape = (bsize, 4usize, height / 8, width / 8);

        let init_noise = match params.seed {
            Some(seed) => {
                let dev = self.device.clone();
                dev.set_seed(seed)
                    .map_err(|e| BatataError::Inference(e.to_string()))?;
                Tensor::randn(0f32, 1f32, latent_shape, &dev)
            }
            None => Tensor::randn(0f32, 1f32, latent_shape, &self.device),
        }
        .map_err(|e| BatataError::Inference(e.to_string()))?;

        let mut latents = (init_noise * scheduler.init_noise_sigma())
            .and_then(|t| t.to_dtype(self.dtype))
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        let timesteps = scheduler.timesteps().to_vec();
        info!("SD denoising: {} steps @ {}x{}", timesteps.len(), width, height);

        for (i, &timestep) in timesteps.iter().enumerate() {
            let latent_model_input = if use_cfg {
                Tensor::cat(&[&latents, &latents], 0)
                    .map_err(|e| BatataError::Inference(e.to_string()))?
            } else {
                latents.clone()
            };

            let latent_model_input = scheduler
                .scale_model_input(latent_model_input, timestep)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let noise_pred = self
                .unet
                .forward(&latent_model_input, timestep as f64, &text_embeddings)
                .map_err(|e| BatataError::Inference(format!("SD unet forward failed: {e}")))?;

            let noise_pred = if use_cfg {
                let chunks = noise_pred
                    .chunk(2, 0)
                    .map_err(|e| BatataError::Inference(e.to_string()))?;
                let (uncond, cond) = (&chunks[0], &chunks[1]);
                let diff = (cond - uncond)
                    .map_err(|e| BatataError::Inference(e.to_string()))?;
                let scaled = (diff * guidance_scale)
                    .map_err(|e| BatataError::Inference(e.to_string()))?;
                (uncond + scaled).map_err(|e| BatataError::Inference(e.to_string()))?
            } else {
                noise_pred
            };

            latents = scheduler
                .step(&noise_pred, timestep, &latents)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            if (i + 1) % 5 == 0 || i + 1 == timesteps.len() {
                info!("  step {}/{}", i + 1, timesteps.len());
            }
        }

        self.decode_latents(&latents, width, height)
    }

    fn decode_latents(&self, latents: &Tensor, width: usize, height: usize) -> Result<GeneratedImage> {
        let scaled = (latents / VAE_SCALE_V1_5)
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let images = self
            .vae
            .decode(&scaled)
            .map_err(|e| BatataError::Inference(format!("SD vae decode failed: {e}")))?;

        // (images / 2 + 0.5).clamp(0,1) * 255 -> u8
        let images = ((images / 2.0).and_then(|t| t + 0.5))
            .and_then(|t| t.clamp(0f32, 1f32))
            .and_then(|t| t * 255.0)
            .and_then(|t| t.to_dtype(DType::U8))
            .and_then(|t| t.to_device(&Device::Cpu))
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        // images shape: (1, 3, H, W) — take batch 0 and move to HWC.
        let img = images
            .i(0)
            .and_then(|t| t.permute((1, 2, 0)))
            .and_then(|t| t.contiguous())
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let flat = img
            .flatten_all()
            .and_then(|t| t.to_vec1::<u8>())
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        // Encode as PNG.
        let png = encode_png_rgb8(&flat, width as u32, height as u32)?;

        // Touch D to silence unused import on some feature sets.
        let _ = D::Minus1;

        Ok(GeneratedImage {
            bytes: png,
            width: width as u32,
            height: height as u32,
        })
    }
}

#[async_trait]
impl TextToImage for StableDiffusionModel {
    fn name(&self) -> &str {
        "stable-diffusion-v1-5"
    }

    async fn generate(&self, prompt: &str) -> Result<GeneratedImage> {
        self.generate_internal(prompt, &ImageGenParams::default())
    }

    async fn generate_with_params(
        &self,
        prompt: &str,
        params: ImageGenParams,
    ) -> Result<GeneratedImage> {
        self.generate_internal(prompt, &params)
    }
}

fn encode_png_rgb8(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let img = image::RgbImage::from_raw(width, height, pixels.to_vec()).ok_or_else(|| {
        BatataError::Inference("SD: pixel buffer size mismatch for PNG encode".into())
    })?;
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| BatataError::Inference(format!("PNG encode failed: {e}")))?;
    Ok(out)
}

/// Download SD v1.5 weights into `<base>/sd-v1.5/` and return resolved paths.
///
/// Reuses files if already present. Uses the non-fp16 safetensors variants
/// so the model runs in f32 on CPU without conversion.
pub fn download_sd_v1_5(base: &Path) -> Result<StableDiffusionPaths> {
    let target_dir = base.join("sd-v1.5");
    std::fs::create_dir_all(&target_dir)?;

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| BatataError::Inference(e.to_string()))?;

    // Tokenizer from the OpenAI CLIP repo.
    let clip_tok_repo = api.model("openai/clip-vit-large-patch14".to_string());
    let sd_repo = api.model("runwayml/stable-diffusion-v1-5".to_string());

    struct Item {
        remote: &'static str,
        local: &'static str,
        from_clip_repo: bool,
    }
    let items = [
        Item { remote: "tokenizer.json",                              local: "tokenizer.json",         from_clip_repo: true  },
        Item { remote: "text_encoder/model.safetensors",              local: "text_encoder.safetensors", from_clip_repo: false },
        Item { remote: "unet/diffusion_pytorch_model.safetensors",    local: "unet.safetensors",       from_clip_repo: false },
        Item { remote: "vae/diffusion_pytorch_model.safetensors",     local: "vae.safetensors",        from_clip_repo: false },
    ];

    for it in &items {
        let dest = target_dir.join(it.local);
        if dest.exists() {
            info!("skip SD file {} (already present)", it.local);
            continue;
        }
        info!("downloading SD {}...", it.remote);
        let cached = if it.from_clip_repo {
            clip_tok_repo.get(it.remote)
        } else {
            sd_repo.get(it.remote)
        }
        .map_err(|e| BatataError::Inference(format!("SD download {} failed: {e}", it.remote)))?;
        std::fs::copy(&cached, &dest)?;
    }

    Ok(StableDiffusionPaths::under(base))
}
