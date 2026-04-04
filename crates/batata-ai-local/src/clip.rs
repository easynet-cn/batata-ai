use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use tokenizers::Tokenizer;
use tracing::info;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::multimodal::ImageData;

const IMAGE_SIZE: usize = 224;
const MAX_TEXT_LEN: usize = 77;

/// CLIP model for text and image embeddings
pub struct ClipModel {
    model: clip::ClipModel,
    tokenizer: Tokenizer,
    device: Device,
    embed_dim: usize,
}

impl ClipModel {
    /// Download and load CLIP ViT-B/32 from HuggingFace Hub
    pub fn download_and_load(device: &Device) -> Result<Self> {
        let repo_id = "openai/clip-vit-base-patch32";
        info!("downloading CLIP model from {repo_id}...");

        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| BatataError::Inference(e.to_string()))?;
        let repo = api.model(repo_id.to_string());

        let model_path = repo
            .get("model.safetensors")
            .map_err(|e| BatataError::Inference(format!("model download failed: {e}")))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| BatataError::Inference(format!("tokenizer download failed: {e}")))?;

        Self::load(&model_path, &tokenizer_path, device)
    }

    /// Load from local files
    pub fn load(model_path: &Path, tokenizer_path: &Path, device: &Device) -> Result<Self> {
        info!("loading CLIP model from {}", model_path.display());

        let config = clip::ClipConfig::vit_base_patch32();
        let embed_dim = config.text_config.embed_dim;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.to_path_buf()], DType::F32, device)
                .map_err(|e| BatataError::Inference(format!("failed to load weights: {e}")))?
        };

        let model = clip::ClipModel::new(vb, &config)
            .map_err(|e| BatataError::Inference(format!("failed to build model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;

        info!("CLIP model loaded (embed_dim={embed_dim})");

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            embed_dim,
        })
    }

    /// Embedding vector dimension (512 for ViT-B/32)
    pub fn dimensions(&self) -> usize {
        self.embed_dim
    }

    /// Generate embeddings for text inputs
    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| BatataError::Inference(format!("tokenization failed: {e}")))?;

            let mut ids = encoding.get_ids().to_vec();
            ids.truncate(MAX_TEXT_LEN);
            // Pad to MAX_TEXT_LEN
            while ids.len() < MAX_TEXT_LEN {
                ids.push(0);
            }

            let input_ids = Tensor::new(ids.as_slice(), &self.device)
                .map_err(|e| BatataError::Inference(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let features = self
                .model
                .get_text_features(&input_ids)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            // L2 normalize
            let features = clip::div_l2_norm(&features)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let embedding: Vec<f32> = features
                .squeeze(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?
                .to_vec1()
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            all_embeddings.push(embedding);
        }

        Ok(all_embeddings)
    }

    /// Generate embedding for an image
    pub fn embed_image(&self, image: &ImageData) -> Result<Vec<f32>> {
        let tensor = preprocess_image(image, &self.device)?;

        let features = self
            .model
            .get_image_features(&tensor.unsqueeze(0).map_err(|e| BatataError::Inference(e.to_string()))?)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        let features = clip::div_l2_norm(&features)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        let embedding: Vec<f32> = features
            .squeeze(0)
            .map_err(|e| BatataError::Inference(e.to_string()))?
            .to_vec1()
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        Ok(embedding)
    }

    /// Compute cosine similarity between two embedding vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

/// Preprocess image for CLIP: resize to 224x224, normalize with CLIP stats
fn preprocess_image(image: &ImageData, device: &Device) -> Result<Tensor> {
    let (src_w, src_h) = (image.width as usize, image.height as usize);
    let channels = image.bytes.len() / (src_w * src_h);

    let mut chw = vec![0f32; 3 * IMAGE_SIZE * IMAGE_SIZE];
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.26130258, 0.27577711];

    for c in 0..3 {
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let src_x = (x as f32 * src_w as f32 / IMAGE_SIZE as f32) as usize;
                let src_y = (y as f32 * src_h as f32 / IMAGE_SIZE as f32) as usize;
                let src_x = src_x.min(src_w - 1);
                let src_y = src_y.min(src_h - 1);

                let src_idx = (src_y * src_w + src_x) * channels + c.min(channels - 1);
                let pixel = image.bytes[src_idx] as f32 / 255.0;
                chw[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] =
                    (pixel - mean[c]) / std[c];
            }
        }
    }

    Tensor::from_vec(chw, (3, IMAGE_SIZE, IMAGE_SIZE), device)
        .map_err(|e| BatataError::Inference(e.to_string()))
}
