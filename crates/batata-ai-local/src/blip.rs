use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::blip;
use tokenizers::Tokenizer;
use tracing::info;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::multimodal::ImageData;

const IMAGE_SIZE: usize = 384;

/// BLIP model for image captioning
pub struct BlipModel {
    model: blip::BlipForConditionalGeneration,
    tokenizer: Tokenizer,
    device: Device,
}

impl BlipModel {
    /// Download and load BLIP image captioning model from HuggingFace Hub
    pub fn download_and_load(device: &Device) -> Result<Self> {
        let repo_id = "Salesforce/blip-image-captioning-large";
        info!("downloading BLIP model from {repo_id}...");

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
        info!("loading BLIP model from {}", model_path.display());

        let config = blip::Config::image_captioning_large();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.to_path_buf()], DType::F32, device)
                .map_err(|e| BatataError::Inference(format!("failed to load weights: {e}")))?
        };

        let model = blip::BlipForConditionalGeneration::new(&config, vb)
            .map_err(|e| BatataError::Inference(format!("failed to build model: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| BatataError::Inference(format!("failed to load tokenizer: {e}")))?;

        info!("BLIP model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Generate a caption for an image
    pub fn caption(&mut self, image: ImageData) -> Result<String> {
        let image_tensor = self.preprocess_image(&image)?;

        // Encode image
        let image_embeds = image_tensor
            .unsqueeze(0)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        let image_embeds = self
            .model
            .vision_model()
            .forward(&image_embeds)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        // Decode caption greedily
        let bos_token_id = 30522u32; // [DEC] token for BLIP
        let eos_token_id = self.tokenizer.token_to_id("[SEP]").unwrap_or(102);
        let pad_token_id = self.tokenizer.token_to_id("[PAD]").unwrap_or(0);

        let mut token_ids = vec![bos_token_id];
        let max_tokens = 50;

        for _ in 0..max_tokens {
            let input_ids = Tensor::new(token_ids.as_slice(), &self.device)
                .map_err(|e| BatataError::Inference(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let logits = self
                .model
                .text_decoder()
                .forward(&input_ids, &image_embeds)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let logits = logits
                .i((.., token_ids.len() - 1, ..))
                .map_err(|e| BatataError::Inference(e.to_string()))?
                .squeeze(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            let next_token = logits
                .argmax(0)
                .map_err(|e| BatataError::Inference(e.to_string()))?
                .to_scalar::<u32>()
                .map_err(|e| BatataError::Inference(e.to_string()))?;

            if next_token == eos_token_id || next_token == pad_token_id {
                break;
            }

            token_ids.push(next_token);
        }

        // Skip the BOS token
        let decode_tokens: Vec<u32> = token_ids.into_iter().skip(1).collect();
        let caption = self
            .tokenizer
            .decode(&decode_tokens, true)
            .map_err(|e| BatataError::Inference(format!("decode failed: {e}")))?;

        self.model.reset_kv_cache();

        Ok(caption.trim().to_string())
    }

    /// Preprocess raw image data to tensor (resize to 384x384, normalize)
    fn preprocess_image(&self, image: &ImageData) -> Result<Tensor> {
        // Simple bilinear resize to IMAGE_SIZE x IMAGE_SIZE
        let (src_w, src_h) = (image.width as usize, image.height as usize);
        let channels = image.bytes.len() / (src_w * src_h);
        let rgb_channels = 3.min(channels);

        let mut resized = vec![0f32; IMAGE_SIZE * IMAGE_SIZE * 3];

        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let src_x = (x as f32 * src_w as f32 / IMAGE_SIZE as f32) as usize;
                let src_y = (y as f32 * src_h as f32 / IMAGE_SIZE as f32) as usize;
                let src_x = src_x.min(src_w - 1);
                let src_y = src_y.min(src_h - 1);

                let src_idx = (src_y * src_w + src_x) * channels;
                let dst_idx = (y * IMAGE_SIZE + x) * 3;

                for c in 0..rgb_channels {
                    // Normalize to [0, 1] then apply ImageNet normalization
                    let pixel = image.bytes[src_idx + c] as f32 / 255.0;
                    let mean = [0.48145466, 0.4578275, 0.40821073][c];
                    let std = [0.26862954, 0.26130258, 0.27577711][c];
                    resized[dst_idx + c] = (pixel - mean) / std;
                }
            }
        }

        // Convert from HWC to CHW format
        let mut chw = vec![0f32; 3 * IMAGE_SIZE * IMAGE_SIZE];
        for c in 0..3 {
            for y in 0..IMAGE_SIZE {
                for x in 0..IMAGE_SIZE {
                    chw[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] =
                        resized[(y * IMAGE_SIZE + x) * 3 + c];
                }
            }
        }

        Tensor::from_vec(chw, (3, IMAGE_SIZE, IMAGE_SIZE), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))
    }
}
