//! Multimodal AI capability traits.
//!
//! Each trait represents a specific AI modality beyond text chat:
//! - [`SpeechToText`] — Audio transcription (Whisper)
//! - [`TextToSpeech`] — Speech synthesis (Parler TTS)
//! - [`TextToImage`] — Image generation (Stable Diffusion)
//! - [`ImageToText`] — Image captioning / visual Q&A (BLIP)
//! - [`ObjectDetection`] — Detect objects in images (YOLO)
//! - [`Embedding`] — Text/image embedding vectors (CLIP, BERT)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

// ── Common types ──────────────────────────────────────────────────

/// Raw audio data with format metadata
#[derive(Debug, Clone)]
pub struct AudioData {
    /// PCM samples (mono, f32)
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g., 16000 for Whisper)
    pub sample_rate: u32,
}

/// Raw image data
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Raw pixel bytes (RGB or RGBA)
    pub bytes: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

/// A detected object in an image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    /// Object class label
    pub label: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Bounding box: (x_min, y_min, x_max, y_max) normalized to [0, 1]
    pub bbox: (f32, f32, f32, f32),
}

/// A segment of transcribed text with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub text: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
}

/// Full transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Full transcribed text
    pub text: String,
    /// Optional per-segment timing
    pub segments: Vec<TranscriptionSegment>,
    /// Detected language (if applicable)
    pub language: Option<String>,
}

/// Image generation result
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// PNG or JPEG bytes
    pub bytes: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

// ── Traits ────────────────────────────────────────────────────────

/// Speech-to-Text: transcribe audio to text (e.g., Whisper)
#[async_trait]
pub trait SpeechToText: Send + Sync {
    fn name(&self) -> &str;

    /// Transcribe audio data to text
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult>;

    /// Transcribe from a file path
    async fn transcribe_file(&self, path: &std::path::Path) -> Result<TranscriptionResult>;
}

/// Text-to-Speech: generate speech from text (e.g., Parler TTS)
#[async_trait]
pub trait TextToSpeech: Send + Sync {
    fn name(&self) -> &str;

    /// Generate audio from text
    async fn synthesize(&self, text: &str) -> Result<AudioData>;

    /// Generate audio with a voice/style description
    async fn synthesize_with_style(&self, text: &str, style: &str) -> Result<AudioData>;
}

/// Text-to-Image: generate images from text prompts (e.g., Stable Diffusion)
#[async_trait]
pub trait TextToImage: Send + Sync {
    fn name(&self) -> &str;

    /// Generate an image from a text prompt
    async fn generate(&self, prompt: &str) -> Result<GeneratedImage>;

    /// Generate with additional parameters
    async fn generate_with_params(
        &self,
        prompt: &str,
        params: ImageGenParams,
    ) -> Result<GeneratedImage>;
}

/// Parameters for image generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenParams {
    /// Negative prompt (what to avoid)
    pub negative_prompt: Option<String>,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Number of inference steps
    pub steps: u32,
    /// Guidance scale (CFG)
    pub guidance_scale: f32,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for ImageGenParams {
    fn default() -> Self {
        Self {
            negative_prompt: None,
            width: 512,
            height: 512,
            steps: 20,
            guidance_scale: 7.5,
            seed: None,
        }
    }
}

/// Image-to-Text: image captioning and visual Q&A (e.g., BLIP)
#[async_trait]
pub trait ImageToText: Send + Sync {
    fn name(&self) -> &str;

    /// Generate a caption for an image
    async fn caption(&self, image: ImageData) -> Result<String>;

    /// Answer a question about an image (Visual QA)
    async fn visual_qa(&self, image: ImageData, question: &str) -> Result<String>;
}

/// Object Detection: detect and locate objects in images (e.g., YOLO)
#[async_trait]
pub trait ObjectDetection: Send + Sync {
    fn name(&self) -> &str;

    /// Detect objects in an image
    async fn detect(&self, image: ImageData) -> Result<Vec<DetectedObject>>;

    /// Detect with a confidence threshold
    async fn detect_with_threshold(
        &self,
        image: ImageData,
        threshold: f32,
    ) -> Result<Vec<DetectedObject>>;
}

/// Text/Image Embedding: generate embedding vectors (e.g., CLIP, BERT, Sentence Transformers)
#[async_trait]
pub trait Embedding: Send + Sync {
    fn name(&self) -> &str;

    /// Embedding vector dimension
    fn dimensions(&self) -> usize;

    /// Generate embeddings for text inputs
    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Generate embedding for a single image (if supported)
    async fn embed_image(&self, image: ImageData) -> Result<Vec<f32>> {
        let _ = image;
        Err(crate::error::BatataError::Provider(
            "image embedding not supported".into(),
        ))
    }
}
