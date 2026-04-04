use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the chat tool
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChatParams {
    /// The user message to send
    pub message: String,
    /// Optional system prompt
    pub system: Option<String>,
    /// Optional temperature (0.0 - 1.0)
    pub temperature: Option<f32>,
    /// Optional maximum tokens to generate
    pub max_tokens: Option<u32>,
}

/// Parameters for listing available providers
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListProvidersParams {}

/// Parameters for listing available skills
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListSkillsParams {}

/// Parameters for executing a skill
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExecuteSkillParams {
    /// Name of the skill to execute
    pub skill_name: String,
    /// JSON parameters to pass to the skill
    pub params: serde_json::Value,
}

/// Result of listing providers
#[derive(Debug, Serialize, JsonSchema)]
pub struct ProvidersResult {
    pub providers: Vec<ProviderInfo>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ProviderInfo {
    pub name: String,
    pub chat: bool,
    pub streaming: bool,
    pub embeddings: bool,
    pub function_calling: bool,
}

/// Result of listing skills
#[derive(Debug, Serialize, JsonSchema)]
pub struct SkillsResult {
    pub skills: Vec<SkillInfo>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
}

/// Parameters for speech-to-text transcription
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TranscribeParams {
    /// Path to the WAV audio file to transcribe
    pub file_path: String,
}

/// Parameters for image captioning
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CaptionParams {
    /// Path to the image file (PNG/JPEG)
    pub file_path: String,
}

/// Parameters for text embedding
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EmbedTextParams {
    /// Text to generate embeddings for
    pub texts: Vec<String>,
}

/// Result of text embedding
#[derive(Debug, Serialize, JsonSchema)]
pub struct EmbedResult {
    /// Embedding vectors
    pub embeddings: Vec<Vec<f32>>,
    /// Vector dimensions
    pub dimensions: usize,
}
