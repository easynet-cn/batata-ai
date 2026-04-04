use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    Chat,
    Embedding,
    ImageGeneration,
    SpeechToText,
    TextToSpeech,
    ObjectDetection,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "chat"),
            Self::Embedding => write!(f, "embedding"),
            Self::ImageGeneration => write!(f, "image_generation"),
            Self::SpeechToText => write!(f, "speech_to_text"),
            Self::TextToSpeech => write!(f, "text_to_speech"),
            Self::ObjectDetection => write!(f, "object_detection"),
        }
    }
}

impl std::str::FromStr for ModelType {
    type Err = crate::error::BatataError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "chat" => Ok(Self::Chat),
            "embedding" => Ok(Self::Embedding),
            "image_generation" => Ok(Self::ImageGeneration),
            "speech_to_text" => Ok(Self::SpeechToText),
            "text_to_speech" => Ok(Self::TextToSpeech),
            "object_detection" => Ok(Self::ObjectDetection),
            other => Err(crate::error::BatataError::Config(format!(
                "unknown model type: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub id: String,
    pub name: String,
    pub model_type: ModelType,
    pub context_length: Option<i32>,
    pub description: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub enabled: bool,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}
