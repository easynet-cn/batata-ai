use thiserror::Error;

#[derive(Debug, Error)]
pub enum BatataError {
    #[error("provider error: {0}")]
    Provider(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("skill error: {0}")]
    Skill(String),

    #[error("prompt error: {0}")]
    Prompt(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("mcp error: {0}")]
    Mcp(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, BatataError>;
