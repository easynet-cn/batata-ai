pub use batata_ai_core as core;
pub use batata_ai_local as local;
pub use batata_ai_mcp as mcp;
pub use batata_ai_prompt as prompt;
pub use batata_ai_provider as provider;

// Re-export key types for convenience
pub use batata_ai_core::config::BatataConfig;
pub use batata_ai_core::error::{BatataError, Result};
pub use batata_ai_core::message::{ChatRequest, ChatResponse, Message};
pub use batata_ai_core::multimodal;
pub use batata_ai_core::provider::{Provider, ProviderRegistry};
pub use batata_ai_core::skill::{Skill, SkillRegistry};
