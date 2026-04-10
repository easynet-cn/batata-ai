pub mod openai_compat;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openrouter")]
pub mod openrouter;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "together")]
pub mod together;

#[cfg(feature = "mistral")]
pub mod mistral;

#[cfg(feature = "siliconflow")]
pub mod siliconflow;

#[cfg(feature = "zhipu")]
pub mod zhipu;
