use std::sync::Arc;

use batata_ai_core::domain::ProviderDefinition;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::provider::Provider;

/// Create a `Provider` instance from a database `ProviderDefinition`.
///
/// Dispatches on `provider_type`:
/// - `"local"` — candle-based local inference (batata-ai-local)
/// - `"openai"` — OpenAI API
/// - `"anthropic"` — Anthropic API
/// - `"ollama"` — Ollama local server
/// - `"openrouter"` — OpenRouter API
/// - `"deepseek"` — DeepSeek API
/// - `"groq"` — Groq API (ultra-fast inference)
/// - `"together"` — Together AI API
/// - `"mistral"` — Mistral AI API
/// - `"siliconflow"` — SiliconFlow API (硅基流动)
/// - `"zhipu"` — Zhipu AI API (智谱AI / GLM)
pub fn create_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    match def.provider_type.as_str() {
        "local" => create_local_provider(def),
        "openai" => create_openai_provider(def),
        "anthropic" => create_anthropic_provider(def),
        "ollama" => create_ollama_provider(def),
        "openrouter" => create_openrouter_provider(def),
        "deepseek" => create_deepseek_provider(def),
        "groq" => create_groq_provider(def),
        "together" => create_together_provider(def),
        "mistral" => create_mistral_provider(def),
        "siliconflow" => create_siliconflow_provider(def),
        "zhipu" => create_zhipu_provider(def),
        other => Err(BatataError::Provider(format!(
            "unsupported provider type: '{other}'"
        ))),
    }
}

/// Create a local candle provider.
///
/// Expected `config` JSON:
/// ```json
/// {
///   "model_name": "qwen3",                   // required
///   "use_gpu": false,                         // optional, default false
///   "model_path": "/path/to/model.gguf",      // optional, omit to auto-download
///   "tokenizer_path": "/path/to/tokenizer.json", // optional, required if model_path set
///   "max_tokens": 512,                        // optional
///   "temperature": 0.7                        // optional
/// }
/// ```
fn create_local_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let config = def.config.as_ref().ok_or_else(|| {
        BatataError::Provider("local provider requires 'config' with 'model_name'".into())
    })?;

    let model_name = config["model_name"]
        .as_str()
        .ok_or_else(|| BatataError::Provider("config.model_name is required".into()))?;

    let use_gpu = config["use_gpu"].as_bool().unwrap_or(false);

    let mut provider = if let (Some(model_path), Some(tokenizer_path)) = (
        config["model_path"].as_str(),
        config["tokenizer_path"].as_str(),
    ) {
        batata_ai_local::provider::LocalProvider::from_local(model_name, model_path, tokenizer_path)
    } else {
        batata_ai_local::provider::LocalProvider::new(model_name)
    };

    provider = provider.with_gpu(use_gpu);

    if let Some(max_tokens) = config["max_tokens"].as_u64() {
        let mut params = batata_ai_local::inference::GenerationParams::default();
        params.max_tokens = max_tokens as usize;
        if let Some(temp) = config["temperature"].as_f64() {
            params.temperature = temp;
        }
        provider = provider.with_generation_params(params);
    } else if let Some(temp) = config["temperature"].as_f64() {
        let mut params = batata_ai_local::inference::GenerationParams::default();
        params.temperature = temp;
        provider = provider.with_generation_params(params);
    }

    Ok(Arc::new(provider))
}

fn create_openai_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("openai provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::openai::OpenAiProvider::new(api_key);

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }
    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_anthropic_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("anthropic provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::anthropic::AnthropicProvider::new(api_key);

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }
    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_ollama_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let mut provider = batata_ai_provider::ollama::OllamaProvider::new();

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }

    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_openrouter_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("openrouter provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::openrouter::OpenRouterProvider::new(api_key);

    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_deepseek_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("deepseek provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::deepseek::DeepSeekProvider::new(api_key);

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }
    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_groq_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("groq provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::groq::GroqProvider::new(api_key);

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }
    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_together_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("together provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::together::TogetherProvider::new(api_key);

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }
    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_mistral_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("mistral provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::mistral::MistralProvider::new(api_key);

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }
    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_siliconflow_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("siliconflow provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::siliconflow::SiliconFlowProvider::new(api_key);

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }
    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}

fn create_zhipu_provider(def: &ProviderDefinition) -> Result<Arc<dyn Provider>> {
    let api_key = def
        .api_key
        .as_deref()
        .ok_or_else(|| BatataError::Provider("zhipu provider requires api_key".into()))?;

    let mut provider = batata_ai_provider::zhipu::ZhipuProvider::new(api_key);

    if let Some(url) = &def.base_url {
        provider = provider.with_base_url(url);
    }
    if let Some(model) = def.config.as_ref().and_then(|c| c["default_model"].as_str()) {
        provider = provider.with_model(model);
    }

    Ok(Arc::new(provider))
}
