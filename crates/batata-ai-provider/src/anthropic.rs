use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};

use batata_ai_core::{
    error::{BatataError, Result},
    message::{ChatRequest, ChatResponse, ChatStream, Role, Usage},
    provider::{Provider, ProviderCapabilities},
};

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    default_model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            default_model: "claude-sonnet-4-6-20260404".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key)
                .map_err(|e| BatataError::Provider(format!("invalid api key: {e}")))?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static("2023-06-01"),
        );
        Ok(headers)
    }

    fn build_body(&self, req: &ChatRequest, stream: bool) -> serde_json::Value {
        let model = req.model.as_deref().unwrap_or(&self.default_model);

        // Extract system message (Anthropic uses a top-level "system" field)
        let system_msg: Option<String> = req
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .map(|m| m.content.clone())
            .reduce(|a, b| format!("{a}\n{b}"));

        // Non-system messages
        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .filter(|m| !matches!(m.role, Role::System))
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens.unwrap_or(4096),
            "stream": stream,
        });

        if let Some(sys) = system_msg {
            body["system"] = serde_json::Value::String(sys);
        }
        if let Some(temp) = req.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        body
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            chat: true,
            streaming: true,
            embeddings: false,
            function_calling: true,
        }
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        let model = req
            .model
            .as_deref()
            .unwrap_or(&self.default_model)
            .to_string();
        let body = self.build_body(&req, false);
        let headers = self.build_headers()?;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/messages", self.base_url))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| BatataError::Provider(e.to_string()))?;

        let status = resp.status();
        let resp_text = resp
            .text()
            .await
            .map_err(|e| BatataError::Provider(e.to_string()))?;

        if !status.is_success() {
            return Err(BatataError::Provider(format!(
                "Anthropic API returned {status}: {resp_text}"
            )));
        }

        let data: serde_json::Value = serde_json::from_str(&resp_text)
            .map_err(|e| BatataError::Provider(format!("invalid JSON: {e}")))?;

        // Anthropic returns content as array: [{"type":"text","text":"..."}]
        let content: String = data["content"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter(|c| c["type"].as_str() == Some("text"))
                    .filter_map(|c| c["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        let usage = data.get("usage").map(|u| Usage {
            prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: (u["input_tokens"].as_u64().unwrap_or(0)
                + u["output_tokens"].as_u64().unwrap_or(0)) as u32,
        });

        Ok(ChatResponse {
            content,
            model,
            usage,
        })
    }

    async fn stream_chat(&self, req: ChatRequest) -> Result<ChatStream> {
        let body = self.build_body(&req, true);
        let headers = self.build_headers()?;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/messages", self.base_url))
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| BatataError::Provider(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            let err_text = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(BatataError::Provider(format!(
                "Anthropic API returned {status}: {err_text}"
            )));
        }

        let byte_stream = resp.bytes_stream();
        let mut buffer = String::new();

        let stream = byte_stream
            .map(move |chunk| {
                let chunk = chunk.map_err(|e| BatataError::Provider(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);
                buffer.push_str(&text);

                let mut content_parts = Vec::new();

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() || line.starts_with("event:") {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        let data = data.trim();
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                            // content_block_delta events
                            if json["type"].as_str() == Some("content_block_delta") {
                                if let Some(text) = json["delta"]["text"].as_str() {
                                    content_parts.push(text.to_string());
                                }
                            }
                        }
                    }
                }

                Ok(content_parts.join(""))
            })
            .filter(|result| {
                let keep = match result {
                    Ok(s) => !s.is_empty(),
                    Err(_) => true,
                };
                futures::future::ready(keep)
            });

        Ok(Box::pin(stream))
    }
}
