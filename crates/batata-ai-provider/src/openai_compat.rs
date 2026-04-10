//! Shared HTTP client for OpenAI-compatible APIs (OpenAI, OpenRouter, DeepSeek, etc.)

use batata_ai_core::{
    error::{BatataError, Result},
    message::{ChatRequest, ChatResponse, ChatStream, Usage},
};
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

/// Configuration for an OpenAI-compatible API endpoint.
pub struct OpenAiCompatConfig {
    pub base_url: String,
    pub api_key: String,
    pub default_model: String,
    /// Extra headers (e.g. OpenRouter's `HTTP-Referer`, `X-Title`)
    pub extra_headers: HeaderMap,
}

/// Build the JSON request body from a `ChatRequest`.
fn build_request_body(req: &ChatRequest, model: &str, stream: bool) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| {
            let mut obj = serde_json::json!({
                "role": m.role,
                "content": m.content,
            });
            if let Some(name) = &m.name {
                obj["name"] = serde_json::Value::String(name.clone());
            }
            obj
        })
        .collect();

    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });

    if let Some(temp) = req.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(max) = req.max_tokens {
        body["max_tokens"] = serde_json::json!(max);
    }

    body
}

fn build_headers(config: &OpenAiCompatConfig) -> Result<HeaderMap> {
    let mut headers = config.extra_headers.clone();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", config.api_key))
            .map_err(|e| BatataError::Provider(format!("invalid api key header: {e}")))?,
    );
    Ok(headers)
}

fn parse_usage(data: &serde_json::Value) -> Option<Usage> {
    let u = data.get("usage")?;
    Some(Usage {
        prompt_tokens: u.get("prompt_tokens")?.as_u64()? as u32,
        completion_tokens: u.get("completion_tokens")?.as_u64()? as u32,
        total_tokens: u.get("total_tokens")?.as_u64()? as u32,
    })
}

/// Non-streaming chat completion.
pub async fn chat(config: &OpenAiCompatConfig, req: ChatRequest) -> Result<ChatResponse> {
    let model = req
        .model
        .as_deref()
        .unwrap_or(&config.default_model)
        .to_string();

    let body = build_request_body(&req, &model, false);
    let headers = build_headers(config)?;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/chat/completions", config.base_url))
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
            "API returned {status}: {resp_text}"
        )));
    }

    let data: serde_json::Value = serde_json::from_str(&resp_text)
        .map_err(|e| BatataError::Provider(format!("invalid JSON response: {e}")))?;

    let content = data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let usage = parse_usage(&data);

    Ok(ChatResponse {
        content,
        model,
        usage,
    })
}

/// Streaming chat completion via SSE.
pub async fn stream_chat(config: &OpenAiCompatConfig, req: ChatRequest) -> Result<ChatStream> {
    let model = req
        .model
        .as_deref()
        .unwrap_or(&config.default_model)
        .to_string();

    let body = build_request_body(&req, &model, true);
    let headers = build_headers(config)?;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/chat/completions", config.base_url))
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
            "API returned {status}: {err_text}"
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

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(delta) = json["choices"][0]["delta"]["content"].as_str() {
                            content_parts.push(delta.to_string());
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
