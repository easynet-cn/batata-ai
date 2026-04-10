use async_trait::async_trait;
use futures::StreamExt;

use batata_ai_core::{
    error::{BatataError, Result},
    message::{ChatRequest, ChatResponse, ChatStream, Usage},
    provider::{Provider, ProviderCapabilities},
};

pub struct OllamaProvider {
    base_url: String,
    default_model: String,
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            default_model: "llama3".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    fn build_body(&self, req: &ChatRequest, stream: bool) -> serde_json::Value {
        let model = req
            .model
            .as_deref()
            .unwrap_or(&self.default_model)
            .to_string();

        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
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
            "stream": stream,
        });

        let mut options = serde_json::Map::new();
        if let Some(temp) = req.temperature {
            options.insert("temperature".to_string(), serde_json::json!(temp));
        }
        if let Some(max) = req.max_tokens {
            options.insert("num_predict".to_string(), serde_json::json!(max));
        }
        if !options.is_empty() {
            body["options"] = serde_json::Value::Object(options);
        }

        body
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            chat: true,
            streaming: true,
            embeddings: true,
            function_calling: false,
        }
    }

    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        let model = req
            .model
            .as_deref()
            .unwrap_or(&self.default_model)
            .to_string();
        let body = self.build_body(&req, false);

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/chat", self.base_url))
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
                "Ollama returned {status}: {resp_text}"
            )));
        }

        let data: serde_json::Value = serde_json::from_str(&resp_text)
            .map_err(|e| BatataError::Provider(format!("invalid JSON: {e}")))?;

        let content = data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // Ollama returns eval_count (output tokens) and prompt_eval_count (input tokens)
        let usage = {
            let prompt_tokens = data["prompt_eval_count"].as_u64().unwrap_or(0) as u32;
            let completion_tokens = data["eval_count"].as_u64().unwrap_or(0) as u32;
            if prompt_tokens > 0 || completion_tokens > 0 {
                Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                })
            } else {
                None
            }
        };

        Ok(ChatResponse {
            content,
            model,
            usage,
        })
    }

    async fn stream_chat(&self, req: ChatRequest) -> Result<ChatStream> {
        let body = self.build_body(&req, true);

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/chat", self.base_url))
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
                "Ollama returned {status}: {err_text}"
            )));
        }

        // Ollama streams NDJSON: each line is a complete JSON object
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

                    if line.is_empty() {
                        continue;
                    }

                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                        if let Some(content) = json["message"]["content"].as_str() {
                            if !content.is_empty() {
                                content_parts.push(content.to_string());
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
