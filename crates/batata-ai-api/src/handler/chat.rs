use actix_web::{post, web, HttpResponse};
use futures::StreamExt;

use batata_ai_core::domain::{ConversationMessage, RequestLog, RequestStatus};
use batata_ai_core::message::{ChatRequest, Message};
use batata_ai_core::routing::{RouteCandidate, RoutingContext, RoutingPriority};

use crate::middleware::auth::AuthContext;

#[derive(serde::Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<MessageInput>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// If true, return a streaming SSE response.
    pub stream: Option<bool>,
    /// Optional conversation_id to append messages to.
    pub conversation_id: Option<String>,
    /// Knowledge base to retrieve context from before answering.
    pub knowledge_base_id: Option<String>,
    /// Per-request RAG retrieval tuning.
    #[serde(default)]
    pub rag: Option<RagOptions>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct RagOptions {
    #[serde(default = "default_rag_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
}

impl Default for RagOptions {
    fn default() -> Self {
        Self {
            top_k: default_rag_top_k(),
            min_score: None,
        }
    }
}

fn default_rag_top_k() -> usize {
    5
}

#[derive(serde::Deserialize)]
pub struct MessageInput {
    pub role: String,
    pub content: String,
}

/// OpenAI-compatible chat completions endpoint.
#[post("/v1/chat/completions")]
pub async fn chat_completions(
    state: web::Data<crate::state::AppState>,
    body: web::Json<ChatCompletionRequest>,
    auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let tenant_id = auth
        .as_ref()
        .map(|a| a.tenant_id.clone())
        .unwrap_or_default();

    let mut messages: Vec<Message> = body
        .messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => batata_ai_core::message::Role::System,
                "assistant" => batata_ai_core::message::Role::Assistant,
                "tool" => batata_ai_core::message::Role::Tool,
                _ => batata_ai_core::message::Role::User,
            };
            Message {
                role,
                content: m.content.clone(),
                name: None,
            }
        })
        .collect();

    // --- RAG context injection (before cache lookup so the key captures it) ---
    if let (Some(kb_id), Some(pipeline)) =
        (body.knowledge_base_id.as_ref(), state.rag_pipeline.clone())
    {
        let opts = body.rag.clone().unwrap_or_default();
        // Use the *last user message* as the retrieval query.
        let user_query = messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, batata_ai_core::message::Role::User))
            .map(|m| m.content.clone())
            .unwrap_or_default();

        if !user_query.is_empty() {
            match pipeline.search(kb_id, &user_query, opts.top_k).await {
                Ok(hits) => {
                    let filtered: Vec<_> = hits
                        .into_iter()
                        .filter(|h| opts.min_score.map_or(true, |m| h.score >= m))
                        .collect();
                    if !filtered.is_empty() {
                        let ctx = filtered
                            .iter()
                            .enumerate()
                            .map(|(i, h)| format!("[{}] {}", i + 1, h.chunk.text))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        let preamble = format!(
                            "You are given the following retrieved context from knowledge base \
                             '{kb_id}'. Use it to answer the user's question. If the context is \
                             not relevant, answer from your own knowledge and say so briefly.\n\n\
                             --- Context ---\n{ctx}\n--- End Context ---"
                        );
                        messages.insert(
                            0,
                            Message {
                                role: batata_ai_core::message::Role::System,
                                content: preamble,
                                name: None,
                            },
                        );
                        tracing::debug!(
                            kb_id,
                            hits = filtered.len(),
                            "RAG context injected into chat request"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(kb_id, error = %e, "RAG search failed, continuing without context");
                }
            }
        }
    }

    let request = ChatRequest {
        messages,
        model: Some(body.model.clone()),
        temperature: body.temperature,
        max_tokens: body.max_tokens,
    };

    // --- Cache lookup ---
    let cache_key = state.cache_key_strategy.generate_key(&request);
    if let Some(ref cache) = state.cache {
        match cache.get(&cache_key).await {
            Ok(Some(entry)) => {
                tracing::debug!(key = %cache_key, "cache hit");
                let response = entry.response;
                return Ok(HttpResponse::Ok().json(serde_json::json!({
                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion",
                    "created": chrono::Utc::now().timestamp(),
                    "model": response.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.content,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": response.usage.map(|u| serde_json::json!({
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                    })),
                })));
            }
            Ok(None) => {
                tracing::debug!(key = %cache_key, "cache miss");
            }
            Err(e) => {
                tracing::warn!(key = %cache_key, error = %e, "cache get failed, proceeding without cache");
            }
        }
    }

    // Build candidates from registered providers
    let candidates: Vec<RouteCandidate> = state
        .router
        .list_providers()
        .iter()
        .map(|(id, name)| RouteCandidate {
            provider_id: id.clone(),
            provider_name: name.clone(),
            model_identifier: body.model.clone(),
            priority: 0,
            score: 0.0,
        })
        .collect();

    let ctx = RoutingContext {
        request: request.clone(),
        required_model: Some(body.model.clone()),
        required_capabilities: vec![],
        priority: RoutingPriority::Quality,
        metadata: Default::default(),
    };

    // --- SSE streaming path ---
    if body.stream.unwrap_or(false) {
        let stream = state
            .router
            .route_stream(ctx, candidates)
            .await
            .map_err(actix_web::error::ErrorInternalServerError)?;

        let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model = body.model.clone();
        let created = chrono::Utc::now().timestamp();

        let sse_stream = stream.map(move |chunk| {
            match chunk {
                Ok(content) => {
                    let data = serde_json::json!({
                        "id": &chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": &model,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": content },
                            "finish_reason": serde_json::Value::Null,
                        }]
                    });
                    Ok::<_, actix_web::Error>(web::Bytes::from(format!("data: {}\n\n", data)))
                }
                Err(e) => {
                    tracing::error!(error = %e, "stream chunk error");
                    Ok(web::Bytes::from("data: [DONE]\n\n"))
                }
            }
        });

        let done_stream = futures::stream::once(async {
            Ok::<_, actix_web::Error>(web::Bytes::from("data: [DONE]\n\n"))
        });

        let full_stream = sse_stream.chain(done_stream);

        return Ok(HttpResponse::Ok()
            .content_type("text/event-stream")
            .streaming(full_stream));
    }

    let start = std::time::Instant::now();
    let result = state.router.route(ctx, candidates).await;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            let now = chrono::Utc::now().naive_utc();

            // --- Cache store ---
            if let Some(ref cache) = state.cache {
                // Default TTL: 5 minutes (300 seconds)
                if let Err(e) = cache.set(&cache_key, &response, Some(300)).await {
                    tracing::warn!(key = %cache_key, error = %e, "failed to store response in cache");
                }
            }

            // Record request log
            let log = RequestLog {
                id: uuid::Uuid::new_v4().to_string(),
                tenant_id: tenant_id.clone(),
                provider_id: String::new(),
                provider_name: String::new(),
                model_identifier: response.model.clone(),
                routing_policy: state.router.policy_name().to_string(),
                status: RequestStatus::Success,
                latency_ms: elapsed_ms,
                prompt_tokens: response.usage.as_ref().map(|u| u.prompt_tokens),
                completion_tokens: response.usage.as_ref().map(|u| u.completion_tokens),
                total_tokens: response.usage.as_ref().map(|u| u.total_tokens),
                estimated_cost: None,
                error_message: None,
                metadata: None,
                created_at: now,
            };
            let _ = state.log_repo.create(&log).await;

            // If conversation_id provided, save assistant message
            if let Some(conv_id) = &body.conversation_id {
                let msg = ConversationMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    conversation_id: conv_id.clone(),
                    tenant_id: tenant_id.clone(),
                    role: "assistant".to_string(),
                    content: response.content.clone(),
                    model: Some(response.model.clone()),
                    usage: response.usage.as_ref().map(|u| {
                        serde_json::json!({
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens,
                        })
                    }),
                    latency_ms: Some(elapsed_ms as i64),
                    metadata: None,
                    created_at: now,
                };
                let _ = state.message_repo.create(&msg).await;
            }

            // OpenAI-compatible response
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                "object": "chat.completion",
                "created": chrono::Utc::now().timestamp(),
                "model": response.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content,
                    },
                    "finish_reason": "stop",
                }],
                "usage": response.usage.map(|u| serde_json::json!({
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                })),
            })))
        }
        Err(e) => {
            // Log failure
            let log = RequestLog {
                id: uuid::Uuid::new_v4().to_string(),
                tenant_id,
                provider_id: String::new(),
                provider_name: String::new(),
                model_identifier: body.model.clone(),
                routing_policy: state.router.policy_name().to_string(),
                status: RequestStatus::Failed,
                latency_ms: elapsed_ms,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                estimated_cost: None,
                error_message: Some(e.to_string()),
                metadata: None,
                created_at: chrono::Utc::now().naive_utc(),
            };
            let _ = state.log_repo.create(&log).await;

            Err(actix_web::error::ErrorInternalServerError(e.to_string()))
        }
    }
}
