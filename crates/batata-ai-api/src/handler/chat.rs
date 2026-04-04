use actix_web::{post, web, HttpResponse};

use batata_ai_core::message::{ChatRequest, Message};

#[derive(serde::Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<MessageInput>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(serde::Deserialize)]
pub struct MessageInput {
    pub role: String,
    pub content: String,
}

/// OpenAI-compatible chat completions endpoint.
#[post("/v1/chat/completions")]
pub async fn chat_completions(
    _state: web::Data<crate::state::AppState>,
    body: web::Json<ChatCompletionRequest>,
) -> actix_web::Result<HttpResponse> {
    let messages: Vec<Message> = body
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

    let _request = ChatRequest {
        messages,
        model: Some(body.model.clone()),
        temperature: body.temperature,
        max_tokens: body.max_tokens,
    };

    // TODO: Use RouterService to route the request
    // let response = state.router_service.chat(&body.model, request, priority).await?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "TODO: implement routing",
            },
            "finish_reason": "stop",
        }],
    })))
}
