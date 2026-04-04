use actix_web::{delete, get, post, web, HttpResponse};
use std::sync::Arc;

use batata_ai_core::domain::Conversation;
use batata_ai_core::repository::{ConversationMessageRepository, ConversationRepository};

#[derive(serde::Deserialize)]
pub struct CreateConversationRequest {
    pub title: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct PaginationParams {
    pub page: Option<u64>,
    pub page_size: Option<u64>,
}

#[post("/v1/conversations")]
pub async fn create_conversation(
    conv_repo: web::Data<Arc<dyn ConversationRepository>>,
    body: web::Json<CreateConversationRequest>,
    // TODO: extract tenant_id from AuthContext
) -> actix_web::Result<HttpResponse> {
    let now = chrono::Utc::now().naive_utc();
    let conversation = Conversation {
        id: uuid::Uuid::new_v4().to_string(),
        tenant_id: String::new(), // TODO: from auth context
        title: body.title.clone(),
        model: body.model.clone(),
        system_prompt: body.system_prompt.clone(),
        metadata: None,
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    let result = conv_repo
        .create(&conversation)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(result))
}

#[get("/v1/conversations")]
pub async fn list_conversations(
    conv_repo: web::Data<Arc<dyn ConversationRepository>>,
    query: web::Query<PaginationParams>,
) -> actix_web::Result<HttpResponse> {
    let page = query.page.unwrap_or(0);
    let page_size = query.page_size.unwrap_or(20);

    // TODO: get tenant_id from auth context
    let tenant_id = "";

    let conversations = conv_repo
        .find_by_tenant(tenant_id, page, page_size)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let total = conv_repo
        .count_by_tenant(tenant_id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "data": conversations,
        "total": total,
        "page": page,
        "page_size": page_size,
    })))
}

#[get("/v1/conversations/{id}")]
pub async fn get_conversation(
    conv_repo: web::Data<Arc<dyn ConversationRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let conversation = conv_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("conversation not found"))?;

    Ok(HttpResponse::Ok().json(conversation))
}

#[delete("/v1/conversations/{id}")]
pub async fn delete_conversation(
    conv_repo: web::Data<Arc<dyn ConversationRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    conv_repo
        .delete(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::NoContent().finish())
}

#[get("/v1/conversations/{id}/messages")]
pub async fn list_messages(
    msg_repo: web::Data<Arc<dyn ConversationMessageRepository>>,
    path: web::Path<String>,
    query: web::Query<PaginationParams>,
) -> actix_web::Result<HttpResponse> {
    let conversation_id = path.into_inner();
    let page = query.page.unwrap_or(0);
    let page_size = query.page_size.unwrap_or(50);

    let messages = msg_repo
        .find_by_conversation(&conversation_id, page, page_size)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let total = msg_repo
        .count_by_conversation(&conversation_id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "data": messages,
        "total": total,
        "page": page,
        "page_size": page_size,
    })))
}
