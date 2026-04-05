use actix_web::{delete, get, post, put, web, HttpResponse};
use std::sync::Arc;

use batata_ai_core::domain::{ModelDefinition, ModelType};
use batata_ai_core::repository::ModelRepository;

#[derive(serde::Deserialize)]
pub struct CreateModelRequest {
    pub name: String,
    pub model_type: ModelType,
    pub context_length: Option<i32>,
    pub description: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
pub struct UpdateModelRequest {
    pub name: Option<String>,
    pub model_type: Option<ModelType>,
    pub context_length: Option<i32>,
    pub description: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub enabled: Option<bool>,
}

#[post("/v1/admin/models")]
pub async fn create_model(
    model_repo: web::Data<Arc<dyn ModelRepository>>,
    body: web::Json<CreateModelRequest>,
) -> actix_web::Result<HttpResponse> {
    let now = chrono::Utc::now().naive_utc();
    let model = ModelDefinition {
        id: uuid::Uuid::new_v4().to_string(),
        name: body.name.clone(),
        model_type: body.model_type.clone(),
        context_length: body.context_length,
        description: body.description.clone(),
        metadata: body.metadata.clone(),
        enabled: true,
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    let result = model_repo
        .create(&model)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(result))
}

#[get("/v1/admin/models")]
pub async fn list_models_admin(
    model_repo: web::Data<Arc<dyn ModelRepository>>,
) -> actix_web::Result<HttpResponse> {
    let models = model_repo
        .find_all()
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "data": models,
    })))
}

#[put("/v1/admin/models/{id}")]
pub async fn update_model(
    model_repo: web::Data<Arc<dyn ModelRepository>>,
    path: web::Path<String>,
    body: web::Json<UpdateModelRequest>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let existing = model_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("model not found"))?;

    let updated = ModelDefinition {
        id: existing.id,
        name: body.name.clone().unwrap_or(existing.name),
        model_type: body.model_type.clone().unwrap_or(existing.model_type),
        context_length: body.context_length.or(existing.context_length),
        description: body.description.clone().or(existing.description),
        metadata: body.metadata.clone().or(existing.metadata),
        enabled: body.enabled.unwrap_or(existing.enabled),
        created_at: existing.created_at,
        updated_at: chrono::Utc::now().naive_utc(),
        deleted_at: existing.deleted_at,
    };

    let result = model_repo
        .update(&updated)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(result))
}

#[delete("/v1/admin/models/{id}")]
pub async fn delete_model(
    model_repo: web::Data<Arc<dyn ModelRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    model_repo
        .delete(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::NoContent().finish())
}
