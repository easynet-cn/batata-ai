use actix_web::{delete, get, post, put, web, HttpResponse};
use std::sync::Arc;

use batata_ai_core::domain::ProviderDefinition;
use batata_ai_core::repository::ProviderRepository;

#[derive(serde::Deserialize)]
pub struct CreateProviderRequest {
    pub name: String,
    pub provider_type: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub config: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
pub struct UpdateProviderRequest {
    pub name: Option<String>,
    pub provider_type: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub config: Option<serde_json::Value>,
    pub enabled: Option<bool>,
}

#[post("/v1/admin/providers")]
pub async fn create_provider(
    provider_repo: web::Data<Arc<dyn ProviderRepository>>,
    body: web::Json<CreateProviderRequest>,
) -> actix_web::Result<HttpResponse> {
    let now = chrono::Utc::now().naive_utc();
    let provider = ProviderDefinition {
        id: uuid::Uuid::new_v4().to_string(),
        name: body.name.clone(),
        provider_type: body.provider_type.clone(),
        api_key: body.api_key.clone(),
        base_url: body.base_url.clone(),
        config: body.config.clone(),
        enabled: true,
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    let result = provider_repo
        .create(&provider)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(result))
}

#[get("/v1/admin/providers")]
pub async fn list_providers(
    provider_repo: web::Data<Arc<dyn ProviderRepository>>,
) -> actix_web::Result<HttpResponse> {
    let providers = provider_repo
        .find_all()
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "data": providers,
    })))
}

#[get("/v1/admin/providers/{id}")]
pub async fn get_provider(
    provider_repo: web::Data<Arc<dyn ProviderRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let provider = provider_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("provider not found"))?;

    Ok(HttpResponse::Ok().json(provider))
}

#[put("/v1/admin/providers/{id}")]
pub async fn update_provider(
    provider_repo: web::Data<Arc<dyn ProviderRepository>>,
    path: web::Path<String>,
    body: web::Json<UpdateProviderRequest>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let existing = provider_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("provider not found"))?;

    let updated = ProviderDefinition {
        id: existing.id,
        name: body.name.clone().unwrap_or(existing.name),
        provider_type: body.provider_type.clone().unwrap_or(existing.provider_type),
        api_key: body.api_key.clone().or(existing.api_key),
        base_url: body.base_url.clone().or(existing.base_url),
        config: body.config.clone().or(existing.config),
        enabled: body.enabled.unwrap_or(existing.enabled),
        created_at: existing.created_at,
        updated_at: chrono::Utc::now().naive_utc(),
        deleted_at: existing.deleted_at,
    };

    let result = provider_repo
        .update(&updated)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(result))
}

#[delete("/v1/admin/providers/{id}")]
pub async fn delete_provider(
    provider_repo: web::Data<Arc<dyn ProviderRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    provider_repo
        .delete(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::NoContent().finish())
}
