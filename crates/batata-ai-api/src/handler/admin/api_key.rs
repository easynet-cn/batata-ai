use actix_web::{delete, get, post, web, HttpResponse};
use std::sync::Arc;

use batata_ai_core::crypto::generate_app_key_pair;
use batata_ai_core::domain::ApiKey;
use batata_ai_core::repository::ApiKeyRepository;

use crate::middleware::auth::{generate_api_key, hash_api_key};

#[derive(serde::Deserialize)]
pub struct CreateApiKeyRequest {
    pub tenant_id: String,
    pub name: String,
    pub scopes: Option<serde_json::Value>,
    pub rate_limit: Option<i32>,
    pub expires_at: Option<chrono::NaiveDateTime>,
}

#[derive(serde::Serialize)]
pub struct CreateApiKeyResponse {
    pub id: String,
    pub tenant_id: String,
    pub name: String,
    /// Bearer token (only returned on creation, store it safely).
    pub key: String,
    pub key_prefix: String,
    /// App key for dual-key auth (public identifier).
    pub app_key: String,
    /// App secret for dual-key auth (only returned on creation, store it safely).
    pub app_secret: String,
    pub scopes: serde_json::Value,
    pub rate_limit: Option<i32>,
    pub expires_at: Option<chrono::NaiveDateTime>,
    pub created_at: chrono::NaiveDateTime,
}

#[derive(serde::Deserialize)]
pub struct ListApiKeysQuery {
    pub tenant_id: String,
}

#[post("/v1/admin/api-keys")]
pub async fn create_api_key(
    api_key_repo: web::Data<Arc<dyn ApiKeyRepository>>,
    body: web::Json<CreateApiKeyRequest>,
) -> actix_web::Result<HttpResponse> {
    // Generate Bearer token
    let (plain_key, prefix) = generate_api_key();
    let key_hash = hash_api_key(&plain_key);

    // Generate app_key + app_secret pair
    let (app_key, app_secret) = generate_app_key_pair();
    let app_secret_hash = hash_api_key(&app_secret);

    let now = chrono::Utc::now().naive_utc();
    let scopes = body.scopes.clone().unwrap_or(serde_json::json!(["*"]));

    let api_key = ApiKey {
        id: uuid::Uuid::new_v4().to_string(),
        tenant_id: body.tenant_id.clone(),
        name: body.name.clone(),
        key_hash,
        key_prefix: prefix.clone(),
        app_key: Some(app_key.clone()),
        app_secret_hash: Some(app_secret_hash),
        scopes: scopes.clone(),
        rate_limit: body.rate_limit,
        expires_at: body.expires_at,
        enabled: true,
        last_used_at: None,
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    let result = api_key_repo
        .create(&api_key)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    // Return credentials only on creation — client must store them safely.
    let response = CreateApiKeyResponse {
        id: result.id,
        tenant_id: result.tenant_id,
        name: result.name,
        key: plain_key,
        key_prefix: prefix,
        app_key,
        app_secret,
        scopes,
        rate_limit: result.rate_limit,
        expires_at: result.expires_at,
        created_at: result.created_at,
    };

    Ok(HttpResponse::Created().json(response))
}

#[get("/v1/admin/api-keys")]
pub async fn list_api_keys(
    api_key_repo: web::Data<Arc<dyn ApiKeyRepository>>,
    query: web::Query<ListApiKeysQuery>,
) -> actix_web::Result<HttpResponse> {
    let keys = api_key_repo
        .find_by_tenant(&query.tenant_id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "data": keys,
    })))
}

#[delete("/v1/admin/api-keys/{id}")]
pub async fn delete_api_key(
    api_key_repo: web::Data<Arc<dyn ApiKeyRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    api_key_repo
        .delete(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::NoContent().finish())
}
