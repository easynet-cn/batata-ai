use actix_web::{delete, get, post, put, web, HttpResponse};
use std::sync::Arc;

use batata_ai_core::domain::Tenant;
use batata_ai_core::repository::TenantRepository;

#[derive(serde::Deserialize)]
pub struct CreateTenantRequest {
    pub name: String,
    pub slug: String,
    pub config: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
pub struct UpdateTenantRequest {
    pub name: Option<String>,
    pub slug: Option<String>,
    pub config: Option<serde_json::Value>,
    pub enabled: Option<bool>,
}

#[post("/v1/admin/tenants")]
pub async fn create_tenant(
    tenant_repo: web::Data<Arc<dyn TenantRepository>>,
    body: web::Json<CreateTenantRequest>,
) -> actix_web::Result<HttpResponse> {
    let now = chrono::Utc::now().naive_utc();
    let tenant = Tenant {
        id: uuid::Uuid::new_v4().to_string(),
        name: body.name.clone(),
        slug: body.slug.clone(),
        config: body.config.clone(),
        enabled: true,
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    let result = tenant_repo
        .create(&tenant)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(result))
}

#[get("/v1/admin/tenants")]
pub async fn list_tenants(
    tenant_repo: web::Data<Arc<dyn TenantRepository>>,
) -> actix_web::Result<HttpResponse> {
    let tenants = tenant_repo
        .find_all()
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "data": tenants,
    })))
}

#[get("/v1/admin/tenants/{id}")]
pub async fn get_tenant(
    tenant_repo: web::Data<Arc<dyn TenantRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let tenant = tenant_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("tenant not found"))?;

    Ok(HttpResponse::Ok().json(tenant))
}

#[put("/v1/admin/tenants/{id}")]
pub async fn update_tenant(
    tenant_repo: web::Data<Arc<dyn TenantRepository>>,
    path: web::Path<String>,
    body: web::Json<UpdateTenantRequest>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let existing = tenant_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("tenant not found"))?;

    let updated = Tenant {
        id: existing.id,
        name: body.name.clone().unwrap_or(existing.name),
        slug: body.slug.clone().unwrap_or(existing.slug),
        config: body.config.clone().or(existing.config),
        enabled: body.enabled.unwrap_or(existing.enabled),
        created_at: existing.created_at,
        updated_at: chrono::Utc::now().naive_utc(),
        deleted_at: existing.deleted_at,
    };

    let result = tenant_repo
        .update(&updated)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(result))
}

#[delete("/v1/admin/tenants/{id}")]
pub async fn delete_tenant(
    tenant_repo: web::Data<Arc<dyn TenantRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    tenant_repo
        .delete(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::NoContent().finish())
}
