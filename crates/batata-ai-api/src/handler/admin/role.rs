use std::sync::Arc;

use actix_web::{delete, get, post, web, HttpResponse};
use casbin::{CoreApi, MgmtApi, RbacApi};
use tokio::sync::RwLock;

use batata_ai_core::repository::UserRepository;

#[derive(serde::Deserialize)]
pub struct AssignRoleRequest {
    pub role: String,
}

/// Assign a role to a user within their tenant.
#[post("/v1/admin/users/{id}/roles")]
pub async fn assign_role(
    enforcer: web::Data<Arc<RwLock<casbin::Enforcer>>>,
    user_repo: web::Data<Arc<dyn UserRepository>>,
    path: web::Path<String>,
    body: web::Json<AssignRoleRequest>,
) -> actix_web::Result<HttpResponse> {
    let user_id = path.into_inner();
    let user = user_repo
        .find_by_id(&user_id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("user not found"))?;

    let mut enforcer = enforcer.write().await;
    enforcer
        .add_grouping_policy(vec![
            user.id.clone(),
            body.role.clone(),
            user.tenant_id.clone(),
        ])
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "user_id": user.id,
        "role": body.role,
        "tenant_id": user.tenant_id,
    })))
}

/// Get all roles for a user within their tenant.
#[get("/v1/admin/users/{id}/roles")]
pub async fn get_roles(
    enforcer: web::Data<Arc<RwLock<casbin::Enforcer>>>,
    user_repo: web::Data<Arc<dyn UserRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let user_id = path.into_inner();
    let user = user_repo
        .find_by_id(&user_id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("user not found"))?;

    let enforcer = enforcer.read().await;
    let roles = enforcer.get_roles_for_user(&user.id, Some(&user.tenant_id));

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "user_id": user.id,
        "tenant_id": user.tenant_id,
        "roles": roles,
    })))
}

/// Remove a role from a user within their tenant.
#[delete("/v1/admin/users/{id}/roles")]
pub async fn remove_role(
    enforcer: web::Data<Arc<RwLock<casbin::Enforcer>>>,
    user_repo: web::Data<Arc<dyn UserRepository>>,
    path: web::Path<String>,
    body: web::Json<AssignRoleRequest>,
) -> actix_web::Result<HttpResponse> {
    let user_id = path.into_inner();
    let user = user_repo
        .find_by_id(&user_id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("user not found"))?;

    let mut enforcer = enforcer.write().await;
    enforcer
        .remove_grouping_policy(vec![
            user.id.clone(),
            body.role.clone(),
            user.tenant_id.clone(),
        ])
        .await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    Ok(HttpResponse::NoContent().finish())
}

/// Initialize default role permission policies.
/// Call this once when the enforcer is freshly created to seed the base policies.
pub async fn init_default_policies(enforcer: &mut casbin::Enforcer) -> Result<(), casbin::Error> {
    let policies = vec![
        // tenant_admin: full access within tenant
        vec!["tenant_admin", "*", "/v1/admin/users", "read"],
        vec!["tenant_admin", "*", "/v1/admin/users", "write"],
        vec!["tenant_admin", "*", "/v1/admin/users/:id", "read"],
        vec!["tenant_admin", "*", "/v1/admin/users/:id", "write"],
        vec!["tenant_admin", "*", "/v1/admin/users/:id", "delete"],
        vec!["tenant_admin", "*", "/v1/admin/users/:id/password", "write"],
        vec!["tenant_admin", "*", "/v1/admin/users/:id/roles", "read"],
        vec!["tenant_admin", "*", "/v1/admin/users/:id/roles", "write"],
        vec!["tenant_admin", "*", "/v1/admin/users/:id/roles", "delete"],
        vec!["tenant_admin", "*", "/v1/admin/api-keys", "read"],
        vec!["tenant_admin", "*", "/v1/admin/api-keys", "write"],
        vec!["tenant_admin", "*", "/v1/admin/api-keys/:id", "delete"],
        vec!["tenant_admin", "*", "/v1/chat/completions", "write"],
        vec!["tenant_admin", "*", "/v1/models", "read"],
        vec!["tenant_admin", "*", "/v1/conversations", "read"],
        vec!["tenant_admin", "*", "/v1/conversations", "write"],
        vec!["tenant_admin", "*", "/v1/conversations/:id", "read"],
        vec!["tenant_admin", "*", "/v1/conversations/:id", "delete"],
        vec!["tenant_admin", "*", "/v1/conversations/:id/messages", "read"],
        vec!["tenant_admin", "*", "/v1/usage", "read"],
        // editor: use chat, manage conversations
        vec!["editor", "*", "/v1/chat/completions", "write"],
        vec!["editor", "*", "/v1/models", "read"],
        vec!["editor", "*", "/v1/conversations", "read"],
        vec!["editor", "*", "/v1/conversations", "write"],
        vec!["editor", "*", "/v1/conversations/:id", "read"],
        vec!["editor", "*", "/v1/conversations/:id", "delete"],
        vec!["editor", "*", "/v1/conversations/:id/messages", "read"],
        vec!["editor", "*", "/v1/usage", "read"],
        // viewer: read-only
        vec!["viewer", "*", "/v1/models", "read"],
        vec!["viewer", "*", "/v1/conversations", "read"],
        vec!["viewer", "*", "/v1/conversations/:id", "read"],
        vec!["viewer", "*", "/v1/conversations/:id/messages", "read"],
        vec!["viewer", "*", "/v1/usage", "read"],
    ];

    let str_policies: Vec<Vec<String>> = policies
        .into_iter()
        .map(|p| p.into_iter().map(String::from).collect())
        .collect();

    enforcer.add_policies(str_policies).await?;

    Ok(())
}
