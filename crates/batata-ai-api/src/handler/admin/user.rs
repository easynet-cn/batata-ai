use actix_web::{delete, get, post, put, web, HttpResponse};
use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::SaltString;
use argon2::{Argon2, PasswordHasher};
use std::sync::Arc;

use batata_ai_core::domain::User;
use batata_ai_core::repository::UserRepository;

#[derive(serde::Deserialize)]
pub struct CreateUserRequest {
    pub tenant_id: String,
    pub username: String,
    pub password: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct UpdateUserRequest {
    pub display_name: Option<String>,
    pub email: Option<String>,
    pub enabled: Option<bool>,
}

#[derive(serde::Deserialize)]
pub struct ListUsersQuery {
    pub tenant_id: String,
}

#[derive(serde::Deserialize)]
pub struct ChangePasswordRequest {
    pub password: String,
}

fn hash_password(password: &str) -> actix_web::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| actix_web::error::ErrorInternalServerError(format!("hash error: {e}")))?;
    Ok(hash.to_string())
}

#[post("/v1/admin/users")]
pub async fn create_user(
    user_repo: web::Data<Arc<dyn UserRepository>>,
    body: web::Json<CreateUserRequest>,
) -> actix_web::Result<HttpResponse> {
    let now = chrono::Utc::now().naive_utc();
    let password_hash = hash_password(&body.password)?;

    let user = User {
        id: uuid::Uuid::new_v4().to_string(),
        tenant_id: body.tenant_id.clone(),
        username: body.username.clone(),
        password_hash,
        display_name: body.display_name.clone(),
        email: body.email.clone(),
        enabled: true,
        last_login_at: None,
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    let result = user_repo
        .create(&user)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(result))
}

#[get("/v1/admin/users")]
pub async fn list_users(
    user_repo: web::Data<Arc<dyn UserRepository>>,
    query: web::Query<ListUsersQuery>,
) -> actix_web::Result<HttpResponse> {
    let users = user_repo
        .find_by_tenant(&query.tenant_id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "data": users,
    })))
}

#[get("/v1/admin/users/{id}")]
pub async fn get_user(
    user_repo: web::Data<Arc<dyn UserRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let user = user_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("user not found"))?;

    Ok(HttpResponse::Ok().json(user))
}

#[put("/v1/admin/users/{id}")]
pub async fn update_user(
    user_repo: web::Data<Arc<dyn UserRepository>>,
    path: web::Path<String>,
    body: web::Json<UpdateUserRequest>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let existing = user_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("user not found"))?;

    let updated = User {
        id: existing.id,
        tenant_id: existing.tenant_id,
        username: existing.username,
        password_hash: existing.password_hash,
        display_name: body.display_name.clone().or(existing.display_name),
        email: body.email.clone().or(existing.email),
        enabled: body.enabled.unwrap_or(existing.enabled),
        last_login_at: existing.last_login_at,
        created_at: existing.created_at,
        updated_at: chrono::Utc::now().naive_utc(),
        deleted_at: existing.deleted_at,
    };

    let result = user_repo
        .update(&updated)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(result))
}

#[put("/v1/admin/users/{id}/password")]
pub async fn change_password(
    user_repo: web::Data<Arc<dyn UserRepository>>,
    path: web::Path<String>,
    body: web::Json<ChangePasswordRequest>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let existing = user_repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("user not found"))?;

    let password_hash = hash_password(&body.password)?;

    let updated = User {
        password_hash,
        updated_at: chrono::Utc::now().naive_utc(),
        ..existing
    };

    user_repo
        .update(&updated)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::NoContent().finish())
}

#[delete("/v1/admin/users/{id}")]
pub async fn delete_user(
    user_repo: web::Data<Arc<dyn UserRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    user_repo
        .delete(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::NoContent().finish())
}
