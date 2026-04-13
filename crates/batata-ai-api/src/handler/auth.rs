use actix_web::{post, web, HttpResponse};
use argon2::{Argon2, PasswordHash, PasswordVerifier};
use std::sync::Arc;

use batata_ai_core::repository::UserRepository;

use crate::middleware::jwt::JwtConfig;

#[derive(serde::Deserialize)]
pub struct LoginRequest {
    pub tenant_id: String,
    pub username: String,
    pub password: String,
}

#[derive(serde::Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: &'static str,
    pub expires_in: u64,
}

#[derive(serde::Deserialize)]
pub struct RefreshRequest {
    pub refresh_token: String,
}

/// Login with username/password, returns JWT access + refresh tokens.
#[post("/v1/auth/login")]
pub async fn login(
    user_repo: web::Data<Arc<dyn UserRepository>>,
    jwt_config: web::Data<JwtConfig>,
    body: web::Json<LoginRequest>,
) -> actix_web::Result<HttpResponse> {
    let user = user_repo
        .find_by_username(&body.tenant_id, &body.username)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("invalid credentials"))?;

    if !user.enabled {
        return Err(actix_web::error::ErrorUnauthorized("user is disabled"));
    }

    // Verify password using argon2
    let parsed_hash = PasswordHash::new(&user.password_hash)
        .map_err(|_| actix_web::error::ErrorInternalServerError("invalid password hash"))?;

    Argon2::default()
        .verify_password(body.password.as_bytes(), &parsed_hash)
        .map_err(|_| actix_web::error::ErrorUnauthorized("invalid credentials"))?;

    let scopes = serde_json::json!(["*"]);

    let access_token = jwt_config
        .issue_access_token(&user.id, &user.tenant_id, &user.username, scopes)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let new_refresh_token = jwt_config
        .issue_refresh_token(&user.id, &user.tenant_id)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(TokenResponse {
        access_token,
        refresh_token: new_refresh_token,
        token_type: "Bearer",
        expires_in: jwt_config.access_token_ttl,
    }))
}

/// Refresh access token using a refresh token.
#[post("/v1/auth/refresh")]
pub async fn refresh(
    jwt_config: web::Data<JwtConfig>,
    user_repo: web::Data<Arc<dyn UserRepository>>,
    body: web::Json<RefreshRequest>,
) -> actix_web::Result<HttpResponse> {
    let token_data = jwt_config
        .validate_token(&body.refresh_token)
        .map_err(|_| actix_web::error::ErrorUnauthorized("invalid or expired refresh token"))?;

    let claims = token_data.claims;

    // Re-fetch user to get current state
    let user = user_repo
        .find_by_id(&claims.sub)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("user not found"))?;

    if !user.enabled {
        return Err(actix_web::error::ErrorUnauthorized("user is disabled"));
    }

    let scopes = serde_json::json!(["*"]);

    let access_token = jwt_config
        .issue_access_token(&user.id, &user.tenant_id, &user.username, scopes)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    let new_refresh_token = jwt_config
        .issue_refresh_token(&user.id, &user.tenant_id)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(TokenResponse {
        access_token,
        refresh_token: new_refresh_token,
        token_type: "Bearer",
        expires_in: jwt_config.access_token_ttl,
    }))
}
