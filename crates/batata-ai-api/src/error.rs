use actix_web::{HttpResponse, ResponseError};
use serde::Serialize;
use std::fmt;

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

#[derive(Debug)]
pub struct AppError {
    pub status: actix_web::http::StatusCode,
    pub message: String,
    pub error_type: String,
    pub code: Option<String>,
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl ResponseError for AppError {
    fn status_code(&self) -> actix_web::http::StatusCode {
        self.status
    }

    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status).json(ApiError {
            error: ApiErrorBody {
                message: self.message.clone(),
                r#type: self.error_type.clone(),
                code: self.code.clone(),
            },
        })
    }
}

impl AppError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: actix_web::http::StatusCode::BAD_REQUEST,
            message: msg.into(),
            error_type: "invalid_request_error".to_string(),
            code: None,
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: actix_web::http::StatusCode::NOT_FOUND,
            message: msg.into(),
            error_type: "not_found".to_string(),
            code: None,
        }
    }

    pub fn unauthorized(msg: impl Into<String>) -> Self {
        Self {
            status: actix_web::http::StatusCode::UNAUTHORIZED,
            message: msg.into(),
            error_type: "authentication_error".to_string(),
            code: None,
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
            error_type: "internal_error".to_string(),
            code: None,
        }
    }

    pub fn rate_limited() -> Self {
        Self {
            status: actix_web::http::StatusCode::TOO_MANY_REQUESTS,
            message: "rate limit exceeded".to_string(),
            error_type: "rate_limit_error".to_string(),
            code: Some("rate_limit_exceeded".to_string()),
        }
    }
}

impl From<batata_ai_core::error::BatataError> for AppError {
    fn from(err: batata_ai_core::error::BatataError) -> Self {
        match &err {
            batata_ai_core::error::BatataError::NotFound(msg) => Self::not_found(msg.clone()),
            batata_ai_core::error::BatataError::ModelNotFound(msg) => Self::not_found(msg.clone()),
            _ => Self::internal(err.to_string()),
        }
    }
}
