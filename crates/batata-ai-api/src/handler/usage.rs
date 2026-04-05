use actix_web::{get, web, HttpResponse};
use std::sync::Arc;

use batata_ai_core::repository::TenantUsageRepository;

use crate::middleware::auth::AuthContext;

#[derive(serde::Deserialize)]
pub struct UsageQuery {
    pub period: Option<String>,
}

/// Get usage statistics for the current tenant.
#[get("/v1/usage")]
pub async fn get_usage(
    usage_repo: web::Data<Arc<dyn TenantUsageRepository>>,
    auth: Option<web::ReqData<AuthContext>>,
    query: web::Query<UsageQuery>,
) -> actix_web::Result<HttpResponse> {
    let tenant_id = auth
        .as_ref()
        .map(|a| a.tenant_id.clone())
        .unwrap_or_default();

    if let Some(period) = &query.period {
        // Specific period
        let usage = usage_repo
            .find_by_tenant_period(&tenant_id, period)
            .await
            .map_err(actix_web::error::ErrorInternalServerError)?;

        Ok(HttpResponse::Ok().json(serde_json::json!({
            "data": usage,
        })))
    } else {
        // Recent periods
        let usages = usage_repo
            .find_by_tenant(&tenant_id, 12)
            .await
            .map_err(actix_web::error::ErrorInternalServerError)?;

        Ok(HttpResponse::Ok().json(serde_json::json!({
            "data": usages,
        })))
    }
}
