use actix_web::{delete, get, post, put, web, HttpResponse};
use std::sync::Arc;

use batata_ai_core::domain::{DeploymentStatus, ModelDeployment};
use batata_ai_core::repository::{ModelBenchmarkRepository, ModelDeploymentRepository};

#[derive(serde::Deserialize)]
pub struct CreateDeploymentRequest {
    pub model_id: String,
    pub version_tag: String,
    pub provider_model_id: String,
    pub provider_id: String,
    pub traffic_weight: Option<i32>,
    pub tenant_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(serde::Deserialize)]
pub struct UpdateWeightRequest {
    pub traffic_weight: i32,
}

/// Create a new model deployment (staged by default).
#[post("/v1/admin/deployments")]
pub async fn create_deployment(
    repo: web::Data<Arc<dyn ModelDeploymentRepository>>,
    body: web::Json<CreateDeploymentRequest>,
) -> actix_web::Result<HttpResponse> {
    let now = chrono::Utc::now().naive_utc();
    let deployment = ModelDeployment {
        id: uuid::Uuid::new_v4().to_string(),
        model_id: body.model_id.clone(),
        version_tag: body.version_tag.clone(),
        provider_model_id: body.provider_model_id.clone(),
        provider_id: body.provider_id.clone(),
        status: DeploymentStatus::Staged,
        traffic_weight: body.traffic_weight.unwrap_or(0),
        tenant_id: body.tenant_id.clone(),
        metadata: body.metadata.clone(),
        created_at: now,
        updated_at: now,
    };

    let result = repo
        .create(&deployment)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(result))
}

/// List deployments for a model.
#[get("/v1/admin/models/{model_id}/deployments")]
pub async fn list_deployments(
    repo: web::Data<Arc<dyn ModelDeploymentRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let model_id = path.into_inner();
    let deployments = repo
        .find_by_model(&model_id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({ "data": deployments })))
}

/// Get a single deployment.
#[get("/v1/admin/deployments/{id}")]
pub async fn get_deployment(
    repo: web::Data<Arc<dyn ModelDeploymentRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let deployment = repo
        .find_by_id(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .ok_or_else(|| actix_web::error::ErrorNotFound("deployment not found"))?;

    Ok(HttpResponse::Ok().json(deployment))
}

/// Update traffic weight for canary / A-B testing.
#[put("/v1/admin/deployments/{id}/weight")]
pub async fn update_weight(
    repo: web::Data<Arc<dyn ModelDeploymentRepository>>,
    path: web::Path<String>,
    body: web::Json<UpdateWeightRequest>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    repo.set_traffic_weight(&id, body.traffic_weight)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "id": id,
        "traffic_weight": body.traffic_weight,
    })))
}

/// Promote a deployment to active (retires previous active).
#[post("/v1/admin/deployments/{id}/promote")]
pub async fn promote_deployment(
    repo: web::Data<Arc<dyn ModelDeploymentRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let deployment = repo
        .promote(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(deployment))
}

/// Delete a deployment.
#[delete("/v1/admin/deployments/{id}")]
pub async fn delete_deployment(
    repo: web::Data<Arc<dyn ModelDeploymentRepository>>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    repo.delete(&id)
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::NoContent().finish())
}

/// Get benchmarks for a deployment.
#[get("/v1/admin/deployments/{id}/benchmarks")]
pub async fn get_benchmarks(
    repo: web::Data<Arc<dyn ModelBenchmarkRepository>>,
    path: web::Path<String>,
    query: web::Query<LimitQuery>,
) -> actix_web::Result<HttpResponse> {
    let id = path.into_inner();
    let benchmarks = repo
        .find_by_deployment(&id, query.limit.unwrap_or(30))
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({ "data": benchmarks })))
}

#[derive(serde::Deserialize)]
pub struct LimitQuery {
    pub limit: Option<u64>,
}

#[derive(serde::Deserialize)]
pub struct CompareQuery {
    pub deployment_a: String,
    pub deployment_b: String,
    pub limit: Option<u64>,
}

/// Compare two deployments (A/B test results).
#[get("/v1/admin/deployments/compare")]
pub async fn compare_deployments(
    repo: web::Data<Arc<dyn ModelBenchmarkRepository>>,
    query: web::Query<CompareQuery>,
) -> actix_web::Result<HttpResponse> {
    let (a, b) = repo
        .compare_deployments(
            &query.deployment_a,
            &query.deployment_b,
            query.limit.unwrap_or(30),
        )
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "deployment_a": { "id": query.deployment_a, "benchmarks": a },
        "deployment_b": { "id": query.deployment_b, "benchmarks": b },
    })))
}
