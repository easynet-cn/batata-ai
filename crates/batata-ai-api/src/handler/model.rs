use actix_web::{get, web, HttpResponse};
use std::sync::Arc;

use batata_ai_core::repository::ModelRepository;

#[get("/v1/models")]
pub async fn list_models(
    model_repo: web::Data<Arc<dyn ModelRepository>>,
) -> actix_web::Result<HttpResponse> {
    let models = model_repo
        .find_all()
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "object": "list",
        "data": models,
    })))
}
