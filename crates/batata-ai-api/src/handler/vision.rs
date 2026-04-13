//! Vision endpoints: product recognition, image captioning, object detection.

use std::sync::Arc;

use actix_multipart::Multipart;
use actix_web::{get, post, web, HttpResponse};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use batata_ai_core::multimodal::ImageData;
use batata_ai_local::clip::ClipModel;
use batata_ai_local::product_classifier::{ProductClassifier, ProductEntry, RecognitionResult};

use crate::middleware::auth::AuthContext;

// ── Shared vision state ──────────────────────────────────────────

/// Holds the loaded vision models shared across all handlers.
pub struct VisionState {
    pub clip: Arc<RwLock<ClipModel>>,
    pub blip: Option<Arc<RwLock<batata_ai_local::blip::BlipModel>>>,
    pub classifier: Arc<RwLock<ProductClassifier>>,
}

// ── Request / Response types ─────────────────────────────────────

#[derive(Debug, Serialize)]
struct RecognizeResponse {
    results: Vec<RecognitionResult>,
}

#[derive(Debug, Serialize)]
struct CaptionResponse {
    caption: String,
}

#[derive(Debug, Deserialize)]
pub struct RecognizeQuery {
    /// Number of top candidates to return (default: 5)
    pub top_k: Option<usize>,
    /// Minimum confidence threshold (default: 0.0)
    pub min_confidence: Option<f32>,
}

#[derive(Debug, Serialize)]
struct ProductListResponse {
    products: Vec<ProductEntry>,
    count: usize,
}

// ── Helpers ──────────────────────────────────────────────────────

/// Extract image bytes from a multipart payload.
/// Expects a field named "image".
async fn extract_image_from_multipart(
    payload: &mut Multipart,
) -> actix_web::Result<(Vec<u8>, Option<String>)> {
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;

    while let Some(item) = payload.next().await {
        let mut field = item.map_err(actix_web::error::ErrorBadRequest)?;
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "image" | "file" => {
                filename = field
                    .content_disposition()
                    .and_then(|cd| cd.get_filename())
                    .map(|s| s.to_string());
                let mut buf = Vec::new();
                while let Some(chunk) = field.next().await {
                    let chunk = chunk.map_err(actix_web::error::ErrorBadRequest)?;
                    buf.extend_from_slice(&chunk);
                }
                image_bytes = Some(buf);
            }
            _ => {
                while let Some(chunk) = field.next().await {
                    let _ = chunk.map_err(actix_web::error::ErrorBadRequest)?;
                }
            }
        }
    }

    let bytes = image_bytes.ok_or_else(|| {
        actix_web::error::ErrorBadRequest("missing `image` field in multipart body")
    })?;

    Ok((bytes, filename))
}

/// Decode raw image bytes (JPEG, PNG, etc.) into RGB ImageData.
fn decode_image(bytes: &[u8]) -> std::result::Result<ImageData, String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("invalid image: {e}"))?;
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(ImageData {
        bytes: rgb.into_raw(),
        width,
        height,
    })
}

// ── Endpoints ────────────────────────────────────────────────────

/// `POST /v1/vision/recognize` — Recognize products from an uploaded image.
///
/// Multipart fields:
/// - `image`: the image file (JPEG, PNG, etc.)
///
/// Query parameters:
/// - `top_k`: number of top candidates (default: 5)
/// - `min_confidence`: minimum confidence threshold (default: 0.0)
#[post("/v1/vision/recognize")]
pub async fn recognize_product(
    vision: web::Data<VisionState>,
    query: web::Query<RecognizeQuery>,
    mut payload: Multipart,
    _auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let (bytes, _filename) = extract_image_from_multipart(&mut payload).await?;

    let image = tokio::task::spawn_blocking(move || decode_image(&bytes))
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .map_err(actix_web::error::ErrorBadRequest)?;

    let top_k = query.top_k.unwrap_or(5);
    let min_confidence = query.min_confidence.unwrap_or(0.0);

    let clip = vision.clip.read().await;
    let classifier = vision.classifier.read().await;

    let mut results = classifier
        .recognize(&clip, &image, top_k)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    // Filter by minimum confidence
    if min_confidence > 0.0 {
        results.retain(|r| r.confidence >= min_confidence);
    }

    Ok(HttpResponse::Ok().json(RecognizeResponse { results }))
}

/// `POST /v1/vision/caption` — Generate a text caption for an image.
///
/// Multipart fields:
/// - `image`: the image file
#[post("/v1/vision/caption")]
pub async fn caption_image(
    vision: web::Data<VisionState>,
    mut payload: Multipart,
    _auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let blip = vision.blip.as_ref().ok_or_else(|| {
        actix_web::error::ErrorServiceUnavailable("BLIP captioning model not loaded")
    })?;

    let (bytes, _filename) = extract_image_from_multipart(&mut payload).await?;

    let image = tokio::task::spawn_blocking(move || decode_image(&bytes))
        .await
        .map_err(actix_web::error::ErrorInternalServerError)?
        .map_err(actix_web::error::ErrorBadRequest)?;

    let mut blip_model = blip.write().await;
    let caption = blip_model
        .caption(image)
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Ok().json(CaptionResponse { caption }))
}

/// `POST /v1/vision/products` — Add a product to the classifier catalog.
///
/// JSON body: `ProductEntry`
#[post("/v1/vision/products")]
pub async fn add_product(
    vision: web::Data<VisionState>,
    body: web::Json<ProductEntry>,
    _auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let clip = vision.clip.read().await;
    let mut classifier = vision.classifier.write().await;

    classifier
        .add_product(&clip, body.into_inner())
        .map_err(actix_web::error::ErrorInternalServerError)?;

    Ok(HttpResponse::Created().json(serde_json::json!({
        "status": "ok",
        "total_products": classifier.product_count(),
    })))
}

/// `GET /v1/vision/products` — List all products in the classifier catalog.
#[get("/v1/vision/products")]
pub async fn list_products(
    vision: web::Data<VisionState>,
    _auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let classifier = vision.classifier.read().await;
    let products: Vec<ProductEntry> = classifier
        .list_products()
        .into_iter()
        .cloned()
        .collect();
    let count = products.len();

    Ok(HttpResponse::Ok().json(ProductListResponse { products, count }))
}

/// `POST /v1/vision/products/batch` — Batch add products to the catalog.
///
/// JSON body: array of `ProductEntry`
#[post("/v1/vision/products/batch")]
pub async fn batch_add_products(
    vision: web::Data<VisionState>,
    body: web::Json<Vec<ProductEntry>>,
    _auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let clip = vision.clip.read().await;
    let mut classifier = vision.classifier.write().await;

    let mut added = 0;
    for product in body.into_inner() {
        classifier
            .add_product(&clip, product)
            .map_err(actix_web::error::ErrorInternalServerError)?;
        added += 1;
    }

    Ok(HttpResponse::Created().json(serde_json::json!({
        "status": "ok",
        "added": added,
        "total_products": classifier.product_count(),
    })))
}
