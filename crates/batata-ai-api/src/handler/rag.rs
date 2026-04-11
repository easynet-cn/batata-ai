//! RAG endpoints: KB + document management, ingest, search.
//!
//! KB + document metadata live in the database (see the
//! `knowledge_bases` and `kb_documents` tables). Chunks live in
//! `kb_chunks` and are accessed through the pipeline's `VectorStore`
//! trait, which may be in-memory or SeaORM backed depending on config.

use std::path::Path;

use actix_multipart::Multipart;
use actix_web::{delete, get, post, web, HttpResponse};
use chrono::Utc;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use batata_ai_core::error::BatataError;
use batata_ai_core::rag::{KbDocument, KnowledgeBase, RagHit};

use crate::middleware::auth::AuthContext;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn rag_disabled() -> actix_web::Error {
    actix_web::error::ErrorServiceUnavailable("RAG pipeline not configured on this server")
}

fn rag_meta_disabled() -> actix_web::Error {
    actix_web::error::ErrorServiceUnavailable("RAG metadata store not configured on this server")
}

fn map_not_found(what: impl Into<String>) -> actix_web::Error {
    actix_web::error::ErrorNotFound(what.into())
}

fn map_core_err(e: BatataError) -> actix_web::Error {
    match &e {
        BatataError::NotFound(_) => actix_web::error::ErrorNotFound(e.to_string()),
        _ => actix_web::error::ErrorInternalServerError(e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Knowledge base CRUD
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateKbRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Embedder name recorded as metadata. `None` means "same as server default".
    #[serde(default)]
    pub embedder: Option<String>,
    /// Vector dimension — `None` means "read from live pipeline".
    #[serde(default)]
    pub dim: Option<i32>,
    #[serde(default)]
    pub chunk_window: Option<i32>,
    #[serde(default)]
    pub chunk_overlap: Option<i32>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct KbResponse {
    pub id: String,
    pub tenant_id: Option<String>,
    pub name: String,
    pub description: Option<String>,
    pub embedder: String,
    pub dim: i32,
    pub chunk_window: i32,
    pub chunk_overlap: i32,
    pub metadata: Option<serde_json::Value>,
}

impl From<KnowledgeBase> for KbResponse {
    fn from(k: KnowledgeBase) -> Self {
        Self {
            id: k.id,
            tenant_id: k.tenant_id,
            name: k.name,
            description: k.description,
            embedder: k.embedder,
            dim: k.dim,
            chunk_window: k.chunk_window,
            chunk_overlap: k.chunk_overlap,
            metadata: k.metadata,
        }
    }
}

#[post("/v1/kb")]
pub async fn create_kb(
    state: web::Data<crate::state::AppState>,
    body: web::Json<CreateKbRequest>,
    auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let repo = state.kb_repo.clone().ok_or_else(rag_meta_disabled)?;
    let pipeline = state.rag_pipeline.clone();

    let tenant_id = auth.as_ref().map(|a| a.tenant_id.clone()).filter(|s| !s.is_empty());

    // Pull defaults from the live pipeline if available; fall back to server-defaults.
    let (default_dim, default_embedder) = match &pipeline {
        Some(p) => (p.embedder.dimensions() as i32, "clip".to_string()),
        None => (512, "clip".to_string()),
    };

    let now = Utc::now().naive_utc();
    let entity = KnowledgeBase {
        id: uuid::Uuid::new_v4().to_string(),
        tenant_id,
        name: body.name.clone(),
        description: body.description.clone(),
        embedder: body.embedder.clone().unwrap_or(default_embedder),
        dim: body.dim.unwrap_or(default_dim),
        chunk_window: body.chunk_window.unwrap_or(512),
        chunk_overlap: body.chunk_overlap.unwrap_or(64),
        metadata: body.metadata.clone(),
        created_at: now,
        updated_at: now,
        deleted_at: None,
    };

    let created = repo.create(&entity).await.map_err(map_core_err)?;
    Ok(HttpResponse::Created().json(KbResponse::from(created)))
}

#[get("/v1/kb")]
pub async fn list_kbs(
    state: web::Data<crate::state::AppState>,
    auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let repo = state.kb_repo.clone().ok_or_else(rag_meta_disabled)?;
    let tenant_id = auth
        .as_ref()
        .map(|a| a.tenant_id.clone())
        .filter(|s| !s.is_empty());
    let rows = repo
        .find_by_tenant(tenant_id.as_deref())
        .await
        .map_err(map_core_err)?;
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "items": rows.into_iter().map(KbResponse::from).collect::<Vec<_>>()
    })))
}

#[get("/v1/kb/{id}")]
pub async fn get_kb(
    state: web::Data<crate::state::AppState>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let repo = state.kb_repo.clone().ok_or_else(rag_meta_disabled)?;
    let id = path.into_inner();
    let kb = repo
        .find_by_id(&id)
        .await
        .map_err(map_core_err)?
        .ok_or_else(|| map_not_found(format!("kb {id}")))?;
    Ok(HttpResponse::Ok().json(KbResponse::from(kb)))
}

#[delete("/v1/kb/{id}")]
pub async fn delete_kb(
    state: web::Data<crate::state::AppState>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let kb_repo = state.kb_repo.clone().ok_or_else(rag_meta_disabled)?;
    let doc_repo = state.kb_document_repo.clone();
    let pipeline = state.rag_pipeline.clone();
    let id = path.into_inner();

    // Order: chunks → documents → KB row. If the request dies mid-way, a
    // retry is still safe because each step is idempotent and the KB row
    // is the "is this KB still alive" breadcrumb.
    if let Some(p) = &pipeline {
        if let Err(e) = p.store.delete_by_kb(&id).await {
            tracing::warn!(kb = %id, error = %e, "failed to drop chunks during KB delete");
        }
    }
    if let Some(dr) = &doc_repo {
        if let Err(e) = dr.soft_delete_by_kb(&id).await {
            tracing::warn!(kb = %id, error = %e, "failed to soft-delete documents during KB delete");
        }
    }
    let ok = kb_repo.soft_delete(&id).await.map_err(map_core_err)?;
    if ok {
        Ok(HttpResponse::NoContent().finish())
    } else {
        Err(map_not_found(format!("kb {id}")))
    }
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct IngestTextRequest {
    pub source_uri: String,
    /// Inline text body. When omitted, the server loads `source_uri`
    /// from the local filesystem (for `file://` or plain paths) and
    /// extracts text using the loader matching the detected format
    /// (PDF / HTML / Markdown / plain text).
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub mime: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct DocumentResponse {
    pub id: String,
    pub kb_id: String,
    pub tenant_id: Option<String>,
    pub source_uri: String,
    pub title: Option<String>,
    pub mime: Option<String>,
    pub status: String,
    pub error: Option<String>,
    pub chunk_count: i32,
    pub metadata: Option<serde_json::Value>,
}

impl From<KbDocument> for DocumentResponse {
    fn from(d: KbDocument) -> Self {
        Self {
            id: d.id,
            kb_id: d.kb_id,
            tenant_id: d.tenant_id,
            source_uri: d.source_uri,
            title: d.title,
            mime: d.mime,
            status: d.status,
            error: d.error,
            chunk_count: d.chunk_count,
            metadata: d.metadata,
        }
    }
}

#[post("/v1/kb/{id}/documents")]
pub async fn ingest_document(
    state: web::Data<crate::state::AppState>,
    path: web::Path<String>,
    body: web::Json<IngestTextRequest>,
    auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let kb_id = path.into_inner();
    let pipeline = state.rag_pipeline.clone().ok_or_else(rag_disabled)?;
    let doc_repo = state.kb_document_repo.clone();
    let kb_repo = state.kb_repo.clone();

    // If meta-store is configured, require the KB to exist before ingest
    // and assert the live pipeline's embedder dimension matches what the
    // KB was registered with — otherwise a silent dim mismatch corrupts
    // cosine search.
    if let Some(repo) = &kb_repo {
        let kb = repo
            .find_by_id(&kb_id)
            .await
            .map_err(map_core_err)?
            .ok_or_else(|| map_not_found(format!("kb {kb_id}")))?;
        let live_dim = pipeline.embedder.dimensions() as i32;
        if live_dim != kb.dim {
            return Err(actix_web::error::ErrorConflict(format!(
                "embedder dimension mismatch: kb '{kb_id}' registered dim = {}, \
                 live pipeline dim = {live_dim}. Recreate the KB or switch embedders.",
                kb.dim
            )));
        }
    }

    let tenant_id = auth.as_ref().map(|a| a.tenant_id.clone()).filter(|s| !s.is_empty());
    let now = Utc::now().naive_utc();

    // Resolve the text payload: either inline (`text`) or loaded from a
    // filesystem URI via the built-in loaders.
    let (resolved_text, effective_mime) = match &body.text {
        Some(t) => (t.clone(), body.mime.clone()),
        None => {
            let (fmt, text) = tokio::task::block_in_place(|| {
                batata_ai_rag::load_uri(&body.source_uri, body.mime.as_deref())
            })
            .map_err(map_core_err)?;
            (text, Some(body.mime.clone().unwrap_or_else(|| fmt.mime().to_string())))
        }
    };

    // Create a pending document row first so failures leave an audit trail.
    let mut document: Option<KbDocument> = None;
    if let Some(repo) = &doc_repo {
        let entity = KbDocument {
            id: uuid::Uuid::new_v4().to_string(),
            kb_id: kb_id.clone(),
            tenant_id: tenant_id.clone(),
            source_uri: body.source_uri.clone(),
            title: body.title.clone(),
            mime: effective_mime.clone(),
            status: "pending".into(),
            error: None,
            chunk_count: 0,
            metadata: body.metadata.clone(),
            created_at: now,
            updated_at: now,
            deleted_at: None,
        };
        document = Some(repo.create(&entity).await.map_err(map_core_err)?);
    }

    // Run the actual ingest.
    let ingest_result = pipeline
        .ingest_text(&kb_id, &body.source_uri, &resolved_text)
        .await;

    match (ingest_result, document, doc_repo) {
        (Ok(pipeline_doc_id), Some(doc), Some(repo)) => {
            // Chunk count = how many chunks the chunker produced.
            let chunk_count = count_chunks(&pipeline, &resolved_text);
            let updated = KbDocument {
                status: "ready".into(),
                chunk_count,
                // Override id with what the pipeline assigned so callers can
                // cross-reference `kb_chunks.doc_id`.
                id: pipeline_doc_id,
                updated_at: Utc::now().naive_utc(),
                ..doc
            };
            let saved = repo.update(&updated).await.map_err(map_core_err)?;
            Ok(HttpResponse::Ok().json(DocumentResponse::from(saved)))
        }
        // No repo configured (or no pre-created row) — ephemeral ingest only.
        (Ok(pipeline_doc_id), None, _) => {
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "kb_id": kb_id,
                "doc_id": pipeline_doc_id,
                "persisted": false,
            })))
        }
        (Ok(pipeline_doc_id), Some(_), None) => {
            // Unreachable in practice: document is only Some when doc_repo
            // is Some. Still handle it explicitly to satisfy exhaustiveness.
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "kb_id": kb_id,
                "doc_id": pipeline_doc_id,
                "persisted": false,
            })))
        }
        (Err(e), Some(doc), Some(repo)) => {
            let updated = KbDocument {
                status: "failed".into(),
                error: Some(e.to_string()),
                updated_at: Utc::now().naive_utc(),
                ..doc
            };
            let _ = repo.update(&updated).await;
            Err(actix_web::error::ErrorInternalServerError(e.to_string()))
        }
        (Err(e), _, _) => Err(actix_web::error::ErrorInternalServerError(e.to_string())),
    }
}

fn count_chunks(pipeline: &batata_ai_rag::IngestPipeline, text: &str) -> i32 {
    pipeline.chunker.chunk(text).len() as i32
}

/// `POST /v1/kb/{id}/documents/upload` — multipart/form-data upload.
///
/// The client sends a `file` field (the actual bytes). Optional fields:
/// - `title`   text
/// - `metadata` JSON string
///
/// The server detects the document format from the part's `content-type`
/// header or the filename extension, runs the matching loader, then
/// hands the extracted text to the ingest pipeline. The raw upload is
/// kept only in memory; persistence to ObjectStore is a separate follow-up.
#[post("/v1/kb/{id}/documents/upload")]
pub async fn upload_document(
    state: web::Data<crate::state::AppState>,
    path: web::Path<String>,
    mut payload: Multipart,
    auth: Option<web::ReqData<AuthContext>>,
) -> actix_web::Result<HttpResponse> {
    let kb_id = path.into_inner();
    let pipeline = state.rag_pipeline.clone().ok_or_else(rag_disabled)?;
    let doc_repo = state.kb_document_repo.clone();
    let kb_repo = state.kb_repo.clone();

    // Validate KB existence + dim if the metadata store is configured.
    if let Some(repo) = &kb_repo {
        let kb = repo
            .find_by_id(&kb_id)
            .await
            .map_err(map_core_err)?
            .ok_or_else(|| map_not_found(format!("kb {kb_id}")))?;
        let live_dim = pipeline.embedder.dimensions() as i32;
        if live_dim != kb.dim {
            return Err(actix_web::error::ErrorConflict(format!(
                "embedder dimension mismatch: kb '{kb_id}' registered dim = {}, \
                 live pipeline dim = {live_dim}.",
                kb.dim
            )));
        }
    }

    // Collect the first file part.
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut content_type: Option<String> = None;
    let mut title: Option<String> = None;
    let mut metadata: Option<serde_json::Value> = None;

    while let Some(item) = payload.next().await {
        let mut field = item.map_err(actix_web::error::ErrorBadRequest)?;
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                content_type = field.content_type().map(|ct| ct.essence_str().to_string());
                filename = field
                    .content_disposition()
                    .and_then(|cd| cd.get_filename())
                    .map(|s| s.to_string());
                let mut buf = Vec::new();
                while let Some(chunk) = field.next().await {
                    let chunk = chunk.map_err(actix_web::error::ErrorBadRequest)?;
                    buf.extend_from_slice(&chunk);
                }
                file_bytes = Some(buf);
            }
            "title" => {
                let mut buf = Vec::new();
                while let Some(chunk) = field.next().await {
                    let chunk = chunk.map_err(actix_web::error::ErrorBadRequest)?;
                    buf.extend_from_slice(&chunk);
                }
                title = Some(String::from_utf8_lossy(&buf).into_owned());
            }
            "metadata" => {
                let mut buf = Vec::new();
                while let Some(chunk) = field.next().await {
                    let chunk = chunk.map_err(actix_web::error::ErrorBadRequest)?;
                    buf.extend_from_slice(&chunk);
                }
                let raw = String::from_utf8_lossy(&buf).into_owned();
                metadata = serde_json::from_str(&raw).ok();
            }
            _ => {
                // Drain unknown fields without blowing up the request.
                while let Some(chunk) = field.next().await {
                    let _ = chunk.map_err(actix_web::error::ErrorBadRequest)?;
                }
            }
        }
    }

    let bytes = file_bytes
        .ok_or_else(|| actix_web::error::ErrorBadRequest("missing `file` field in multipart body"))?;
    let filename = filename.unwrap_or_else(|| "upload.bin".to_string());

    // Format detection: MIME > extension > plain.
    let format = batata_ai_rag::DocumentFormat::detect(
        content_type.as_deref(),
        Path::new(&filename),
    );
    let effective_mime = content_type.clone().unwrap_or_else(|| format.mime().to_string());

    // Run the (possibly expensive, e.g. PDF) parser on a blocking worker.
    let bytes_for_loader = bytes.clone();
    let text = tokio::task::spawn_blocking(move || {
        batata_ai_rag::load_bytes(&bytes_for_loader, format)
    })
    .await
    .map_err(|e| actix_web::error::ErrorInternalServerError(format!("loader join: {e}")))?
    .map_err(map_core_err)?;

    let tenant_id = auth.as_ref().map(|a| a.tenant_id.clone()).filter(|s| !s.is_empty());
    let now = Utc::now().naive_utc();

    // Persist raw bytes to ObjectStore if one is configured. The object
    // key layout is `{prefix}/{tenant-or-platform}/{kb_id}/{uuid}-{filename}`,
    // which keeps per-tenant cleanup easy and avoids collisions.
    let upload_id = uuid::Uuid::new_v4().to_string();
    let tenant_segment = tenant_id.clone().unwrap_or_else(|| "_platform".to_string());
    let object_key = format!(
        "{}/{}/{}/{}-{}",
        state.rag_upload_prefix.trim_matches('/'),
        tenant_segment,
        kb_id,
        upload_id,
        filename.replace('/', "_"),
    );
    let source_uri = if let Some(store) = &state.rag_object_store {
        match store.put(&object_key, &bytes, &effective_mime).await {
            Ok(_) => format!("objstore://{}/{}", store.backend(), object_key),
            Err(e) => {
                tracing::warn!(error = %e, "ObjectStore put failed; falling back to upload:// uri");
                format!("upload://{filename}")
            }
        }
    } else {
        format!("upload://{filename}")
    };

    let mut document: Option<KbDocument> = None;
    if let Some(repo) = &doc_repo {
        let entity = KbDocument {
            id: uuid::Uuid::new_v4().to_string(),
            kb_id: kb_id.clone(),
            tenant_id: tenant_id.clone(),
            source_uri: source_uri.clone(),
            title: title.clone(),
            mime: Some(effective_mime.clone()),
            status: "pending".into(),
            error: None,
            chunk_count: 0,
            metadata: metadata.clone(),
            created_at: now,
            updated_at: now,
            deleted_at: None,
        };
        document = Some(repo.create(&entity).await.map_err(map_core_err)?);
    }

    let ingest_result = pipeline.ingest_text(&kb_id, &source_uri, &text).await;

    match (ingest_result, document, doc_repo) {
        (Ok(pipeline_doc_id), Some(doc), Some(repo)) => {
            let chunk_count = count_chunks(&pipeline, &text);
            let updated = KbDocument {
                status: "ready".into(),
                chunk_count,
                id: pipeline_doc_id,
                updated_at: Utc::now().naive_utc(),
                ..doc
            };
            let saved = repo.update(&updated).await.map_err(map_core_err)?;
            Ok(HttpResponse::Ok().json(DocumentResponse::from(saved)))
        }
        (Ok(pipeline_doc_id), None, _) | (Ok(pipeline_doc_id), Some(_), None) => {
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "kb_id": kb_id,
                "doc_id": pipeline_doc_id,
                "source_uri": source_uri,
                "mime": effective_mime,
                "persisted": false,
                "bytes": bytes.len(),
            })))
        }
        (Err(e), Some(doc), Some(repo)) => {
            let updated = KbDocument {
                status: "failed".into(),
                error: Some(e.to_string()),
                updated_at: Utc::now().naive_utc(),
                ..doc
            };
            let _ = repo.update(&updated).await;
            Err(actix_web::error::ErrorInternalServerError(e.to_string()))
        }
        (Err(e), _, _) => Err(actix_web::error::ErrorInternalServerError(e.to_string())),
    }
}

#[get("/v1/kb/{id}/documents")]
pub async fn list_documents(
    state: web::Data<crate::state::AppState>,
    path: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let repo = state.kb_document_repo.clone().ok_or_else(rag_meta_disabled)?;
    let kb_id = path.into_inner();
    let rows = repo.find_by_kb(&kb_id).await.map_err(map_core_err)?;
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "kb_id": kb_id,
        "items": rows.into_iter().map(DocumentResponse::from).collect::<Vec<_>>()
    })))
}

#[delete("/v1/kb/{kb_id}/documents/{doc_id}")]
pub async fn delete_document(
    state: web::Data<crate::state::AppState>,
    path: web::Path<(String, String)>,
) -> actix_web::Result<HttpResponse> {
    let (kb_id, doc_id) = path.into_inner();
    let pipeline = state.rag_pipeline.clone().ok_or_else(rag_disabled)?;
    let doc_repo = state.kb_document_repo.clone();

    // Remove chunks first; the doc row is the breadcrumb.
    pipeline
        .store
        .delete_by_doc(&kb_id, &doc_id)
        .await
        .map_err(map_core_err)?;

    if let Some(repo) = doc_repo {
        let _ = repo.soft_delete(&doc_id).await;
    }
    Ok(HttpResponse::NoContent().finish())
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub min_score: Option<f32>,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub chunk_id: String,
    pub doc_id: String,
    pub ord: u32,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
}

impl From<RagHit> for SearchHit {
    fn from(h: RagHit) -> Self {
        Self {
            chunk_id: h.chunk.id,
            doc_id: h.chunk.doc_id,
            ord: h.chunk.ord,
            text: h.chunk.text,
            score: h.score,
            metadata: h.chunk.metadata,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub kb_id: String,
    pub hits: Vec<SearchHit>,
}

#[post("/v1/kb/{id}/search")]
pub async fn search_kb(
    state: web::Data<crate::state::AppState>,
    path: web::Path<String>,
    body: web::Json<SearchRequest>,
) -> actix_web::Result<HttpResponse> {
    let kb_id = path.into_inner();
    let pipeline = state.rag_pipeline.clone().ok_or_else(rag_disabled)?;

    let hits = pipeline
        .search(&kb_id, &body.query, body.top_k)
        .await
        .map_err(map_core_err)?;

    let filtered: Vec<SearchHit> = hits
        .into_iter()
        .filter(|h| body.min_score.map_or(true, |m| h.score >= m))
        .map(SearchHit::from)
        .collect();

    Ok(HttpResponse::Ok().json(SearchResponse {
        kb_id,
        hits: filtered,
    }))
}
