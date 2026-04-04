use std::path::Path;
use std::sync::{Arc, Mutex};

use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::{Json, Parameters}},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
};
use tokio_util::sync::CancellationToken;

use batata_ai_core::message::{ChatRequest, Message};
use batata_ai_core::provider::ProviderRegistry;
use batata_ai_core::skill::SkillRegistry;
use batata_ai_local::blip::BlipModel;
use batata_ai_local::clip::ClipModel;
use batata_ai_local::whisper::{WhisperModel, WhisperSize};

use crate::tools::*;

/// MCP Server that exposes batata-ai providers, skills, and multimodal capabilities.
///
/// **Text tools:** chat, list_providers, list_skills, execute_skill
/// **Multimodal tools:** transcribe, caption, embed_text
///
/// Supports stdio and HTTP/SSE transport.
#[derive(Clone)]
pub struct BatataMcpServer {
    providers: Arc<ProviderRegistry>,
    skills: Arc<SkillRegistry>,
    whisper: Arc<Mutex<Option<WhisperModel>>>,
    blip: Arc<Mutex<Option<BlipModel>>>,
    clip: Arc<Mutex<Option<ClipModel>>>,
    tool_router: ToolRouter<Self>,
}

impl BatataMcpServer {
    pub fn new(providers: Arc<ProviderRegistry>, skills: Arc<SkillRegistry>) -> Self {
        Self {
            providers,
            skills,
            whisper: Arc::new(Mutex::new(None)),
            blip: Arc::new(Mutex::new(None)),
            clip: Arc::new(Mutex::new(None)),
            tool_router: Self::tool_router(),
        }
    }

    fn ensure_whisper(&self) -> Result<(), String> {
        let mut guard = self.whisper.lock().map_err(|e| e.to_string())?;
        if guard.is_none() {
            let device = batata_ai_local::model::resolve_device(false)
                .map_err(|e| e.to_string())?;
            let model = WhisperModel::download_and_load(WhisperSize::Base, &device)
                .map_err(|e| e.to_string())?;
            *guard = Some(model);
        }
        Ok(())
    }

    fn ensure_blip(&self) -> Result<(), String> {
        let mut guard = self.blip.lock().map_err(|e| e.to_string())?;
        if guard.is_none() {
            let device = batata_ai_local::model::resolve_device(false)
                .map_err(|e| e.to_string())?;
            let model = BlipModel::download_and_load(&device).map_err(|e| e.to_string())?;
            *guard = Some(model);
        }
        Ok(())
    }

    fn ensure_clip(&self) -> Result<(), String> {
        let mut guard = self.clip.lock().map_err(|e| e.to_string())?;
        if guard.is_none() {
            let device = batata_ai_local::model::resolve_device(false)
                .map_err(|e| e.to_string())?;
            let model = ClipModel::download_and_load(&device).map_err(|e| e.to_string())?;
            *guard = Some(model);
        }
        Ok(())
    }

    /// Start the MCP server on stdio transport
    pub async fn serve_stdio(
        self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use rmcp::ServiceExt;
        tracing::info!("starting batata-ai MCP server on stdio");
        let transport = (tokio::io::stdin(), tokio::io::stdout());
        let running = self.serve(transport).await?;
        running.waiting().await?;
        Ok(())
    }

    /// Start the MCP server on HTTP/SSE transport.
    /// Endpoint: `http://{addr}/mcp`
    /// ```ignore
    /// server.serve_http("0.0.0.0:8080").await?;
    /// ```
    pub async fn serve_http(
        self,
        addr: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ct = CancellationToken::new();
        let config = StreamableHttpServerConfig::default()
            .with_cancellation_token(ct.child_token());
        self.serve_http_with_config(addr, config).await
    }

    /// Start the MCP server on HTTP/SSE with custom configuration.
    pub async fn serve_http_with_config(
        self,
        addr: &str,
        config: StreamableHttpServerConfig,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ct = config.cancellation_token.clone();
        let session_manager = Arc::new(LocalSessionManager::default());

        let providers = self.providers.clone();
        let skills = self.skills.clone();

        let service = StreamableHttpService::new(
            move || Ok(BatataMcpServer::new(providers.clone(), skills.clone())),
            session_manager,
            config,
        );

        let router = axum::Router::new().nest_service("/mcp", service);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;
        tracing::info!("batata-ai MCP server listening on http://{local_addr}/mcp");

        axum::serve(listener, router)
            .with_graceful_shutdown(async move { ct.cancelled().await })
            .await?;

        Ok(())
    }

    /// Create an axum Router with the MCP service mounted at `/mcp`.
    pub fn into_router(self) -> axum::Router {
        let ct = CancellationToken::new();
        let config = StreamableHttpServerConfig::default()
            .with_cancellation_token(ct.child_token());
        let session_manager = Arc::new(LocalSessionManager::default());

        let providers = self.providers.clone();
        let skills = self.skills.clone();

        let service = StreamableHttpService::new(
            move || Ok(BatataMcpServer::new(providers.clone(), skills.clone())),
            session_manager,
            config,
        );

        axum::Router::new().nest_service("/mcp", service)
    }
}

#[tool_router]
impl BatataMcpServer {
    // ── Text tools ───────────────────────────────────────────

    #[tool(description = "Send a message to the AI provider and get a response")]
    async fn chat(
        &self,
        Parameters(params): Parameters<ChatParams>,
    ) -> Result<String, String> {
        let provider = self
            .providers
            .default_provider()
            .ok_or_else(|| "no default provider configured".to_string())?;

        let mut messages = Vec::new();
        if let Some(system) = params.system {
            messages.push(Message::system(system));
        }
        messages.push(Message::user(params.message));

        let request = ChatRequest {
            messages,
            model: None,
            temperature: params.temperature,
            max_tokens: params.max_tokens,
        };

        let response = provider.chat(request).await.map_err(|e| e.to_string())?;
        Ok(response.content)
    }

    #[tool(description = "List all registered AI providers and their capabilities")]
    fn list_providers(&self, _params: Parameters<ListProvidersParams>) -> Json<ProvidersResult> {
        let providers = self
            .providers
            .list()
            .iter()
            .filter_map(|name| {
                self.providers.get(name).map(|p| {
                    let caps = p.capabilities();
                    ProviderInfo {
                        name: name.to_string(),
                        chat: caps.chat,
                        streaming: caps.streaming,
                        embeddings: caps.embeddings,
                        function_calling: caps.function_calling,
                    }
                })
            })
            .collect();
        Json(ProvidersResult { providers })
    }

    #[tool(description = "List all registered skills")]
    fn list_skills(&self, _params: Parameters<ListSkillsParams>) -> Json<SkillsResult> {
        let skills = self
            .skills
            .list()
            .into_iter()
            .map(|(name, desc)| SkillInfo {
                name: name.to_string(),
                description: desc.to_string(),
            })
            .collect();
        Json(SkillsResult { skills })
    }

    #[tool(description = "Execute a skill by name with the given parameters")]
    async fn execute_skill(
        &self,
        Parameters(params): Parameters<ExecuteSkillParams>,
    ) -> Result<String, String> {
        let skill = self
            .skills
            .get(&params.skill_name)
            .ok_or_else(|| format!("skill not found: {}", params.skill_name))?;
        let output = skill.execute(params.params).await.map_err(|e| e.to_string())?;
        Ok(output.content)
    }

    // ── Multimodal tools ─────────────────────────────────────

    /// Transcribe a WAV audio file to text using Whisper
    #[tool(description = "Transcribe a WAV audio file to text (speech-to-text). First call downloads Whisper model (~140MB).")]
    fn transcribe(
        &self,
        Parameters(params): Parameters<TranscribeParams>,
    ) -> Result<String, String> {
        self.ensure_whisper()?;
        let mut guard = self.whisper.lock().map_err(|e| e.to_string())?;
        let model = guard.as_mut().ok_or("whisper not loaded")?;
        let result = model
            .transcribe_file(Path::new(&params.file_path))
            .map_err(|e| e.to_string())?;
        Ok(result.text)
    }

    /// Generate a text caption for an image file
    #[tool(description = "Generate a text description of an image (image captioning). Accepts PNG/JPEG path. First call downloads BLIP model (~1.8GB).")]
    fn caption(
        &self,
        Parameters(params): Parameters<CaptionParams>,
    ) -> Result<String, String> {
        self.ensure_blip()?;

        let image_bytes = std::fs::read(&params.file_path)
            .map_err(|e| format!("failed to read image: {e}"))?;

        let image_data = decode_image_file(&image_bytes)
            .map_err(|e| format!("failed to decode image: {e}"))?;

        let mut guard = self.blip.lock().map_err(|e| e.to_string())?;
        let model = guard.as_mut().ok_or("blip not loaded")?;
        model.caption(image_data).map_err(|e| e.to_string())
    }

    /// Generate text embeddings using CLIP
    #[tool(description = "Generate embedding vectors for texts (512-dim). Useful for semantic search. First call downloads CLIP model (~600MB).")]
    fn embed_text(
        &self,
        Parameters(params): Parameters<EmbedTextParams>,
    ) -> Result<Json<EmbedResult>, String> {
        self.ensure_clip()?;
        let guard = self.clip.lock().map_err(|e| e.to_string())?;
        let model = guard.as_ref().ok_or("clip not loaded")?;
        let embeddings = model.embed_texts(&params.texts).map_err(|e| e.to_string())?;
        let dimensions = model.dimensions();
        Ok(Json(EmbedResult {
            embeddings,
            dimensions,
        }))
    }
}

#[tool_handler]
impl ServerHandler for BatataMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "batata-ai MCP server — AI chat, speech-to-text, image captioning, text embedding, skills",
            )
    }
}

/// Minimal image decoder: supports raw PPM or assumes raw RGB bytes with dimensions from header.
/// For production use, integrate the `image` crate.
fn decode_image_file(
    bytes: &[u8],
) -> std::result::Result<batata_ai_core::multimodal::ImageData, String> {
    // Try PPM (P6) format
    if bytes.len() > 3 && &bytes[0..2] == b"P6" {
        return decode_ppm(bytes);
    }

    // For other formats (PNG/JPEG), return an error suggesting the image crate
    Err("unsupported image format. Currently supports PPM (P6). For PNG/JPEG, convert to PPM first: `convert input.jpg -depth 8 output.ppm`".into())
}

fn decode_ppm(bytes: &[u8]) -> std::result::Result<batata_ai_core::multimodal::ImageData, String> {
    let s = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    let mut parts = s.split_ascii_whitespace();
    let _magic = parts.next().ok_or("missing magic")?;
    let width: u32 = parts
        .next()
        .ok_or("missing width")?
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let height: u32 = parts
        .next()
        .ok_or("missing height")?
        .parse()
        .map_err(|e: std::num::ParseIntError| e.to_string())?;
    let _max_val = parts.next().ok_or("missing max value")?;

    // Find the start of binary data (after the header newline)
    let header_end = bytes
        .windows(1)
        .enumerate()
        .filter(|(_, w)| w[0] == b'\n')
        .nth(2)
        .map(|(i, _)| i + 1)
        .ok_or("invalid PPM header")?;

    let pixel_data = &bytes[header_end..];
    let expected = (width * height * 3) as usize;
    if pixel_data.len() < expected {
        return Err(format!(
            "PPM data too short: expected {expected}, got {}",
            pixel_data.len()
        ));
    }

    Ok(batata_ai_core::multimodal::ImageData {
        bytes: pixel_data[..expected].to_vec(),
        width,
        height,
    })
}
