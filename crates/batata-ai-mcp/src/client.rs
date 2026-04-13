use serde::{Deserialize, Serialize};
use tokio::process::Command;

use batata_ai_core::error::BatataError;

use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, Content, RawContent},
    service::{Peer, RoleClient, RunningService},
    transport::{StreamableHttpClientTransport, TokioChildProcess},
};

/// Information about a tool available on an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    /// The name of the tool.
    pub name: String,
    /// An optional human-readable description of what the tool does.
    pub description: Option<String>,
}

/// A single content block returned from a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContent {
    /// The type of content: "text", "image", "resource", "audio", etc.
    pub content_type: String,
    /// Text content, if the content type is "text".
    pub text: Option<String>,
}

/// The result of calling a tool on an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The content blocks returned by the tool.
    pub content: Vec<ToolContent>,
    /// Whether the tool call resulted in an error.
    pub is_error: bool,
}

/// Internal enum to hold the running service with different transport types.
enum ClientInner {
    Stdio(RunningService<RoleClient, ()>),
    Http(RunningService<RoleClient, ()>),
}

impl ClientInner {
    fn peer(&self) -> &Peer<RoleClient> {
        match self {
            ClientInner::Stdio(s) => s.peer(),
            ClientInner::Http(s) => s.peer(),
        }
    }
}

/// An MCP client that can connect to an MCP server and invoke its tools.
///
/// Supports two transport modes:
/// - **stdio**: spawns a subprocess and communicates via stdin/stdout
/// - **HTTP (Streamable HTTP)**: connects to an HTTP-based MCP server endpoint
///
/// # Example
///
/// ```rust,no_run
/// # async fn example() -> Result<(), batata_ai_core::error::BatataError> {
/// use batata_ai_mcp::client::McpClient;
///
/// // Connect via stdio
/// let client = McpClient::connect_stdio("my-mcp-server", &[]).await?;
/// let tools = client.list_tools().await?;
/// for tool in &tools {
///     println!("{}: {:?}", tool.name, tool.description);
/// }
///
/// let result = client.call_tool("my_tool", serde_json::json!({"key": "value"})).await?;
/// println!("result: {:?}", result);
///
/// client.shutdown().await?;
/// # Ok(())
/// # }
/// ```
pub struct McpClient {
    inner: ClientInner,
}

impl McpClient {
    /// Connect to an MCP server by spawning a subprocess and communicating via stdio.
    ///
    /// # Arguments
    /// * `command` - The command to execute (e.g. `"npx"`, `"python"`, or a path to a binary).
    /// * `args` - Arguments to pass to the command.
    ///
    /// # Errors
    /// Returns `BatataError::Mcp` if the process cannot be spawned or the MCP handshake fails.
    pub async fn connect_stdio(command: &str, args: &[&str]) -> Result<Self, BatataError> {
        let mut cmd = Command::new(command);
        cmd.args(args);

        let transport = TokioChildProcess::new(cmd)
            .map_err(|e| BatataError::Mcp(format!("failed to spawn MCP server process: {e}")))?;

        let service = ().serve(transport).await.map_err(|e| {
            BatataError::Mcp(format!("MCP client initialization failed (stdio): {e}"))
        })?;

        Ok(Self {
            inner: ClientInner::Stdio(service),
        })
    }

    /// Connect to an MCP server over HTTP (Streamable HTTP transport).
    ///
    /// Uses the `reqwest` HTTP client (provided by rmcp) under the hood.
    ///
    /// # Arguments
    /// * `url` - The full URL of the MCP server endpoint (e.g. `"http://localhost:8080/mcp"`).
    ///
    /// # Errors
    /// Returns `BatataError::Mcp` if the connection or MCP handshake fails.
    pub async fn connect_sse(url: &str) -> Result<Self, BatataError> {
        let transport = StreamableHttpClientTransport::from_uri(url);

        let service = ().serve(transport).await.map_err(|e| {
            BatataError::Mcp(format!("MCP client initialization failed (HTTP): {e}"))
        })?;

        Ok(Self {
            inner: ClientInner::Http(service),
        })
    }

    /// List all tools available on the connected MCP server.
    ///
    /// Handles pagination automatically -- all tools are returned in a single `Vec`.
    pub async fn list_tools(&self) -> Result<Vec<ToolInfo>, BatataError> {
        let tools = self
            .inner
            .peer()
            .list_all_tools()
            .await
            .map_err(|e| BatataError::Mcp(format!("failed to list tools: {e}")))?;

        Ok(tools
            .into_iter()
            .map(|t| ToolInfo {
                name: t.name.to_string(),
                description: t.description.map(|d| d.to_string()),
            })
            .collect())
    }

    /// Call a tool by name with JSON arguments.
    ///
    /// # Arguments
    /// * `name` - The name of the tool to call.
    /// * `args` - A JSON value representing the tool arguments. Must be a JSON object
    ///   (i.e. `serde_json::Value::Object`), or `serde_json::Value::Null` for no arguments.
    ///
    /// # Errors
    /// Returns `BatataError::Mcp` if the arguments are not a JSON object, or if the call fails.
    pub async fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult, BatataError> {
        let arguments = match args {
            serde_json::Value::Object(map) => Some(map),
            serde_json::Value::Null => None,
            _ => {
                return Err(BatataError::Mcp(
                    "tool arguments must be a JSON object or null".to_string(),
                ));
            }
        };

        let mut params = CallToolRequestParams::new(name.to_string());
        if let Some(args) = arguments {
            params = params.with_arguments(args);
        }

        let result = self
            .inner
            .peer()
            .call_tool(params)
            .await
            .map_err(|e| BatataError::Mcp(format!("tool call '{name}' failed: {e}")))?;

        Ok(ToolResult {
            content: result.content.into_iter().map(convert_content).collect(),
            is_error: result.is_error.unwrap_or(false),
        })
    }

    /// Shutdown the MCP client, closing the connection to the server.
    ///
    /// For stdio transport, this also terminates the child process.
    pub async fn shutdown(self) -> Result<(), BatataError> {
        match self.inner {
            ClientInner::Stdio(service) => {
                service
                    .cancel()
                    .await
                    .map_err(|e| BatataError::Mcp(format!("shutdown failed: {e}")))?;
            }
            ClientInner::Http(service) => {
                service
                    .cancel()
                    .await
                    .map_err(|e| BatataError::Mcp(format!("shutdown failed: {e}")))?;
            }
        }
        Ok(())
    }
}

/// Convert an rmcp `Content` (which is `Annotated<RawContent>`) into our `ToolContent`.
fn convert_content(content: Content) -> ToolContent {
    match content.raw {
        RawContent::Text(text) => ToolContent {
            content_type: "text".to_string(),
            text: Some(text.text),
        },
        RawContent::Image(img) => ToolContent {
            content_type: "image".to_string(),
            text: Some(format!("data:{};base64,{}", img.mime_type, img.data)),
        },
        RawContent::Audio(audio) => ToolContent {
            content_type: "audio".to_string(),
            text: Some(format!("data:{};base64,{}", audio.mime_type, audio.data)),
        },
        RawContent::Resource(resource) => ToolContent {
            content_type: "resource".to_string(),
            text: Some(format!("{:?}", resource)),
        },
        RawContent::ResourceLink(link) => ToolContent {
            content_type: "resource_link".to_string(),
            text: Some(link.uri.to_string()),
        },
    }
}
