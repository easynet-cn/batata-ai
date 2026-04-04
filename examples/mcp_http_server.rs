use std::sync::Arc;

use batata_ai::core::provider::ProviderRegistry;
use batata_ai::core::skill::SkillRegistry;
use batata_ai::local::provider::LocalProvider;
use batata_ai::mcp::BatataMcpServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    // Set up providers
    let mut providers = ProviderRegistry::new();
    let local = LocalProvider::new("qwen2");
    providers.register(Box::new(local));
    providers.set_default("local");

    let providers = Arc::new(providers);
    let skills = Arc::new(SkillRegistry::new());

    // Start MCP server on HTTP/SSE
    println!("Starting batata-ai MCP server (HTTP/SSE)");
    println!("Endpoint: http://{bind_addr}/mcp");
    println!();
    println!("Test with:");
    println!("  curl -X POST http://{bind_addr}/mcp \\");
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -H 'Accept: application/json, text/event-stream' \\");
    println!("    -d '{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{{}},\"clientInfo\":{{\"name\":\"test\",\"version\":\"1.0\"}}}}}}'");
    println!();

    let server = BatataMcpServer::new(providers, skills);
    server
        .serve_http(&bind_addr)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
