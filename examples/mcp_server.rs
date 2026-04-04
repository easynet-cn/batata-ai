use std::sync::Arc;

use batata_ai::core::provider::ProviderRegistry;
use batata_ai::core::skill::SkillRegistry;
use batata_ai::local::provider::LocalProvider;
use batata_ai::mcp::BatataMcpServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Log to stderr so stdout stays clean for MCP JSON-RPC
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    // Set up providers
    let mut providers = ProviderRegistry::new();
    let local = LocalProvider::new("qwen2");
    providers.register(Box::new(local));
    providers.set_default("local");

    let providers = Arc::new(providers);
    let skills = Arc::new(SkillRegistry::new());

    // Start MCP server on stdio
    let server = BatataMcpServer::new(providers, skills);
    server
        .serve_stdio()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
