use batata_ai::core::message::{ChatRequest, Message};
use batata_ai::core::provider::Provider;
use batata_ai::local::provider::LocalProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("=== batata-ai local inference demo ===");
    println!("Loading Qwen2-1.5B (Q4 quantized)...");
    println!("First run will download ~1.1GB model from Hugging Face Hub.\n");

    let provider = LocalProvider::new("qwen2");

    let request = ChatRequest {
        messages: vec![
            Message::system("You are a helpful assistant. Be concise."),
            Message::user("What is Rust programming language in one sentence?"),
        ],
        model: None,
        temperature: Some(0.7),
        max_tokens: Some(128),
    };

    println!("User: What is Rust programming language in one sentence?\n");

    let response = provider.chat(request).await?;

    println!("Assistant: {}", response.content);
    println!("\n[model: {}]", response.model);

    Ok(())
}
