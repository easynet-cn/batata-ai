use std::io::Write;

use futures::StreamExt;

use batata_ai::core::message::{ChatRequest, Message};
use batata_ai::core::provider::Provider;
use batata_ai::local::provider::LocalProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("=== batata-ai streaming demo ===");
    println!("Loading Qwen2-1.5B (Q4 quantized)...");
    println!("First run will download ~1.1GB model from Hugging Face Hub.\n");

    let provider = LocalProvider::new("qwen2");

    let request = ChatRequest {
        messages: vec![
            Message::system("You are a helpful assistant. Be concise."),
            Message::user("Explain what Rust ownership is in 3 sentences."),
        ],
        model: None,
        temperature: Some(0.7),
        max_tokens: Some(200),
    };

    println!("User: Explain what Rust ownership is in 3 sentences.\n");
    print!("Assistant: ");

    let mut stream = provider.stream_chat(request).await?;

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                print!("{token}");
                std::io::stdout().flush()?;
            }
            Err(e) => {
                eprintln!("\nError: {e}");
                break;
            }
        }
    }

    println!("\n");
    Ok(())
}
