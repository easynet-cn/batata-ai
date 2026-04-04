use std::path::PathBuf;

use batata_ai::local::models::{download_model_to, ModelDescriptor};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: download_model <model_name> [target_dir]");
        println!();
        println!("Available models:");
        for name in ModelDescriptor::available_models() {
            let desc = ModelDescriptor::by_name(name).unwrap();
            println!("  {:<15} {} ({})", name, desc.repo_id, desc.filenames[0]);
        }
        println!();
        println!("Examples:");
        println!("  download_model phi3                    # download to ./models/");
        println!("  download_model qwen2 /opt/models       # download to /opt/models/");
        println!("  download_model llama3 ./my-models       # download to ./my-models/");
        return Ok(());
    }

    let model_name = &args[1];
    let target_dir = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        PathBuf::from("./models")
    };

    println!("Downloading '{model_name}' to {}", target_dir.display());
    println!();

    let (model_path, tokenizer_path) = download_model_to(model_name, &target_dir)?;

    println!();
    println!("Done! Files:");
    println!("  Model:     {}", model_path.display());
    println!("  Tokenizer: {}", tokenizer_path.display());
    println!();
    println!("Use in code:");
    println!("  let provider = LocalProvider::from_local(");
    println!("      \"{model_name}\",");
    println!("      \"{}\",", model_path.display());
    println!("      \"{}\",", tokenizer_path.display());
    println!("  );");

    Ok(())
}
