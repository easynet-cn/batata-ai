use batata_ai::local::clip::ClipModel;
use batata_ai::local::model::resolve_device;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    println!("=== batata-ai CLIP embedding demo ===\n");

    let device = resolve_device(false)?;
    let clip = ClipModel::download_and_load(&device)?;

    // Demo: compute semantic similarity between texts
    let texts = vec![
        "a photo of a cat".to_string(),
        "a photo of a dog".to_string(),
        "the Rust programming language".to_string(),
        "a cute kitten sleeping".to_string(),
    ];

    println!("Generating embeddings for {} texts...\n", texts.len());
    let embeddings = clip.embed_texts(&texts)?;

    println!("Dimensions: {}\n", clip.dimensions());
    println!("Cosine similarity matrix:");
    println!("{:<35} ", "");
    for (i, t) in texts.iter().enumerate() {
        print!("[{}] ", i);
        let display: String = t.chars().take(20).collect();
        print!("{:<22} ", display);
    }
    println!();

    for (i, t) in texts.iter().enumerate() {
        let display: String = t.chars().take(30).collect();
        print!("[{}] {:<30} ", i, display);
        for (j, _) in texts.iter().enumerate() {
            let sim = ClipModel::cosine_similarity(&embeddings[i], &embeddings[j]);
            print!("{:>6.3}                 ", sim);
        }
        println!();
    }

    println!();
    println!("Note: Higher similarity = more semantically related");
    println!("  'cat' vs 'kitten' should be high");
    println!("  'cat' vs 'Rust lang' should be low");

    Ok(())
}
