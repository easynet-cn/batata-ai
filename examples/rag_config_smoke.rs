//! Parse a RAG config file and build the pipeline — verifies the
//! config → builder → pipeline path end-to-end without starting the
//! full API server.

use std::path::PathBuf;

use batata_ai::core::config::BatataConfig;
use batata_ai::rag::build_pipeline;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,batata_ai=debug")
        .init();

    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: rag_config_smoke <path-to-toml>"))?;

    println!("loading config from {}", path.display());
    let cfg = BatataConfig::load(Some(&path))?;
    println!("parsed rag config: {:?}", cfg.rag);

    // This smoke test never exercises the DB store — pass None.
    // This smoke test never exercises the DB store — pass None.
    let pipeline = build_pipeline(&cfg.rag, None)?;
    let Some(p) = pipeline else {
        println!("pipeline disabled");
        return Ok(());
    };
    println!("pipeline built OK, embedder dim = {}", p.embedder.dimensions());

    // Ingest a small mixed CN/EN corpus.
    let corpus = [
        ("mem://cat", "一只小猫正在窗边安静地晒太阳。"),
        ("mem://rust", "Rust 是一门注重内存安全的系统编程语言。"),
        ("mem://tea", "在寒冷的冬天,热茶可以暖暖身子。"),
        ("mem://ml", "深度学习和向量检索是信息检索的基础技术。"),
    ];
    for (uri, text) in corpus {
        let doc = p.ingest_text("smoke-kb", uri, text).await?;
        println!("ingested {uri} -> doc {doc}");
    }

    // Run a few queries — compare semantic match quality.
    for q in ["小猫喵喵", "编程语言", "向量搜索"] {
        let hits = p.search("smoke-kb", q, 2).await?;
        println!("\nquery: {q}");
        for h in hits {
            println!("  score={:.3} text={}", h.score, h.chunk.text);
        }
    }
    Ok(())
}
