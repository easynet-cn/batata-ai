# batata-ai

Rust-based AI platform (AI底座) — workspace project.

## Tech Stack
- **Provider abstraction**: rig-core
- **Local inference**: candle (GGUF quantization)
- **MCP protocol**: rmcp (official SDK)
- **Async runtime**: tokio

## Crate Naming
All crates use `batata-ai-` prefix (e.g., `batata-ai-core`).

## Structure
- `crates/batata-ai-core` — Core traits: Provider, Skill, Message, Config
- `crates/batata-ai-provider` — Provider implementations (OpenAI, Anthropic, Local, Ollama)
- `crates/batata-ai-mcp` — MCP server/client (rmcp)
- `crates/batata-ai-prompt` — Prompt template engine
- `crates/batata-ai-local` — Local candle inference engine
- `src/lib.rs` — Facade re-exports

## Conventions
- Rust 2024 edition
- `thiserror` for library errors, `anyhow` in examples/bins
- `async-trait` for async trait definitions
- `tracing` for logging
