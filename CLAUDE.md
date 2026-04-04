# batata-ai

Rust-based AI platform (AI底座) — workspace project.

## Tech Stack
- **Provider abstraction**: rig-core
- **Local inference**: candle (GGUF quantization)
- **MCP protocol**: rmcp (official SDK)
- **Async runtime**: tokio
- **ORM**: sea-orm (SQLite/MySQL/PostgreSQL)
- **Object storage**: local/S3/MinIO/Alibaba OSS

## Crate Naming
All crates use `batata-ai-` prefix (e.g., `batata-ai-core`).

## Structure
- `crates/batata-ai-core` — Core traits, domain models, repository traits, routing abstractions
- `crates/batata-ai-provider` — Provider implementations (OpenAI, Anthropic, Ollama, OpenRouter)
- `crates/batata-ai-mcp` — MCP server/client (rmcp)
- `crates/batata-ai-prompt` — Prompt template engine
- `crates/batata-ai-local` — Local candle inference engine
- `crates/batata-ai-storage` — sea-orm based persistence (13 tables)
- `crates/batata-ai-router` — Routing engine with policy-based provider selection
- `crates/batata-ai-object-store` — Object storage backends (local/S3/OSS)
- `src/lib.rs` — Facade re-exports

## Core Module Layout
- `domain/` — Domain models split by concern (model, provider, prompt, skill, routing, cost, object_store, request_log)
- `repository.rs` — Repository trait abstractions (generic CRUD + specialized queries)
- `routing.rs` — Runtime routing traits (RoutingPolicy, StatusStore, ProviderStatus)
- `object_store.rs` — ObjectStore trait for file operations

## Database (13 tables)
- `providers`, `models`, `model_providers` — AI provider/model management (many-to-many)
- `model_costs` — Per-provider model pricing
- `prompts`, `prompt_versions` — Prompt templates with version history
- `skills`, `skill_versions` — Skill definitions with version history
- `routing_policies` — Routing strategy configurations
- `request_logs` — Request audit logging
- `object_store_configs`, `object_store_buckets`, `stored_objects` — Object storage (two-layer: credentials + buckets)

## Cross-cutting Features
- **Soft delete**: 10 tables support `deleted_at` timestamp (excludes version history and audit logs)
- **Versioning**: Prompts and Skills auto-snapshot to history tables on update, support rollback
- **Enabled/disabled**: Three-layer check for routing: provider.enabled → model.enabled → model_provider.enabled

## Conventions
- Rust 2024 edition
- `thiserror` for library errors, `anyhow` in examples/bins
- `async-trait` for async trait definitions
- `tracing` for logging
- UUID v4 string IDs
- `NaiveDateTime` for timestamps in domain models
