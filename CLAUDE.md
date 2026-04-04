# batata-ai

Rust-based AI platform (AI底座) — workspace project.

## Tech Stack
- **Provider abstraction**: rig-core
- **Local inference**: candle (GGUF quantization)
- **MCP protocol**: rmcp (official SDK)
- **HTTP API**: actix-web
- **Async runtime**: tokio
- **ORM**: sea-orm (SQLite/MySQL/PostgreSQL)
- **Object storage**: local/S3/MinIO/Alibaba OSS

## Crate Naming
All crates use `batata-ai-` prefix (e.g., `batata-ai-core`).

## Structure (9 crates)
- `crates/batata-ai-core` — Core traits, domain models, repository traits, routing abstractions
- `crates/batata-ai-provider` — Provider implementations (OpenAI, Anthropic, Ollama, OpenRouter)
- `crates/batata-ai-mcp` — MCP server/client (rmcp)
- `crates/batata-ai-prompt` — Prompt template engine
- `crates/batata-ai-local` — Local candle inference engine
- `crates/batata-ai-storage` — sea-orm based persistence (17 tables)
- `crates/batata-ai-router` — Routing engine with policy-based provider selection
- `crates/batata-ai-object-store` — Object storage backends (local/S3/OSS)
- `crates/batata-ai-api` — HTTP API gateway (actix-web)
- `src/lib.rs` — Facade re-exports

## Core Module Layout
- `domain/` — Domain models: model, provider, prompt, skill, routing, cost, object_store, request_log, tenant, api_key, conversation
- `repository.rs` — Repository trait abstractions (generic CRUD + specialized queries)
- `routing.rs` — Runtime routing traits (RoutingPolicy, StatusStore, ProviderStatus)
- `object_store.rs` — ObjectStore trait for file operations

## Database (17 tables)

### Platform-level (no tenant_id)
- `providers`, `models`, `model_providers` — AI provider/model management (many-to-many)
- `model_costs` — Per-provider model pricing
- `object_store_configs` — Object storage credentials

### Mixed-level (tenant_id optional — NULL = platform, Some = tenant)
- `prompts`, `prompt_versions` — Prompt templates with version history
- `skills`, `skill_versions` — Skill definitions with version history
- `routing_policies` — Routing strategy configurations
- `object_store_buckets` — Storage buckets

### Tenant-level (tenant_id required)
- `tenants` — Tenant management
- `api_keys` — API key authentication
- `conversations` — Chat conversations
- `conversation_messages` — Chat messages (no soft delete)
- `request_logs` — Request audit logging (no soft delete)
- `stored_objects` — File metadata

## API Gateway (actix-web)
- `GET /health` — Health check
- `GET /v1/models` — List models
- `POST /v1/chat/completions` — OpenAI-compatible chat
- `CRUD /v1/conversations` — Conversation management
- `GET /v1/conversations/{id}/messages` — Message history
- Auth: Bearer token (API Key → SHA-256 hash lookup → tenant context)

## Cross-cutting Features
- **Multi-tenancy**: Three categories (platform / mixed / tenant-level)
- **Soft delete**: `deleted_at` timestamp on 14 tables (excludes version history, messages, audit logs)
- **Versioning**: Prompts and Skills auto-snapshot to history tables on update, support rollback
- **Enabled/disabled**: Three-layer check for routing: provider.enabled → model.enabled → model_provider.enabled
- **Auth**: API Key with SHA-256 hash, scopes, rate_limit, expiration

## Conventions
- Rust 2024 edition
- `thiserror` for library errors, `anyhow` in examples/bins
- `async-trait` for async trait definitions
- `tracing` for logging
- UUID v4 string IDs
- `NaiveDateTime` for timestamps in domain models
