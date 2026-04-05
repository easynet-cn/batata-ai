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
- `crates/batata-ai-core` — Core traits, domain models, repository traits, routing/cache/guardrail abstractions
- `crates/batata-ai-provider` — Provider implementations (OpenAI, Anthropic, Ollama, OpenRouter)
- `crates/batata-ai-mcp` — MCP server/client (rmcp)
- `crates/batata-ai-prompt` — Prompt template engine
- `crates/batata-ai-local` — Local candle inference engine
- `crates/batata-ai-storage` — sea-orm based persistence (18 tables)
- `crates/batata-ai-router` — Routing engine, cache, guardrails implementations
- `crates/batata-ai-object-store` — Object storage backends (local/S3/OSS)
- `crates/batata-ai-api` — HTTP API gateway (actix-web)
- `src/lib.rs` — Facade re-exports

## Core Module Layout
- `domain/` — 13 domain models: model, provider, prompt, skill, routing, cost, object_store, request_log, tenant, api_key, conversation, usage
- `repository.rs` — 15 repository trait abstractions
- `routing.rs` — Runtime routing traits (RoutingPolicy, StatusStore, ProviderStatus)
- `cache.rs` — CacheStore + CacheKeyStrategy traits
- `guardrail.rs` — Guardrail trait + GuardrailPipeline
- `object_store.rs` — ObjectStore trait for file operations

## Database (18 tables)

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
- `tenant_usages` — Usage tracking per period (no soft delete)

## API Gateway (actix-web)
- `GET /health` — Health check
- `GET /v1/models` — List models
- `POST /v1/chat/completions` — OpenAI-compatible chat (with cache + guardrails)
- `CRUD /v1/conversations` — Conversation management
- `GET /v1/conversations/{id}/messages` — Message history
- `GET /v1/usage` — Tenant usage statistics
- Middleware: auth (API Key), rate limiting (token bucket), request tracing

## Cross-cutting Features
- **Multi-tenancy**: Three categories (platform / mixed / tenant-level)
- **Soft delete**: `deleted_at` timestamp on 14 tables
- **Versioning**: Prompts and Skills auto-snapshot + rollback
- **Routing**: 6 policies (priority/cost/latency/weighted/fallback/chain) + auto-failover
- **Cache**: CacheStore trait + InMemoryCache with TTL, integrated in chat handler
- **Guardrails**: KeywordFilter + LengthLimit, input/output dual-direction filtering
- **Rate limiting**: Per-API-key token bucket
- **Usage tracking**: Per-tenant request/token/cost aggregation by period
- **Auth**: API Key (SHA-256 hash, scopes, rate_limit, expiration)
- **Observability**: Request tracing middleware (method, path, status, latency)

## Conventions
- Rust 2024 edition
- `thiserror` for library errors, `anyhow` in examples/bins
- `async-trait` for async trait definitions
- `tracing` for logging
- UUID v4 string IDs
- `NaiveDateTime` for timestamps in domain models
