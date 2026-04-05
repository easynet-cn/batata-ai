# Build stage
FROM rust:1.87-slim AS builder

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo build --release --example api_server

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/examples/api_server /app/batata-ai-server

ENV BIND=0.0.0.0:8080
ENV DATABASE_URL=sqlite:///app/data/batata-ai.db?mode=rwc
ENV RUST_LOG=info,batata_ai=debug

EXPOSE 8080

CMD ["/app/batata-ai-server"]
