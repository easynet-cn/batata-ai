use actix_web_prom::{PrometheusMetrics, PrometheusMetricsBuilder};
use prometheus::{
    opts, register_counter_vec_with_registry, register_gauge_with_registry,
    register_histogram_vec_with_registry, CounterVec, Gauge, HistogramVec, Registry,
};

/// Holds all custom application-level Prometheus metrics.
///
/// Inject via `web::Data<Metrics>` so handlers can record domain-specific
/// measurements (chat completions, token counts, cache behaviour, etc.).
#[derive(Clone)]
pub struct Metrics {
    /// Total chat completion requests, labelled by `model` and `status`.
    pub chat_requests_total: CounterVec,
    /// Chat request duration in seconds, labelled by `model`.
    pub chat_request_duration_seconds: HistogramVec,
    /// Total tokens consumed, labelled by `model` and `type` (prompt | completion).
    pub tokens_total: CounterVec,
    /// Total cache hits.
    pub cache_hits_total: CounterVec,
    /// Total cache misses.
    pub cache_misses_total: CounterVec,
    /// Number of in-flight requests.
    pub active_requests: Gauge,
}

impl Metrics {
    /// Register all custom metrics against the supplied `Registry`.
    fn new(registry: &Registry) -> Self {
        let chat_requests_total = register_counter_vec_with_registry!(
            opts!("batata_ai_chat_requests_total", "Total chat completion requests"),
            &["model", "status"],
            registry
        )
        .expect("failed to register batata_ai_chat_requests_total");

        let chat_request_duration_seconds = register_histogram_vec_with_registry!(
            prometheus::histogram_opts!(
                "batata_ai_chat_request_duration_seconds",
                "Chat request duration in seconds"
            ),
            &["model"],
            registry
        )
        .expect("failed to register batata_ai_chat_request_duration_seconds");

        let tokens_total = register_counter_vec_with_registry!(
            opts!("batata_ai_tokens_total", "Total tokens consumed"),
            &["model", "type"],
            registry
        )
        .expect("failed to register batata_ai_tokens_total");

        let cache_hits_total = register_counter_vec_with_registry!(
            opts!("batata_ai_cache_hits_total", "Total cache hits"),
            &[],
            registry
        )
        .expect("failed to register batata_ai_cache_hits_total");

        let cache_misses_total = register_counter_vec_with_registry!(
            opts!("batata_ai_cache_misses_total", "Total cache misses"),
            &[],
            registry
        )
        .expect("failed to register batata_ai_cache_misses_total");

        let active_requests = register_gauge_with_registry!(
            opts!("batata_ai_active_requests", "Number of in-flight requests"),
            registry
        )
        .expect("failed to register batata_ai_active_requests");

        Self {
            chat_requests_total,
            chat_request_duration_seconds,
            tokens_total,
            cache_hits_total,
            cache_misses_total,
            active_requests,
        }
    }
}

/// Create the Prometheus metrics middleware **and** the companion [`Metrics`]
/// struct that handlers can use to record domain-specific counters.
///
/// The returned `PrometheusMetrics` should be added to the actix-web `App` via
/// `.wrap(prometheus_metrics)`, and the `Metrics` value should be registered as
/// `web::Data<Metrics>`.
pub fn create_metrics() -> (PrometheusMetrics, Metrics) {
    let registry = Registry::new();
    let custom = Metrics::new(&registry);

    let prometheus = PrometheusMetricsBuilder::new("batata_ai")
        .registry(registry)
        .endpoint("/metrics")
        .build()
        .expect("failed to create PrometheusMetrics middleware");

    (prometheus, custom)
}
