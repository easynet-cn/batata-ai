use actix_web::{web, App, HttpServer};

use crate::handler;
use crate::state::AppState;

/// Start the HTTP API server.
pub async fn start(state: AppState, bind: &str) -> std::io::Result<()> {
    let state = web::Data::new(state);

    tracing::info!("starting API server on {}", bind);

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            // Health
            .service(handler::health::health_check)
            // Models
            .service(handler::model::list_models)
            // Chat
            .service(handler::chat::chat_completions)
            // Conversations
            .service(handler::conversation::create_conversation)
            .service(handler::conversation::list_conversations)
            .service(handler::conversation::get_conversation)
            .service(handler::conversation::delete_conversation)
            .service(handler::conversation::list_messages)
    })
    .bind(bind)?
    .run()
    .await
}
