use async_trait::async_trait;
use chrono::NaiveDateTime;
use sea_orm::*;

use batata_ai_core::domain::RequestLog;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::RequestLogRepository;

use crate::entity::request_log;

pub struct SeaOrmRequestLogRepository {
    db: DatabaseConnection,
}

impl SeaOrmRequestLogRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl RequestLogRepository for SeaOrmRequestLogRepository {
    async fn create(&self, log: &RequestLog) -> Result<RequestLog> {
        let active = request_log::ActiveModel {
            id: Set(log.id.clone()),
            provider_id: Set(log.provider_id.clone()),
            provider_name: Set(log.provider_name.clone()),
            model_identifier: Set(log.model_identifier.clone()),
            routing_policy: Set(log.routing_policy.clone()),
            status: Set(log.status.to_string()),
            latency_ms: Set(log.latency_ms as i64),
            prompt_tokens: Set(log.prompt_tokens.map(|v| v as i32)),
            completion_tokens: Set(log.completion_tokens.map(|v| v as i32)),
            total_tokens: Set(log.total_tokens.map(|v| v as i32)),
            estimated_cost: Set(log.estimated_cost),
            error_message: Set(log.error_message.clone()),
            metadata: Set(log.metadata.clone()),
            created_at: Set(log.created_at),
        };
        let result = request_log::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn find_by_id(&self, id: &str) -> Result<Option<RequestLog>> {
        request_log::Entity::find_by_id(id)
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_time_range(
        &self,
        from: NaiveDateTime,
        to: NaiveDateTime,
        limit: u64,
    ) -> Result<Vec<RequestLog>> {
        request_log::Entity::find()
            .filter(request_log::Column::CreatedAt.gte(from))
            .filter(request_log::Column::CreatedAt.lte(to))
            .order_by_desc(request_log::Column::CreatedAt)
            .limit(limit)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_by_provider(
        &self,
        provider_id: &str,
        limit: u64,
    ) -> Result<Vec<RequestLog>> {
        request_log::Entity::find()
            .filter(request_log::Column::ProviderId.eq(provider_id))
            .order_by_desc(request_log::Column::CreatedAt)
            .limit(limit)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }
}
