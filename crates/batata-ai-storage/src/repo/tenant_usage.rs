use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::TenantUsage;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::TenantUsageRepository;

use crate::entity::tenant_usage;

pub struct SeaOrmTenantUsageRepository {
    db: DatabaseConnection,
}

impl SeaOrmTenantUsageRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl TenantUsageRepository for SeaOrmTenantUsageRepository {
    async fn find_or_create(&self, tenant_id: &str, period: &str) -> Result<TenantUsage> {
        // Try to find existing
        let existing = tenant_usage::Entity::find()
            .filter(tenant_usage::Column::TenantId.eq(tenant_id))
            .filter(tenant_usage::Column::Period.eq(period))
            .one(&self.db)
            .await
            .map_err(map_db_err)?;

        if let Some(model) = existing {
            return Ok(model.into());
        }

        // Create new
        let now = chrono::Utc::now().naive_utc();
        let id = uuid::Uuid::new_v4().to_string();
        let active = tenant_usage::ActiveModel {
            id: Set(id),
            tenant_id: Set(tenant_id.to_string()),
            period: Set(period.to_string()),
            total_requests: Set(0),
            total_prompt_tokens: Set(0),
            total_completion_tokens: Set(0),
            total_tokens: Set(0),
            estimated_cost: Set(0.0),
            created_at: Set(now),
            updated_at: Set(now),
        };

        let result = tenant_usage::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;

        Ok(result.into())
    }

    async fn increment(
        &self,
        tenant_id: &str,
        period: &str,
        requests: i64,
        prompt_tokens: i64,
        completion_tokens: i64,
        cost: f64,
    ) -> Result<TenantUsage> {
        let now = chrono::Utc::now().naive_utc();

        // Try to find existing record
        let existing = tenant_usage::Entity::find()
            .filter(tenant_usage::Column::TenantId.eq(tenant_id))
            .filter(tenant_usage::Column::Period.eq(period))
            .one(&self.db)
            .await
            .map_err(map_db_err)?;

        if let Some(model) = existing {
            // Update with incremented values
            let total_tokens = prompt_tokens + completion_tokens;
            let mut active: tenant_usage::ActiveModel = model.clone().into();
            active.total_requests = Set(model.total_requests + requests);
            active.total_prompt_tokens = Set(model.total_prompt_tokens + prompt_tokens);
            active.total_completion_tokens = Set(model.total_completion_tokens + completion_tokens);
            active.total_tokens = Set(model.total_tokens + total_tokens);
            active.estimated_cost = Set(model.estimated_cost + cost);
            active.updated_at = Set(now);

            let result = active.update(&self.db).await.map_err(map_db_err)?;
            Ok(result.into())
        } else {
            // Insert a new record
            let total_tokens = prompt_tokens + completion_tokens;
            let id = uuid::Uuid::new_v4().to_string();
            let active = tenant_usage::ActiveModel {
                id: Set(id),
                tenant_id: Set(tenant_id.to_string()),
                period: Set(period.to_string()),
                total_requests: Set(requests),
                total_prompt_tokens: Set(prompt_tokens),
                total_completion_tokens: Set(completion_tokens),
                total_tokens: Set(total_tokens),
                estimated_cost: Set(cost),
                created_at: Set(now),
                updated_at: Set(now),
            };

            let result = tenant_usage::Entity::insert(active)
                .exec_with_returning(&self.db)
                .await
                .map_err(map_db_err)?;

            Ok(result.into())
        }
    }

    async fn find_by_tenant_period(
        &self,
        tenant_id: &str,
        period: &str,
    ) -> Result<Option<TenantUsage>> {
        tenant_usage::Entity::find()
            .filter(tenant_usage::Column::TenantId.eq(tenant_id))
            .filter(tenant_usage::Column::Period.eq(period))
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_tenant(&self, tenant_id: &str, limit: u64) -> Result<Vec<TenantUsage>> {
        tenant_usage::Entity::find()
            .filter(tenant_usage::Column::TenantId.eq(tenant_id))
            .order_by_desc(tenant_usage::Column::Period)
            .limit(limit)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }
}
