use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::ModelCost;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{ModelCostRepository, Repository};

use crate::entity::model_cost;

pub struct SeaOrmModelCostRepository {
    db: DatabaseConnection,
}

impl SeaOrmModelCostRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<ModelCost> for SeaOrmModelCostRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<ModelCost>> {
        model_cost::Entity::find_by_id(id)
            .filter(model_cost::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<ModelCost>> {
        model_cost::Entity::find()
            .filter(model_cost::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &ModelCost) -> Result<ModelCost> {
        let active = model_cost::ActiveModel {
            id: Set(entity.id.clone()),
            model_id: Set(entity.model_id.clone()),
            provider_id: Set(entity.provider_id.clone()),
            input_cost_per_1k: Set(entity.input_cost_per_1k),
            output_cost_per_1k: Set(entity.output_cost_per_1k),
            currency: Set(entity.currency.clone()),
            effective_from: Set(entity.effective_from),
            created_at: Set(entity.created_at),
            deleted_at: Set(None),
        };
        let result = model_cost::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &ModelCost) -> Result<ModelCost> {
        let active = model_cost::ActiveModel {
            id: Set(entity.id.clone()),
            model_id: Set(entity.model_id.clone()),
            provider_id: Set(entity.provider_id.clone()),
            input_cost_per_1k: Set(entity.input_cost_per_1k),
            output_cost_per_1k: Set(entity.output_cost_per_1k),
            currency: Set(entity.currency.clone()),
            effective_from: Set(entity.effective_from),
            created_at: Set(entity.created_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = model_cost::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<model_cost::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<model_cost::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<model_cost::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl ModelCostRepository for SeaOrmModelCostRepository {
    async fn find_by_model_provider(
        &self,
        model_id: &str,
        provider_id: &str,
    ) -> Result<Option<ModelCost>> {
        model_cost::Entity::find()
            .filter(model_cost::Column::ModelId.eq(model_id))
            .filter(model_cost::Column::ProviderId.eq(provider_id))
            .filter(model_cost::Column::DeletedAt.is_null())
            .order_by_desc(model_cost::Column::EffectiveFrom)
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_model(&self, model_id: &str) -> Result<Vec<ModelCost>> {
        model_cost::Entity::find()
            .filter(model_cost::Column::ModelId.eq(model_id))
            .filter(model_cost::Column::DeletedAt.is_null())
            .order_by_desc(model_cost::Column::EffectiveFrom)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }
}
