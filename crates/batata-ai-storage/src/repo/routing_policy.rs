use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::RoutingPolicyDefinition;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{Repository, RoutingPolicyRepository};

use crate::entity::routing_policy;

pub struct SeaOrmRoutingPolicyRepository {
    db: DatabaseConnection,
}

impl SeaOrmRoutingPolicyRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<RoutingPolicyDefinition> for SeaOrmRoutingPolicyRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<RoutingPolicyDefinition>> {
        routing_policy::Entity::find_by_id(id)
            .filter(routing_policy::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<RoutingPolicyDefinition>> {
        routing_policy::Entity::find()
            .filter(routing_policy::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &RoutingPolicyDefinition) -> Result<RoutingPolicyDefinition> {
        let active = routing_policy::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            policy_type: Set(entity.policy_type.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            priority: Set(entity.priority),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = routing_policy::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &RoutingPolicyDefinition) -> Result<RoutingPolicyDefinition> {
        let active = routing_policy::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            policy_type: Set(entity.policy_type.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            priority: Set(entity.priority),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = routing_policy::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<routing_policy::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<routing_policy::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<routing_policy::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl RoutingPolicyRepository for SeaOrmRoutingPolicyRepository {
    async fn find_by_name(&self, name: &str) -> Result<Option<RoutingPolicyDefinition>> {
        routing_policy::Entity::find()
            .filter(routing_policy::Column::Name.eq(name))
            .filter(routing_policy::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_enabled(&self) -> Result<Vec<RoutingPolicyDefinition>> {
        routing_policy::Entity::find()
            .filter(routing_policy::Column::Enabled.eq(true))
            .filter(routing_policy::Column::DeletedAt.is_null())
            .order_by_asc(routing_policy::Column::Priority)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }
}
