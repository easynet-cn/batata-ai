use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::Tenant;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{Repository, TenantRepository};

use crate::entity::tenant;

pub struct SeaOrmTenantRepository {
    db: DatabaseConnection,
}

impl SeaOrmTenantRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<Tenant> for SeaOrmTenantRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<Tenant>> {
        tenant::Entity::find_by_id(id)
            .filter(tenant::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<Tenant>> {
        tenant::Entity::find()
            .filter(tenant::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &Tenant) -> Result<Tenant> {
        let active = tenant::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            slug: Set(entity.slug.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = tenant::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &Tenant) -> Result<Tenant> {
        let active = tenant::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            slug: Set(entity.slug.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = tenant::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<tenant::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<tenant::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<tenant::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl TenantRepository for SeaOrmTenantRepository {
    async fn find_by_slug(&self, slug: &str) -> Result<Option<Tenant>> {
        tenant::Entity::find()
            .filter(tenant::Column::Slug.eq(slug))
            .filter(tenant::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_enabled(&self) -> Result<Vec<Tenant>> {
        tenant::Entity::find()
            .filter(tenant::Column::Enabled.eq(true))
            .filter(tenant::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }
}
