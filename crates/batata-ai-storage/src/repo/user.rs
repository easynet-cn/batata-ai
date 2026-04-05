use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::User;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{Repository, UserRepository};

use crate::entity::user;

pub struct SeaOrmUserRepository {
    db: DatabaseConnection,
}

impl SeaOrmUserRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<User> for SeaOrmUserRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<User>> {
        user::Entity::find_by_id(id)
            .filter(user::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<User>> {
        user::Entity::find()
            .filter(user::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &User) -> Result<User> {
        let active = user::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            username: Set(entity.username.clone()),
            password_hash: Set(entity.password_hash.clone()),
            display_name: Set(entity.display_name.clone()),
            email: Set(entity.email.clone()),
            enabled: Set(entity.enabled),
            last_login_at: Set(entity.last_login_at),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = user::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &User) -> Result<User> {
        let active = user::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            username: Set(entity.username.clone()),
            password_hash: Set(entity.password_hash.clone()),
            display_name: Set(entity.display_name.clone()),
            email: Set(entity.email.clone()),
            enabled: Set(entity.enabled),
            last_login_at: Set(entity.last_login_at),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = user::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<user::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<user::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<user::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl UserRepository for SeaOrmUserRepository {
    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<User>> {
        user::Entity::find()
            .filter(user::Column::TenantId.eq(tenant_id))
            .filter(user::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_by_username(&self, tenant_id: &str, username: &str) -> Result<Option<User>> {
        user::Entity::find()
            .filter(user::Column::TenantId.eq(tenant_id))
            .filter(user::Column::Username.eq(username))
            .filter(user::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }
}
