use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::{PromptDefinition, PromptVersion};
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{PromptRepository, Repository};

use crate::entity::{prompt, prompt_version};

pub struct SeaOrmPromptRepository {
    db: DatabaseConnection,
}

impl SeaOrmPromptRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<PromptDefinition> for SeaOrmPromptRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<PromptDefinition>> {
        prompt::Entity::find_by_id(id)
            .filter(prompt::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<PromptDefinition>> {
        prompt::Entity::find()
            .filter(prompt::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &PromptDefinition) -> Result<PromptDefinition> {
        let active = prompt::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            name: Set(entity.name.clone()),
            description: Set(entity.description.clone()),
            template: Set(entity.template.clone()),
            variables: Set(entity.variables.clone()),
            category: Set(entity.category.clone()),
            version: Set(entity.version),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = prompt::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &PromptDefinition) -> Result<PromptDefinition> {
        let active = prompt::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            name: Set(entity.name.clone()),
            description: Set(entity.description.clone()),
            template: Set(entity.template.clone()),
            variables: Set(entity.variables.clone()),
            category: Set(entity.category.clone()),
            version: Set(entity.version),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = prompt::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<prompt::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<prompt::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<prompt::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl PromptRepository for SeaOrmPromptRepository {
    async fn find_by_name(&self, name: &str) -> Result<Option<PromptDefinition>> {
        prompt::Entity::find()
            .filter(prompt::Column::Name.eq(name))
            .filter(prompt::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_category(&self, category: &str) -> Result<Vec<PromptDefinition>> {
        prompt::Entity::find()
            .filter(prompt::Column::Category.eq(category))
            .filter(prompt::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<PromptDefinition>> {
        prompt::Entity::find()
            .filter(
                Condition::any()
                    .add(prompt::Column::TenantId.is_null())
                    .add(prompt::Column::TenantId.eq(tenant_id)),
            )
            .filter(prompt::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn update_with_version(
        &self,
        entity: &PromptDefinition,
        change_message: Option<&str>,
        changed_by: Option<&str>,
    ) -> Result<PromptDefinition> {
        // 1. Find current row to snapshot
        let current = prompt::Entity::find_by_id(&entity.id)
            .filter(prompt::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map_err(map_db_err)?
            .ok_or_else(|| BatataError::NotFound(format!("prompt {}", entity.id)))?;

        // 2. Snapshot current -> prompt_versions
        let version_id = uuid::Uuid::new_v4().to_string();
        let snapshot = prompt_version::ActiveModel {
            id: Set(version_id),
            prompt_id: Set(current.id.clone()),
            version: Set(current.version),
            template: Set(current.template.clone()),
            variables: Set(current.variables.clone()),
            description: Set(current.description.clone()),
            change_message: Set(change_message.map(|s| s.to_string())),
            changed_by: Set(changed_by.map(|s| s.to_string())),
            created_at: Set(current.updated_at),
        };
        prompt_version::Entity::insert(snapshot)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;

        // 3. Update main row with new version = current + 1
        let new_version = current.version + 1;
        let active = prompt::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            name: Set(entity.name.clone()),
            description: Set(entity.description.clone()),
            template: Set(entity.template.clone()),
            variables: Set(entity.variables.clone()),
            category: Set(entity.category.clone()),
            version: Set(new_version),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = prompt::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn find_versions(&self, prompt_id: &str) -> Result<Vec<PromptVersion>> {
        prompt_version::Entity::find()
            .filter(prompt_version::Column::PromptId.eq(prompt_id))
            .order_by_desc(prompt_version::Column::Version)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_version(
        &self,
        prompt_id: &str,
        version: i32,
    ) -> Result<Option<PromptVersion>> {
        prompt_version::Entity::find()
            .filter(prompt_version::Column::PromptId.eq(prompt_id))
            .filter(prompt_version::Column::Version.eq(version))
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn rollback_to_version(
        &self,
        prompt_id: &str,
        version: i32,
    ) -> Result<PromptDefinition> {
        // 1. Find the target version snapshot
        let snapshot = prompt_version::Entity::find()
            .filter(prompt_version::Column::PromptId.eq(prompt_id))
            .filter(prompt_version::Column::Version.eq(version))
            .one(&self.db)
            .await
            .map_err(map_db_err)?
            .ok_or_else(|| {
                BatataError::NotFound(format!("prompt {} version {}", prompt_id, version))
            })?;

        // 2. Find current main row
        let current = prompt::Entity::find_by_id(prompt_id)
            .filter(prompt::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map_err(map_db_err)?
            .ok_or_else(|| BatataError::NotFound(format!("prompt {}", prompt_id)))?;

        // 3. Snapshot current before rollback
        let version_id = uuid::Uuid::new_v4().to_string();
        let current_snapshot = prompt_version::ActiveModel {
            id: Set(version_id),
            prompt_id: Set(current.id.clone()),
            version: Set(current.version),
            template: Set(current.template.clone()),
            variables: Set(current.variables.clone()),
            description: Set(current.description.clone()),
            change_message: Set(Some(format!("before rollback to v{}", version))),
            changed_by: Set(None),
            created_at: Set(current.updated_at),
        };
        prompt_version::Entity::insert(current_snapshot)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;

        // 4. Update main row with snapshot content, version = current + 1
        let now = chrono::Utc::now().naive_utc();
        let new_version = current.version + 1;
        let active = prompt::ActiveModel {
            id: Set(prompt_id.to_string()),
            tenant_id: Set(current.tenant_id),
            name: Set(current.name),
            description: Set(snapshot.description),
            template: Set(snapshot.template),
            variables: Set(snapshot.variables),
            category: Set(current.category),
            version: Set(new_version),
            created_at: Set(current.created_at),
            updated_at: Set(now),
            deleted_at: Set(None),
        };
        let result = prompt::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }
}
