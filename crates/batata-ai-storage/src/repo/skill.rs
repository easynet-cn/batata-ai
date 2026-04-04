use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::{SkillDefinition, SkillVersion};
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{Repository, SkillRepository};

use crate::entity::{skill, skill_version};

pub struct SeaOrmSkillRepository {
    db: DatabaseConnection,
}

impl SeaOrmSkillRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<SkillDefinition> for SeaOrmSkillRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<SkillDefinition>> {
        skill::Entity::find_by_id(id)
            .filter(skill::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<SkillDefinition>> {
        skill::Entity::find()
            .filter(skill::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &SkillDefinition) -> Result<SkillDefinition> {
        let active = skill::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            description: Set(entity.description.clone()),
            parameters_schema: Set(entity.parameters_schema.clone()),
            skill_type: Set(entity.skill_type.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            version: Set(entity.version),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = skill::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &SkillDefinition) -> Result<SkillDefinition> {
        let active = skill::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            description: Set(entity.description.clone()),
            parameters_schema: Set(entity.parameters_schema.clone()),
            skill_type: Set(entity.skill_type.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            version: Set(entity.version),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = skill::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<skill::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<skill::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<skill::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl SkillRepository for SeaOrmSkillRepository {
    async fn find_by_name(&self, name: &str) -> Result<Option<SkillDefinition>> {
        skill::Entity::find()
            .filter(skill::Column::Name.eq(name))
            .filter(skill::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_enabled(&self) -> Result<Vec<SkillDefinition>> {
        skill::Entity::find()
            .filter(skill::Column::Enabled.eq(true))
            .filter(skill::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn update_with_version(
        &self,
        entity: &SkillDefinition,
        change_message: Option<&str>,
        changed_by: Option<&str>,
    ) -> Result<SkillDefinition> {
        // 1. Find current row to snapshot
        let current = skill::Entity::find_by_id(&entity.id)
            .filter(skill::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map_err(map_db_err)?
            .ok_or_else(|| BatataError::NotFound(format!("skill {}", entity.id)))?;

        // 2. Snapshot current → skill_versions
        let version_id = uuid::Uuid::new_v4().to_string();
        let snapshot = skill_version::ActiveModel {
            id: Set(version_id),
            skill_id: Set(current.id.clone()),
            version: Set(current.version),
            description: Set(current.description.clone()),
            parameters_schema: Set(current.parameters_schema.clone()),
            config: Set(current.config.clone()),
            change_message: Set(change_message.map(|s| s.to_string())),
            changed_by: Set(changed_by.map(|s| s.to_string())),
            created_at: Set(current.updated_at),
        };
        skill_version::Entity::insert(snapshot)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;

        // 3. Update main row with version = current + 1
        let new_version = current.version + 1;
        let active = skill::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            description: Set(entity.description.clone()),
            parameters_schema: Set(entity.parameters_schema.clone()),
            skill_type: Set(entity.skill_type.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            version: Set(new_version),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = skill::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn find_versions(&self, skill_id: &str) -> Result<Vec<SkillVersion>> {
        skill_version::Entity::find()
            .filter(skill_version::Column::SkillId.eq(skill_id))
            .order_by_desc(skill_version::Column::Version)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_version(
        &self,
        skill_id: &str,
        version: i32,
    ) -> Result<Option<SkillVersion>> {
        skill_version::Entity::find()
            .filter(skill_version::Column::SkillId.eq(skill_id))
            .filter(skill_version::Column::Version.eq(version))
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn rollback_to_version(
        &self,
        skill_id: &str,
        version: i32,
    ) -> Result<SkillDefinition> {
        // 1. Find the target version snapshot
        let snapshot = skill_version::Entity::find()
            .filter(skill_version::Column::SkillId.eq(skill_id))
            .filter(skill_version::Column::Version.eq(version))
            .one(&self.db)
            .await
            .map_err(map_db_err)?
            .ok_or_else(|| {
                BatataError::NotFound(format!("skill {} version {}", skill_id, version))
            })?;

        // 2. Find current main row
        let current = skill::Entity::find_by_id(skill_id)
            .filter(skill::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map_err(map_db_err)?
            .ok_or_else(|| BatataError::NotFound(format!("skill {}", skill_id)))?;

        // 3. Snapshot current before rollback
        let version_id = uuid::Uuid::new_v4().to_string();
        let current_snapshot = skill_version::ActiveModel {
            id: Set(version_id),
            skill_id: Set(current.id.clone()),
            version: Set(current.version),
            description: Set(current.description.clone()),
            parameters_schema: Set(current.parameters_schema.clone()),
            config: Set(current.config.clone()),
            change_message: Set(Some(format!("before rollback to v{}", version))),
            changed_by: Set(None),
            created_at: Set(current.updated_at),
        };
        skill_version::Entity::insert(current_snapshot)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;

        // 4. Update main row with snapshot content, version = current + 1
        let now = chrono::Utc::now().naive_utc();
        let new_version = current.version + 1;
        let active = skill::ActiveModel {
            id: Set(skill_id.to_string()),
            name: Set(current.name),
            description: Set(snapshot.description),
            parameters_schema: Set(snapshot.parameters_schema),
            skill_type: Set(current.skill_type),
            config: Set(snapshot.config),
            enabled: Set(current.enabled),
            version: Set(new_version),
            created_at: Set(current.created_at),
            updated_at: Set(now),
            deleted_at: Set(None),
        };
        let result = skill::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }
}
