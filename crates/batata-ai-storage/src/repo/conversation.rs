use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::Conversation;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{ConversationRepository, Repository};

use crate::entity::conversation;

pub struct SeaOrmConversationRepository {
    db: DatabaseConnection,
}

impl SeaOrmConversationRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<Conversation> for SeaOrmConversationRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<Conversation>> {
        conversation::Entity::find_by_id(id)
            .filter(conversation::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<Conversation>> {
        conversation::Entity::find()
            .filter(conversation::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &Conversation) -> Result<Conversation> {
        let active = conversation::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            title: Set(entity.title.clone()),
            model: Set(entity.model.clone()),
            system_prompt: Set(entity.system_prompt.clone()),
            metadata: Set(entity.metadata.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = conversation::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &Conversation) -> Result<Conversation> {
        let active = conversation::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            title: Set(entity.title.clone()),
            model: Set(entity.model.clone()),
            system_prompt: Set(entity.system_prompt.clone()),
            metadata: Set(entity.metadata.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = conversation::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<conversation::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<conversation::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<conversation::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl ConversationRepository for SeaOrmConversationRepository {
    async fn find_by_tenant(
        &self,
        tenant_id: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<Conversation>> {
        conversation::Entity::find()
            .filter(conversation::Column::TenantId.eq(tenant_id))
            .filter(conversation::Column::DeletedAt.is_null())
            .order_by_desc(conversation::Column::UpdatedAt)
            .offset(page * page_size)
            .limit(page_size)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn count_by_tenant(&self, tenant_id: &str) -> Result<u64> {
        conversation::Entity::find()
            .filter(conversation::Column::TenantId.eq(tenant_id))
            .filter(conversation::Column::DeletedAt.is_null())
            .count(&self.db)
            .await
            .map_err(map_db_err)
    }
}
