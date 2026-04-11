use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::rag::{KnowledgeBase, KnowledgeBaseRepository};

use crate::entity::knowledge_base;

pub struct SeaOrmKnowledgeBaseRepository {
    db: DatabaseConnection,
}

impl SeaOrmKnowledgeBaseRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

fn model_to_domain(m: knowledge_base::Model) -> KnowledgeBase {
    KnowledgeBase {
        id: m.id,
        tenant_id: m.tenant_id,
        name: m.name,
        description: m.description,
        embedder: m.embedder,
        dim: m.dim,
        chunk_window: m.chunk_window,
        chunk_overlap: m.chunk_overlap,
        metadata: m.metadata,
        created_at: m.created_at,
        updated_at: m.updated_at,
        deleted_at: m.deleted_at,
    }
}

#[async_trait]
impl KnowledgeBaseRepository for SeaOrmKnowledgeBaseRepository {
    async fn create(&self, entity: &KnowledgeBase) -> Result<KnowledgeBase> {
        let active = knowledge_base::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            name: Set(entity.name.clone()),
            description: Set(entity.description.clone()),
            embedder: Set(entity.embedder.clone()),
            dim: Set(entity.dim),
            chunk_window: Set(entity.chunk_window),
            chunk_overlap: Set(entity.chunk_overlap),
            metadata: Set(entity.metadata.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let row = knowledge_base::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(model_to_domain(row))
    }

    async fn find_by_id(&self, id: &str) -> Result<Option<KnowledgeBase>> {
        knowledge_base::Entity::find_by_id(id)
            .filter(knowledge_base::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(model_to_domain))
            .map_err(map_db_err)
    }

    async fn find_by_tenant(&self, tenant_id: Option<&str>) -> Result<Vec<KnowledgeBase>> {
        let mut query = knowledge_base::Entity::find()
            .filter(knowledge_base::Column::DeletedAt.is_null());
        query = match tenant_id {
            // Platform-level only
            None => query.filter(knowledge_base::Column::TenantId.is_null()),
            // Mixed: platform KBs + this tenant's KBs
            Some(tid) => query.filter(
                Condition::any()
                    .add(knowledge_base::Column::TenantId.is_null())
                    .add(knowledge_base::Column::TenantId.eq(tid)),
            ),
        };
        query
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(model_to_domain).collect())
            .map_err(map_db_err)
    }

    async fn update(&self, entity: &KnowledgeBase) -> Result<KnowledgeBase> {
        let active = knowledge_base::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            name: Set(entity.name.clone()),
            description: Set(entity.description.clone()),
            embedder: Set(entity.embedder.clone()),
            dim: Set(entity.dim),
            chunk_window: Set(entity.chunk_window),
            chunk_overlap: Set(entity.chunk_overlap),
            metadata: Set(entity.metadata.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let row = knowledge_base::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(model_to_domain(row))
    }

    async fn soft_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<knowledge_base::Entity>(&self.db, id).await
    }
}
