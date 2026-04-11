use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::rag::{KbDocument, KbDocumentRepository};

use crate::entity::kb_document;

pub struct SeaOrmKbDocumentRepository {
    db: DatabaseConnection,
}

impl SeaOrmKbDocumentRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

fn model_to_domain(m: kb_document::Model) -> KbDocument {
    KbDocument {
        id: m.id,
        kb_id: m.kb_id,
        tenant_id: m.tenant_id,
        source_uri: m.source_uri,
        title: m.title,
        mime: m.mime,
        status: m.status,
        error: m.error,
        chunk_count: m.chunk_count,
        metadata: m.metadata,
        created_at: m.created_at,
        updated_at: m.updated_at,
        deleted_at: m.deleted_at,
    }
}

#[async_trait]
impl KbDocumentRepository for SeaOrmKbDocumentRepository {
    async fn create(&self, entity: &KbDocument) -> Result<KbDocument> {
        let active = kb_document::ActiveModel {
            id: Set(entity.id.clone()),
            kb_id: Set(entity.kb_id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            source_uri: Set(entity.source_uri.clone()),
            title: Set(entity.title.clone()),
            mime: Set(entity.mime.clone()),
            status: Set(entity.status.clone()),
            error: Set(entity.error.clone()),
            chunk_count: Set(entity.chunk_count),
            metadata: Set(entity.metadata.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let row = kb_document::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(model_to_domain(row))
    }

    async fn update(&self, entity: &KbDocument) -> Result<KbDocument> {
        let active = kb_document::ActiveModel {
            id: Set(entity.id.clone()),
            kb_id: Set(entity.kb_id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            source_uri: Set(entity.source_uri.clone()),
            title: Set(entity.title.clone()),
            mime: Set(entity.mime.clone()),
            status: Set(entity.status.clone()),
            error: Set(entity.error.clone()),
            chunk_count: Set(entity.chunk_count),
            metadata: Set(entity.metadata.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let row = kb_document::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(model_to_domain(row))
    }

    async fn find_by_id(&self, id: &str) -> Result<Option<KbDocument>> {
        kb_document::Entity::find_by_id(id)
            .filter(kb_document::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(model_to_domain))
            .map_err(map_db_err)
    }

    async fn find_by_kb(&self, kb_id: &str) -> Result<Vec<KbDocument>> {
        kb_document::Entity::find()
            .filter(kb_document::Column::KbId.eq(kb_id))
            .filter(kb_document::Column::DeletedAt.is_null())
            .order_by_asc(kb_document::Column::CreatedAt)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(model_to_domain).collect())
            .map_err(map_db_err)
    }

    async fn soft_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<kb_document::Entity>(&self.db, id).await
    }

    async fn soft_delete_by_kb(&self, kb_id: &str) -> Result<u64> {
        let now = chrono::Utc::now().naive_utc();
        let result = kb_document::Entity::update_many()
            .col_expr(
                kb_document::Column::DeletedAt,
                sea_orm::sea_query::Expr::value(sea_orm::Value::ChronoDateTime(Some(Box::new(
                    now,
                )))),
            )
            .filter(kb_document::Column::KbId.eq(kb_id))
            .filter(kb_document::Column::DeletedAt.is_null())
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.rows_affected)
    }
}
