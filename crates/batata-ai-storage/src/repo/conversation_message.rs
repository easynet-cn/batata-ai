use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::ConversationMessage;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::ConversationMessageRepository;

use crate::entity::conversation_message;

pub struct SeaOrmConversationMessageRepository {
    db: DatabaseConnection,
}

impl SeaOrmConversationMessageRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl ConversationMessageRepository for SeaOrmConversationMessageRepository {
    async fn create(&self, msg: &ConversationMessage) -> Result<ConversationMessage> {
        let active = conversation_message::ActiveModel {
            id: Set(msg.id.clone()),
            conversation_id: Set(msg.conversation_id.clone()),
            tenant_id: Set(msg.tenant_id.clone()),
            role: Set(msg.role.clone()),
            content: Set(msg.content.clone()),
            model: Set(msg.model.clone()),
            usage: Set(msg.usage.clone()),
            latency_ms: Set(msg.latency_ms),
            metadata: Set(msg.metadata.clone()),
            created_at: Set(msg.created_at),
        };
        let result = conversation_message::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn find_by_id(&self, id: &str) -> Result<Option<ConversationMessage>> {
        conversation_message::Entity::find_by_id(id)
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_conversation(
        &self,
        conversation_id: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<ConversationMessage>> {
        conversation_message::Entity::find()
            .filter(conversation_message::Column::ConversationId.eq(conversation_id))
            .order_by_asc(conversation_message::Column::CreatedAt)
            .offset(page * page_size)
            .limit(page_size)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn count_by_conversation(&self, conversation_id: &str) -> Result<u64> {
        conversation_message::Entity::find()
            .filter(conversation_message::Column::ConversationId.eq(conversation_id))
            .count(&self.db)
            .await
            .map_err(map_db_err)
    }
}
