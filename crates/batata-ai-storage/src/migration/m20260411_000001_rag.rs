use sea_orm_migration::{prelude::*, schema::*};

/// Creates the `kb_chunks` table backing the persistent RAG vector store.
///
/// Scope is intentionally narrow: KB and document metadata tables (plus
/// their CRUD) come in a later migration. `kb_chunks` alone is enough for
/// the `VectorStore` trait contract — `upsert`, `delete_by_doc`, `search` —
/// and keeps the upgrade path simple.
#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(KbChunks::Table)
                    .if_not_exists()
                    .col(string(KbChunks::Id).primary_key())
                    .col(string(KbChunks::KbId).not_null())
                    .col(string(KbChunks::DocId).not_null())
                    .col(integer(KbChunks::Ord).not_null())
                    .col(text(KbChunks::Text).not_null())
                    // Embedding is serialised as little-endian f32 bytes.
                    // dim = embedding_bytes.len() / 4. Store raw to keep
                    // the schema backend-agnostic (sqlite / mysql / pg).
                    .col(blob(KbChunks::Embedding).not_null())
                    .col(json_null(KbChunks::Metadata))
                    .col(timestamp(KbChunks::CreatedAt).not_null())
                    .to_owned(),
            )
            .await?;

        // Index for the search path — scans all chunks within a KB.
        manager
            .create_index(
                Index::create()
                    .name("idx_kb_chunks_kb_id")
                    .table(KbChunks::Table)
                    .col(KbChunks::KbId)
                    .to_owned(),
            )
            .await?;

        // Composite index for `delete_by_doc(kb_id, doc_id)`.
        manager
            .create_index(
                Index::create()
                    .name("idx_kb_chunks_kb_doc")
                    .table(KbChunks::Table)
                    .col(KbChunks::KbId)
                    .col(KbChunks::DocId)
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(KbChunks::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum KbChunks {
    Table,
    Id,
    KbId,
    DocId,
    Ord,
    Text,
    Embedding,
    Metadata,
    CreatedAt,
}
