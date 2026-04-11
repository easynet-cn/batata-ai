use sea_orm_migration::{prelude::*, schema::*};

/// Knowledge base + document metadata tables.
///
/// Complements the `kb_chunks` table from m20260411_000001_rag so that
/// callers can list KBs and the documents inside them, soft-delete a
/// document (chunks still get hard-deleted via repo logic), and enforce
/// tenant scoping.
#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // knowledge_bases
        manager
            .create_table(
                Table::create()
                    .table(KnowledgeBases::Table)
                    .if_not_exists()
                    .col(string(KnowledgeBases::Id).primary_key())
                    .col(string_null(KnowledgeBases::TenantId))
                    .col(string(KnowledgeBases::Name).not_null())
                    .col(text_null(KnowledgeBases::Description))
                    .col(string(KnowledgeBases::Embedder).not_null())
                    .col(integer(KnowledgeBases::Dim).not_null())
                    .col(integer(KnowledgeBases::ChunkWindow).not_null())
                    .col(integer(KnowledgeBases::ChunkOverlap).not_null())
                    .col(json_null(KnowledgeBases::Metadata))
                    .col(timestamp(KnowledgeBases::CreatedAt).not_null())
                    .col(timestamp(KnowledgeBases::UpdatedAt).not_null())
                    .col(timestamp_null(KnowledgeBases::DeletedAt))
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_knowledge_bases_tenant")
                    .table(KnowledgeBases::Table)
                    .col(KnowledgeBases::TenantId)
                    .to_owned(),
            )
            .await?;

        // kb_documents
        manager
            .create_table(
                Table::create()
                    .table(KbDocuments::Table)
                    .if_not_exists()
                    .col(string(KbDocuments::Id).primary_key())
                    .col(string(KbDocuments::KbId).not_null())
                    .col(string_null(KbDocuments::TenantId))
                    .col(string(KbDocuments::SourceUri).not_null())
                    .col(string_null(KbDocuments::Title))
                    .col(string_null(KbDocuments::Mime))
                    .col(string(KbDocuments::Status).not_null())
                    .col(text_null(KbDocuments::Error))
                    .col(integer(KbDocuments::ChunkCount).not_null())
                    .col(json_null(KbDocuments::Metadata))
                    .col(timestamp(KbDocuments::CreatedAt).not_null())
                    .col(timestamp(KbDocuments::UpdatedAt).not_null())
                    .col(timestamp_null(KbDocuments::DeletedAt))
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_kb_documents_kb")
                            .from(KbDocuments::Table, KbDocuments::KbId)
                            .to(KnowledgeBases::Table, KnowledgeBases::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_kb_documents_kb")
                    .table(KbDocuments::Table)
                    .col(KbDocuments::KbId)
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(KbDocuments::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(KnowledgeBases::Table).to_owned())
            .await?;
        Ok(())
    }
}

#[derive(DeriveIden)]
enum KnowledgeBases {
    Table,
    Id,
    TenantId,
    Name,
    Description,
    Embedder,
    Dim,
    ChunkWindow,
    ChunkOverlap,
    Metadata,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum KbDocuments {
    Table,
    Id,
    KbId,
    TenantId,
    SourceUri,
    Title,
    Mime,
    Status,
    Error,
    ChunkCount,
    Metadata,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}
