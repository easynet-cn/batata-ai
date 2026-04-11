use sea_orm_migration::{prelude::*, schema::*};

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // SQLite does not support `ALTER TABLE ... ADD COLUMN <col> UNIQUE`
        // — the uniqueness must be expressed as a separate UNIQUE INDEX.
        // That form is also valid on MySQL and PostgreSQL, so we use it
        // everywhere for portability.

        // 1. Add app_key column (plain nullable, no inline UNIQUE).
        manager
            .alter_table(
                Table::alter()
                    .table(ApiKeys::Table)
                    .add_column(string_null(ApiKeys::AppKey))
                    .to_owned(),
            )
            .await?;

        // 2. Add app_secret_hash column (nullable).
        manager
            .alter_table(
                Table::alter()
                    .table(ApiKeys::Table)
                    .add_column(string_null(ApiKeys::AppSecretHash))
                    .to_owned(),
            )
            .await?;

        // 3. Enforce uniqueness on app_key via a separate index.
        //    NULLs in SQLite/PG/MySQL are treated as distinct, so nullable
        //    rows without a key still coexist.
        manager
            .create_index(
                Index::create()
                    .name("idx_api_keys_app_key_unique")
                    .table(ApiKeys::Table)
                    .col(ApiKeys::AppKey)
                    .unique()
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_index(
                Index::drop()
                    .name("idx_api_keys_app_key_unique")
                    .table(ApiKeys::Table)
                    .to_owned(),
            )
            .await?;

        manager
            .alter_table(
                Table::alter()
                    .table(ApiKeys::Table)
                    .drop_column(ApiKeys::AppKey)
                    .to_owned(),
            )
            .await?;

        manager
            .alter_table(
                Table::alter()
                    .table(ApiKeys::Table)
                    .drop_column(ApiKeys::AppSecretHash)
                    .to_owned(),
            )
            .await?;

        Ok(())
    }
}

#[derive(DeriveIden)]
enum ApiKeys {
    Table,
    AppKey,
    AppSecretHash,
}
