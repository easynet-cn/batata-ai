use sea_orm_migration::{prelude::*, schema::*};

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Add app_key column (unique, nullable for backward compat)
        manager
            .alter_table(
                Table::alter()
                    .table(ApiKeys::Table)
                    .add_column(string_null(ApiKeys::AppKey).unique_key())
                    .to_owned(),
            )
            .await?;

        // Add app_secret_hash column (nullable)
        manager
            .alter_table(
                Table::alter()
                    .table(ApiKeys::Table)
                    .add_column(string_null(ApiKeys::AppSecretHash))
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
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
