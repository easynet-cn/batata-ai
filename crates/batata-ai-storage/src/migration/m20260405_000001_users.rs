use sea_orm_migration::{prelude::*, schema::*};

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // users
        manager
            .create_table(
                Table::create()
                    .table(Users::Table)
                    .if_not_exists()
                    .col(string(Users::Id).primary_key())
                    .col(string(Users::TenantId).not_null())
                    .col(string(Users::Username).not_null())
                    .col(string(Users::PasswordHash).not_null())
                    .col(string_null(Users::DisplayName))
                    .col(string_null(Users::Email))
                    .col(boolean(Users::Enabled).default(true).not_null())
                    .col(timestamp_null(Users::LastLoginAt))
                    .col(timestamp(Users::CreatedAt).not_null())
                    .col(timestamp(Users::UpdatedAt).not_null())
                    .col(timestamp_null(Users::DeletedAt))
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_users_tenant")
                            .from(Users::Table, Users::TenantId)
                            .to(Tenants::Table, Tenants::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // unique index: (tenant_id, username)
        manager
            .create_index(
                Index::create()
                    .name("idx_users_tenant_username")
                    .table(Users::Table)
                    .col(Users::TenantId)
                    .col(Users::Username)
                    .unique()
                    .to_owned(),
            )
            .await?;

        // api_keys: add user_id column
        manager
            .alter_table(
                Table::alter()
                    .table(ApiKeys::Table)
                    .add_column(string_null(ApiKeys::UserId))
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Remove user_id from api_keys
        manager
            .alter_table(
                Table::alter()
                    .table(ApiKeys::Table)
                    .drop_column(ApiKeys::UserId)
                    .to_owned(),
            )
            .await?;

        manager
            .drop_table(Table::drop().table(Users::Table).to_owned())
            .await?;

        Ok(())
    }
}

#[derive(DeriveIden)]
enum Tenants {
    Table,
    Id,
}

#[derive(DeriveIden)]
enum Users {
    Table,
    Id,
    TenantId,
    Username,
    PasswordHash,
    DisplayName,
    Email,
    Enabled,
    LastLoginAt,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum ApiKeys {
    Table,
    UserId,
}
