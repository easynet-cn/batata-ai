use sea_orm_migration::{prelude::*, schema::*};

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // providers
        manager
            .create_table(
                Table::create()
                    .table(Providers::Table)
                    .if_not_exists()
                    .col(string(Providers::Id).primary_key())
                    .col(string(Providers::Name).unique_key().not_null())
                    .col(string(Providers::ProviderType).not_null())
                    .col(string_null(Providers::ApiKey))
                    .col(string_null(Providers::BaseUrl))
                    .col(json_null(Providers::Config))
                    .col(boolean(Providers::Enabled).default(true).not_null())
                    .col(timestamp(Providers::CreatedAt).not_null())
                    .col(timestamp(Providers::UpdatedAt).not_null())
                    .col(timestamp_null(Providers::DeletedAt))
                    .to_owned(),
            )
            .await?;

        // models
        manager
            .create_table(
                Table::create()
                    .table(Models::Table)
                    .if_not_exists()
                    .col(string(Models::Id).primary_key())
                    .col(string(Models::Name).unique_key().not_null())
                    .col(string(Models::ModelType).not_null())
                    .col(integer_null(Models::ContextLength))
                    .col(text_null(Models::Description))
                    .col(json_null(Models::Metadata))
                    .col(boolean(Models::Enabled).default(true).not_null())
                    .col(timestamp(Models::CreatedAt).not_null())
                    .col(timestamp(Models::UpdatedAt).not_null())
                    .col(timestamp_null(Models::DeletedAt))
                    .to_owned(),
            )
            .await?;

        // model_providers (many-to-many)
        manager
            .create_table(
                Table::create()
                    .table(ModelProviders::Table)
                    .if_not_exists()
                    .col(string(ModelProviders::ModelId).not_null())
                    .col(string(ModelProviders::ProviderId).not_null())
                    .col(string(ModelProviders::ModelIdentifier).not_null())
                    .col(
                        boolean(ModelProviders::IsDefault)
                            .default(false)
                            .not_null(),
                    )
                    .col(integer(ModelProviders::Priority).default(0).not_null())
                    .col(
                        boolean(ModelProviders::Enabled)
                            .default(true)
                            .not_null(),
                    )
                    .col(timestamp(ModelProviders::CreatedAt).not_null())
                    .col(timestamp_null(ModelProviders::DeletedAt))
                    .primary_key(
                        Index::create()
                            .col(ModelProviders::ModelId)
                            .col(ModelProviders::ProviderId),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_model_providers_model")
                            .from(ModelProviders::Table, ModelProviders::ModelId)
                            .to(Models::Table, Models::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_model_providers_provider")
                            .from(ModelProviders::Table, ModelProviders::ProviderId)
                            .to(Providers::Table, Providers::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // prompts
        manager
            .create_table(
                Table::create()
                    .table(Prompts::Table)
                    .if_not_exists()
                    .col(string(Prompts::Id).primary_key())
                    .col(string(Prompts::Name).unique_key().not_null())
                    .col(text(Prompts::Description).not_null())
                    .col(text(Prompts::Template).not_null())
                    .col(json(Prompts::Variables).not_null())
                    .col(string_null(Prompts::Category))
                    .col(integer(Prompts::Version).default(1).not_null())
                    .col(timestamp(Prompts::CreatedAt).not_null())
                    .col(timestamp(Prompts::UpdatedAt).not_null())
                    .col(timestamp_null(Prompts::DeletedAt))
                    .to_owned(),
            )
            .await?;

        // skills
        manager
            .create_table(
                Table::create()
                    .table(Skills::Table)
                    .if_not_exists()
                    .col(string(Skills::Id).primary_key())
                    .col(string(Skills::Name).unique_key().not_null())
                    .col(text(Skills::Description).not_null())
                    .col(json(Skills::ParametersSchema).not_null())
                    .col(string(Skills::SkillType).not_null())
                    .col(json_null(Skills::Config))
                    .col(boolean(Skills::Enabled).default(true).not_null())
                    .col(integer(Skills::Version).default(1).not_null())
                    .col(timestamp(Skills::CreatedAt).not_null())
                    .col(timestamp(Skills::UpdatedAt).not_null())
                    .col(timestamp_null(Skills::DeletedAt))
                    .to_owned(),
            )
            .await?;

        // prompt_versions (history snapshots)
        manager
            .create_table(
                Table::create()
                    .table(PromptVersions::Table)
                    .if_not_exists()
                    .col(string(PromptVersions::Id).primary_key())
                    .col(string(PromptVersions::PromptId).not_null())
                    .col(integer(PromptVersions::Version).not_null())
                    .col(text(PromptVersions::Template).not_null())
                    .col(json(PromptVersions::Variables).not_null())
                    .col(text(PromptVersions::Description).not_null())
                    .col(text_null(PromptVersions::ChangeMessage))
                    .col(string_null(PromptVersions::ChangedBy))
                    .col(timestamp(PromptVersions::CreatedAt).not_null())
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_prompt_versions_prompt")
                            .from(PromptVersions::Table, PromptVersions::PromptId)
                            .to(Prompts::Table, Prompts::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_prompt_versions_prompt_version")
                    .table(PromptVersions::Table)
                    .col(PromptVersions::PromptId)
                    .col(PromptVersions::Version)
                    .unique()
                    .to_owned(),
            )
            .await?;

        // skill_versions (history snapshots)
        manager
            .create_table(
                Table::create()
                    .table(SkillVersions::Table)
                    .if_not_exists()
                    .col(string(SkillVersions::Id).primary_key())
                    .col(string(SkillVersions::SkillId).not_null())
                    .col(integer(SkillVersions::Version).not_null())
                    .col(text(SkillVersions::Description).not_null())
                    .col(json(SkillVersions::ParametersSchema).not_null())
                    .col(json_null(SkillVersions::Config))
                    .col(text_null(SkillVersions::ChangeMessage))
                    .col(string_null(SkillVersions::ChangedBy))
                    .col(timestamp(SkillVersions::CreatedAt).not_null())
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_skill_versions_skill")
                            .from(SkillVersions::Table, SkillVersions::SkillId)
                            .to(Skills::Table, Skills::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_skill_versions_skill_version")
                    .table(SkillVersions::Table)
                    .col(SkillVersions::SkillId)
                    .col(SkillVersions::Version)
                    .unique()
                    .to_owned(),
            )
            .await?;

        // routing_policies
        manager
            .create_table(
                Table::create()
                    .table(RoutingPolicies::Table)
                    .if_not_exists()
                    .col(string(RoutingPolicies::Id).primary_key())
                    .col(string(RoutingPolicies::Name).unique_key().not_null())
                    .col(string(RoutingPolicies::PolicyType).not_null())
                    .col(json(RoutingPolicies::Config).not_null())
                    .col(boolean(RoutingPolicies::Enabled).default(true).not_null())
                    .col(integer(RoutingPolicies::Priority).default(0).not_null())
                    .col(timestamp(RoutingPolicies::CreatedAt).not_null())
                    .col(timestamp(RoutingPolicies::UpdatedAt).not_null())
                    .col(timestamp_null(RoutingPolicies::DeletedAt))
                    .to_owned(),
            )
            .await?;

        // model_costs
        manager
            .create_table(
                Table::create()
                    .table(ModelCosts::Table)
                    .if_not_exists()
                    .col(string(ModelCosts::Id).primary_key())
                    .col(string(ModelCosts::ModelId).not_null())
                    .col(string(ModelCosts::ProviderId).not_null())
                    .col(double(ModelCosts::InputCostPer1k).not_null())
                    .col(double(ModelCosts::OutputCostPer1k).not_null())
                    .col(string(ModelCosts::Currency).not_null().default("USD"))
                    .col(timestamp(ModelCosts::EffectiveFrom).not_null())
                    .col(timestamp(ModelCosts::CreatedAt).not_null())
                    .col(timestamp_null(ModelCosts::DeletedAt))
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_model_costs_model")
                            .from(ModelCosts::Table, ModelCosts::ModelId)
                            .to(Models::Table, Models::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_model_costs_provider")
                            .from(ModelCosts::Table, ModelCosts::ProviderId)
                            .to(Providers::Table, Providers::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_model_costs_model_provider")
                    .table(ModelCosts::Table)
                    .col(ModelCosts::ModelId)
                    .col(ModelCosts::ProviderId)
                    .to_owned(),
            )
            .await?;

        // request_logs
        manager
            .create_table(
                Table::create()
                    .table(RequestLogs::Table)
                    .if_not_exists()
                    .col(string(RequestLogs::Id).primary_key())
                    .col(string(RequestLogs::ProviderId).not_null())
                    .col(string(RequestLogs::ProviderName).not_null())
                    .col(string(RequestLogs::ModelIdentifier).not_null())
                    .col(string(RequestLogs::RoutingPolicy).not_null())
                    .col(string(RequestLogs::Status).not_null())
                    .col(big_integer(RequestLogs::LatencyMs).not_null())
                    .col(integer_null(RequestLogs::PromptTokens))
                    .col(integer_null(RequestLogs::CompletionTokens))
                    .col(integer_null(RequestLogs::TotalTokens))
                    .col(double_null(RequestLogs::EstimatedCost))
                    .col(text_null(RequestLogs::ErrorMessage))
                    .col(json_null(RequestLogs::Metadata))
                    .col(timestamp(RequestLogs::CreatedAt).not_null())
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_request_logs_created_at")
                    .table(RequestLogs::Table)
                    .col(RequestLogs::CreatedAt)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_request_logs_provider")
                    .table(RequestLogs::Table)
                    .col(RequestLogs::ProviderId)
                    .to_owned(),
            )
            .await?;

        // object_store_configs (凭证层)
        manager
            .create_table(
                Table::create()
                    .table(ObjectStoreConfigs::Table)
                    .if_not_exists()
                    .col(string(ObjectStoreConfigs::Id).primary_key())
                    .col(string(ObjectStoreConfigs::Name).unique_key().not_null())
                    .col(string(ObjectStoreConfigs::Backend).not_null())
                    .col(string_null(ObjectStoreConfigs::Endpoint))
                    .col(string_null(ObjectStoreConfigs::Region))
                    .col(string_null(ObjectStoreConfigs::AccessKey))
                    .col(string_null(ObjectStoreConfigs::SecretKey))
                    .col(boolean(ObjectStoreConfigs::Enabled).default(true).not_null())
                    .col(json_null(ObjectStoreConfigs::Config))
                    .col(timestamp(ObjectStoreConfigs::CreatedAt).not_null())
                    .col(timestamp(ObjectStoreConfigs::UpdatedAt).not_null())
                    .col(timestamp_null(ObjectStoreConfigs::DeletedAt))
                    .to_owned(),
            )
            .await?;

        // object_store_buckets (桶层)
        manager
            .create_table(
                Table::create()
                    .table(ObjectStoreBuckets::Table)
                    .if_not_exists()
                    .col(string(ObjectStoreBuckets::Id).primary_key())
                    .col(string(ObjectStoreBuckets::ConfigId).not_null())
                    .col(string(ObjectStoreBuckets::Name).not_null())
                    .col(string(ObjectStoreBuckets::Bucket).not_null())
                    .col(string_null(ObjectStoreBuckets::RootPath))
                    .col(
                        string(ObjectStoreBuckets::AccessPolicy)
                            .not_null()
                            .default("private"),
                    )
                    .col(string_null(ObjectStoreBuckets::CustomDomain))
                    .col(
                        boolean(ObjectStoreBuckets::IsDefault)
                            .default(false)
                            .not_null(),
                    )
                    .col(
                        boolean(ObjectStoreBuckets::Enabled)
                            .default(true)
                            .not_null(),
                    )
                    .col(json_null(ObjectStoreBuckets::Config))
                    .col(timestamp(ObjectStoreBuckets::CreatedAt).not_null())
                    .col(timestamp(ObjectStoreBuckets::UpdatedAt).not_null())
                    .col(timestamp_null(ObjectStoreBuckets::DeletedAt))
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_object_store_buckets_config")
                            .from(ObjectStoreBuckets::Table, ObjectStoreBuckets::ConfigId)
                            .to(ObjectStoreConfigs::Table, ObjectStoreConfigs::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        // stored_objects
        manager
            .create_table(
                Table::create()
                    .table(StoredObjects::Table)
                    .if_not_exists()
                    .col(string(StoredObjects::Id).primary_key())
                    .col(string(StoredObjects::BucketId).not_null())
                    .col(string(StoredObjects::Key).unique_key().not_null())
                    .col(string_null(StoredObjects::OriginalName))
                    .col(string(StoredObjects::ContentType).not_null())
                    .col(big_integer(StoredObjects::Size).not_null())
                    .col(string_null(StoredObjects::Checksum))
                    .col(json_null(StoredObjects::Metadata))
                    .col(timestamp(StoredObjects::CreatedAt).not_null())
                    .col(timestamp_null(StoredObjects::DeletedAt))
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_stored_objects_bucket")
                            .from(StoredObjects::Table, StoredObjects::BucketId)
                            .to(ObjectStoreBuckets::Table, ObjectStoreBuckets::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(StoredObjects::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(ObjectStoreBuckets::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(ObjectStoreConfigs::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(RequestLogs::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(ModelCosts::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(RoutingPolicies::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(SkillVersions::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(PromptVersions::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(ModelProviders::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Skills::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Prompts::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Models::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Providers::Table).to_owned())
            .await?;
        Ok(())
    }
}

#[derive(DeriveIden)]
enum Providers {
    Table,
    Id,
    Name,
    ProviderType,
    ApiKey,
    BaseUrl,
    Config,
    Enabled,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum Models {
    Table,
    Id,
    Name,
    ModelType,
    ContextLength,
    Description,
    Metadata,
    Enabled,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum ModelProviders {
    Table,
    ModelId,
    ProviderId,
    ModelIdentifier,
    IsDefault,
    Priority,
    Enabled,
    CreatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum Prompts {
    Table,
    Id,
    Name,
    Description,
    Template,
    Variables,
    Category,
    Version,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum Skills {
    Table,
    Id,
    Name,
    Description,
    ParametersSchema,
    SkillType,
    Config,
    Enabled,
    Version,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum PromptVersions {
    Table,
    Id,
    PromptId,
    Version,
    Template,
    Variables,
    Description,
    ChangeMessage,
    ChangedBy,
    CreatedAt,
}

#[derive(DeriveIden)]
enum SkillVersions {
    Table,
    Id,
    SkillId,
    Version,
    Description,
    ParametersSchema,
    Config,
    ChangeMessage,
    ChangedBy,
    CreatedAt,
}

#[derive(DeriveIden)]
enum RoutingPolicies {
    Table,
    Id,
    Name,
    PolicyType,
    Config,
    Enabled,
    Priority,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum ModelCosts {
    Table,
    Id,
    ModelId,
    ProviderId,
    InputCostPer1k,
    OutputCostPer1k,
    Currency,
    EffectiveFrom,
    CreatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum RequestLogs {
    Table,
    Id,
    ProviderId,
    ProviderName,
    ModelIdentifier,
    RoutingPolicy,
    Status,
    LatencyMs,
    PromptTokens,
    CompletionTokens,
    TotalTokens,
    EstimatedCost,
    ErrorMessage,
    Metadata,
    CreatedAt,
}

#[derive(DeriveIden)]
enum ObjectStoreConfigs {
    Table,
    Id,
    Name,
    Backend,
    Endpoint,
    Region,
    AccessKey,
    SecretKey,
    Enabled,
    Config,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum ObjectStoreBuckets {
    Table,
    Id,
    ConfigId,
    Name,
    Bucket,
    RootPath,
    AccessPolicy,
    CustomDomain,
    IsDefault,
    Enabled,
    Config,
    CreatedAt,
    UpdatedAt,
    DeletedAt,
}

#[derive(DeriveIden)]
enum StoredObjects {
    Table,
    Id,
    BucketId,
    Key,
    OriginalName,
    ContentType,
    Size,
    Checksum,
    Metadata,
    CreatedAt,
    DeletedAt,
}
