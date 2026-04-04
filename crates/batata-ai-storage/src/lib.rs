pub mod convert;
pub mod entity;
pub mod migration;
pub mod repo;

use sea_orm::{ConnectOptions, Database, DatabaseConnection, DbErr};
use sea_orm_migration::MigratorTrait;
use tracing;

pub use repo::{
    SeaOrmModelCostRepository, SeaOrmModelRepository, SeaOrmObjectStoreBucketRepository,
    SeaOrmObjectStoreConfigRepository, SeaOrmPromptRepository, SeaOrmProviderRepository,
    SeaOrmRequestLogRepository, SeaOrmRoutingPolicyRepository, SeaOrmSkillRepository,
    SeaOrmStoredObjectRepository,
};

/// Connect to database and run pending migrations.
///
/// Supported URL schemes:
/// - `sqlite://path/to/db.sqlite?mode=rwc` (SQLite)
/// - `mysql://user:pass@host/db` (MySQL)
/// - `postgres://user:pass@host/db` (PostgreSQL)
pub async fn connect_and_migrate(database_url: &str) -> std::result::Result<DatabaseConnection, DbErr> {
    let mut opt = ConnectOptions::new(database_url);
    opt.max_connections(100)
        .min_connections(5)
        .sqlx_logging(false);

    tracing::info!("connecting to database: {}", database_url);
    let db = Database::connect(opt).await?;

    tracing::info!("running migrations...");
    migration::Migrator::up(&db, None).await?;

    tracing::info!("database ready");
    Ok(db)
}
