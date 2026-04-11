pub use sea_orm_migration::prelude::*;

mod m20260404_000001_init;
mod m20260405_000001_users;
mod m20260405_000002_encrypt_secrets;
mod m20260405_000003_app_key;
mod m20260411_000001_rag;
mod m20260411_000002_rag_meta;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20260404_000001_init::Migration),
            Box::new(m20260405_000001_users::Migration),
            Box::new(m20260405_000002_encrypt_secrets::Migration),
            Box::new(m20260405_000003_app_key::Migration),
            Box::new(m20260411_000001_rag::Migration),
            Box::new(m20260411_000002_rag_meta::Migration),
        ]
    }
}

