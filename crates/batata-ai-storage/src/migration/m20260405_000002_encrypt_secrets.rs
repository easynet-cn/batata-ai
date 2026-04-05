use sea_orm::Statement;
use sea_orm_migration::prelude::*;

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, AeadCore,
};
use base64::{engine::general_purpose::STANDARD, Engine};

#[derive(DeriveMigrationName)]
pub struct Migration;

const ENC_PREFIX: &str = "enc:";

fn encrypt(cipher: &Aes256Gcm, plaintext: &str) -> Result<String, DbErr> {
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, plaintext.as_bytes())
        .map_err(|e| DbErr::Custom(format!("encrypt failed: {e}")))?;
    Ok(format!(
        "{}{}:{}",
        ENC_PREFIX,
        STANDARD.encode(nonce),
        STANDARD.encode(ciphertext)
    ))
}

fn load_cipher() -> Result<Aes256Gcm, DbErr> {
    let key_b64 = std::env::var("BATATA_MASTER_KEY")
        .map_err(|_| DbErr::Custom("BATATA_MASTER_KEY env not set — required for encrypting existing secrets".into()))?;
    let key_bytes = STANDARD
        .decode(&key_b64)
        .map_err(|e| DbErr::Custom(format!("invalid BATATA_MASTER_KEY base64: {e}")))?;
    Aes256Gcm::new_from_slice(&key_bytes)
        .map_err(|_| DbErr::Custom("BATATA_MASTER_KEY must be exactly 32 bytes".into()))
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let cipher = load_cipher()?;
        let db = manager.get_connection();

        // Encrypt providers.api_key
        let rows = db
            .query_all(Statement::from_string(
                db.get_database_backend(),
                "SELECT id, api_key FROM providers WHERE api_key IS NOT NULL AND api_key != '' AND api_key NOT LIKE 'enc:%'".to_string(),
            ))
            .await?;

        for row in &rows {
            let id: String = row.try_get("", "id")?;
            let api_key: String = row.try_get("", "api_key")?;
            let encrypted = encrypt(&cipher, &api_key)?;
            db.execute(Statement::from_sql_and_values(
                db.get_database_backend(),
                "UPDATE providers SET api_key = $1 WHERE id = $2",
                [encrypted.into(), id.into()],
            ))
            .await?;
        }
        tracing::info!("encrypted {} provider api_key(s)", rows.len());

        // Encrypt object_store_configs.secret_key
        let rows = db
            .query_all(Statement::from_string(
                db.get_database_backend(),
                "SELECT id, secret_key FROM object_store_configs WHERE secret_key IS NOT NULL AND secret_key != '' AND secret_key NOT LIKE 'enc:%'".to_string(),
            ))
            .await?;

        for row in &rows {
            let id: String = row.try_get("", "id")?;
            let secret_key: String = row.try_get("", "secret_key")?;
            let encrypted = encrypt(&cipher, &secret_key)?;
            db.execute(Statement::from_sql_and_values(
                db.get_database_backend(),
                "UPDATE object_store_configs SET secret_key = $1 WHERE id = $2",
                [encrypted.into(), id.into()],
            ))
            .await?;
        }
        tracing::info!("encrypted {} object_store_config secret_key(s)", rows.len());

        Ok(())
    }

    async fn down(&self, _manager: &SchemaManager) -> Result<(), DbErr> {
        // Decryption rollback is not supported — would require the same master key.
        // Legacy plaintext values are still readable by the decrypt function (no "enc:" prefix).
        Err(DbErr::Custom(
            "cannot rollback secret encryption — decrypt manually if needed".into(),
        ))
    }
}
