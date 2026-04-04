use async_trait::async_trait;
use std::path::{Path, PathBuf};

use batata_ai_core::domain::StoredObject;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::object_store::ObjectStore;

/// Local filesystem object store.
pub struct LocalFileStore {
    root: PathBuf,
    store_config_id: String,
}

impl LocalFileStore {
    pub fn new(root: impl Into<PathBuf>, store_config_id: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            store_config_id: store_config_id.into(),
        }
    }

    fn full_path(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }

    async fn ensure_parent(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
        }
        Ok(())
    }
}

#[async_trait]
impl ObjectStore for LocalFileStore {
    fn backend(&self) -> &str {
        "local"
    }

    async fn put(
        &self,
        key: &str,
        data: &[u8],
        content_type: &str,
    ) -> Result<StoredObject> {
        let path = self.full_path(key);
        self.ensure_parent(&path).await?;

        tokio::fs::write(&path, data)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))?;

        Ok(StoredObject {
            id: uuid::Uuid::new_v4().to_string(),
            bucket_id: self.store_config_id.clone(),
            key: key.to_string(),
            original_name: None,
            content_type: content_type.to_string(),
            size: data.len() as i64,
            checksum: None,
            metadata: None,
            created_at: chrono::Utc::now().naive_utc(),
            deleted_at: None,
        })
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.full_path(key);
        tokio::fs::read(&path)
            .await
            .map_err(|e| BatataError::Storage(e.to_string()))
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let path = self.full_path(key);
        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.full_path(key).exists())
    }

    async fn presign_url(&self, _key: &str, _expires_secs: u64) -> Result<Option<String>> {
        // Local filesystem does not support pre-signed URLs
        Ok(None)
    }
}
