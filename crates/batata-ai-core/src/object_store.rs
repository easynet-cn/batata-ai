use async_trait::async_trait;

use crate::domain::StoredObject;
use crate::error::Result;

/// Runtime trait for object storage backends.
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Backend name (e.g., "local", "s3", "oss", "minio").
    fn backend(&self) -> &str;

    /// Upload data, return stored object metadata.
    async fn put(
        &self,
        key: &str,
        data: &[u8],
        content_type: &str,
    ) -> Result<StoredObject>;

    /// Download object data by key.
    async fn get(&self, key: &str) -> Result<Vec<u8>>;

    /// Delete object by key.
    async fn delete(&self, key: &str) -> Result<()>;

    /// Check if an object exists.
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Generate a pre-signed URL for temporary access.
    /// Returns None if the backend does not support pre-signed URLs.
    async fn presign_url(&self, key: &str, expires_secs: u64) -> Result<Option<String>>;
}
