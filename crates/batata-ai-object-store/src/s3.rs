//! S3-compatible object store (AWS S3, MinIO, etc.)
//!
//! Requires the `s3` feature flag.

#[cfg(feature = "s3")]
mod inner {
    use async_trait::async_trait;
    use s3::bucket::Bucket;
    use s3::creds::Credentials;
    use s3::Region;

    use batata_ai_core::domain::StoredObject;
    use batata_ai_core::error::{BatataError, Result};
    use batata_ai_core::object_store::ObjectStore;

    pub struct S3Store {
        bucket: Box<Bucket>,
        bucket_id: String,
    }

    impl S3Store {
        pub fn new(
            bucket_name: &str,
            region: &str,
            endpoint: Option<&str>,
            access_key: &str,
            secret_key: &str,
            bucket_id: impl Into<String>,
        ) -> Result<Self> {
            let region = match endpoint {
                Some(ep) => Region::Custom {
                    region: region.to_string(),
                    endpoint: ep.to_string(),
                },
                None => region
                    .parse()
                    .map_err(|e: s3::error::S3Error| BatataError::Storage(e.to_string()))?,
            };

            let credentials = Credentials::new(
                Some(access_key),
                Some(secret_key),
                None,
                None,
                None,
            )
            .map_err(|e| BatataError::Storage(e.to_string()))?;

            let bucket = Bucket::new(bucket_name, region, credentials)
                .map_err(|e| BatataError::Storage(e.to_string()))?;

            Ok(Self {
                bucket,
                bucket_id: bucket_id.into(),
            })
        }
    }

    #[async_trait]
    impl ObjectStore for S3Store {
        fn backend(&self) -> &str {
            "s3"
        }

        async fn put(
            &self,
            key: &str,
            data: &[u8],
            content_type: &str,
        ) -> Result<StoredObject> {
            self.bucket
                .put_object_with_content_type(key, data, content_type)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;

            Ok(StoredObject {
                id: uuid::Uuid::new_v4().to_string(),
                bucket_id: self.bucket_id.clone(),
                key: key.to_string(),
                original_name: None,
                content_type: content_type.to_string(),
                size: data.len() as i64,
                checksum: None,
                metadata: None,
                created_at: chrono::Utc::now().naive_utc(),
            })
        }

        async fn get(&self, key: &str) -> Result<Vec<u8>> {
            let response = self
                .bucket
                .get_object(key)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
            Ok(response.to_vec())
        }

        async fn delete(&self, key: &str) -> Result<()> {
            self.bucket
                .delete_object(key)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
            Ok(())
        }

        async fn exists(&self, key: &str) -> Result<bool> {
            match self.bucket.head_object(key).await {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        }

        async fn presign_url(&self, key: &str, expires_secs: u64) -> Result<Option<String>> {
            let url = self
                .bucket
                .presign_get(key, expires_secs as u32, None)
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
            Ok(Some(url))
        }
    }
}

#[cfg(feature = "s3")]
pub use inner::S3Store;
