//! Alibaba Cloud OSS object store.
//!
//! Requires the `oss` feature flag.
//! Uses manual HMAC-SHA1 signing (OSS v1 signature).

#[cfg(feature = "oss")]
mod inner {
    use async_trait::async_trait;

    use batata_ai_core::domain::StoredObject;
    use batata_ai_core::error::{BatataError, Result};
    use batata_ai_core::object_store::ObjectStore;

    pub struct OssStore {
        endpoint: String,
        bucket_name: String,
        access_key: String,
        secret_key: String,
        bucket_id: String,
        client: reqwest::Client,
    }

    impl OssStore {
        pub fn new(
            endpoint: &str,
            bucket_name: &str,
            access_key: &str,
            secret_key: &str,
            bucket_id: impl Into<String>,
        ) -> Self {
            Self {
                endpoint: endpoint.trim_end_matches('/').to_string(),
                bucket_name: bucket_name.to_string(),
                access_key: access_key.to_string(),
                secret_key: secret_key.to_string(),
                bucket_id: bucket_id.into(),
                client: reqwest::Client::new(),
            }
        }

        fn object_url(&self, key: &str) -> String {
            format!(
                "https://{}.{}/{}",
                self.bucket_name, self.endpoint, key
            )
        }

        fn sign(&self, verb: &str, key: &str, content_type: &str, date: &str) -> String {
            use hmac::{Hmac, Mac};
            use sha2::Sha256;

            let string_to_sign = format!(
                "{}\n\n{}\n{}\n/{}/{}",
                verb, content_type, date, self.bucket_name, key
            );

            let mut mac =
                Hmac::<Sha256>::new_from_slice(self.secret_key.as_bytes()).expect("HMAC key");
            mac.update(string_to_sign.as_bytes());
            let result = mac.finalize();
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.encode(result.into_bytes())
        }

        fn auth_header(&self, signature: &str) -> String {
            format!("OSS {}:{}", self.access_key, signature)
        }

        fn rfc2822_now() -> String {
            chrono::Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()
        }
    }

    #[async_trait]
    impl ObjectStore for OssStore {
        fn backend(&self) -> &str {
            "oss"
        }

        async fn put(
            &self,
            key: &str,
            data: &[u8],
            content_type: &str,
        ) -> Result<StoredObject> {
            let date = Self::rfc2822_now();
            let signature = self.sign("PUT", key, content_type, &date);

            self.client
                .put(&self.object_url(key))
                .header("Date", &date)
                .header("Content-Type", content_type)
                .header("Authorization", self.auth_header(&signature))
                .body(data.to_vec())
                .send()
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?
                .error_for_status()
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
            let date = Self::rfc2822_now();
            let signature = self.sign("GET", key, "", &date);

            let resp = self
                .client
                .get(&self.object_url(key))
                .header("Date", &date)
                .header("Authorization", self.auth_header(&signature))
                .send()
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?
                .error_for_status()
                .map_err(|e| BatataError::Storage(e.to_string()))?;

            resp.bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| BatataError::Storage(e.to_string()))
        }

        async fn delete(&self, key: &str) -> Result<()> {
            let date = Self::rfc2822_now();
            let signature = self.sign("DELETE", key, "", &date);

            self.client
                .delete(&self.object_url(key))
                .header("Date", &date)
                .header("Authorization", self.auth_header(&signature))
                .send()
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
            Ok(())
        }

        async fn exists(&self, key: &str) -> Result<bool> {
            let date = Self::rfc2822_now();
            let signature = self.sign("HEAD", key, "", &date);

            let resp = self
                .client
                .head(&self.object_url(key))
                .header("Date", &date)
                .header("Authorization", self.auth_header(&signature))
                .send()
                .await
                .map_err(|e| BatataError::Storage(e.to_string()))?;
            Ok(resp.status().is_success())
        }

        async fn presign_url(&self, key: &str, expires_secs: u64) -> Result<Option<String>> {
            let expires = chrono::Utc::now().timestamp() + expires_secs as i64;
            let string_to_sign = format!(
                "GET\n\n\n{}\n/{}/{}",
                expires, self.bucket_name, key
            );

            use hmac::{Hmac, Mac};
            use sha2::Sha256;
            let mut mac =
                Hmac::<Sha256>::new_from_slice(self.secret_key.as_bytes()).expect("HMAC key");
            mac.update(string_to_sign.as_bytes());
            let result = mac.finalize();
            use base64::Engine;
            let signature =
                base64::engine::general_purpose::STANDARD.encode(result.into_bytes());

            let encoded_sig =
                urlencoding::encode(&signature);
            let url = format!(
                "{}?OSSAccessKeyId={}&Expires={}&Signature={}",
                self.object_url(key),
                self.access_key,
                expires,
                encoded_sig
            );
            Ok(Some(url))
        }
    }
}

#[cfg(feature = "oss")]
pub use inner::OssStore;
