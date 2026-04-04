use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ObjectStoreBackend
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectStoreBackend {
    Local,
    S3,
    Oss,
    Minio,
}

impl std::fmt::Display for ObjectStoreBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::S3 => write!(f, "s3"),
            Self::Oss => write!(f, "oss"),
            Self::Minio => write!(f, "minio"),
        }
    }
}

impl std::str::FromStr for ObjectStoreBackend {
    type Err = crate::error::BatataError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "local" => Ok(Self::Local),
            "s3" => Ok(Self::S3),
            "oss" => Ok(Self::Oss),
            "minio" => Ok(Self::Minio),
            other => Err(crate::error::BatataError::Config(format!(
                "unknown object store backend: {other}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// BucketAccessPolicy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BucketAccessPolicy {
    Private,
    PublicRead,
    PublicReadWrite,
}

impl std::fmt::Display for BucketAccessPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Private => write!(f, "private"),
            Self::PublicRead => write!(f, "public_read"),
            Self::PublicReadWrite => write!(f, "public_read_write"),
        }
    }
}

impl std::str::FromStr for BucketAccessPolicy {
    type Err = crate::error::BatataError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "private" => Ok(Self::Private),
            "public_read" => Ok(Self::PublicRead),
            "public_read_write" => Ok(Self::PublicReadWrite),
            other => Err(crate::error::BatataError::Config(format!(
                "unknown bucket access policy: {other}"
            ))),
        }
    }
}

impl Default for BucketAccessPolicy {
    fn default() -> Self {
        Self::Private
    }
}

// ---------------------------------------------------------------------------
// ObjectStoreConfig — 凭证层（一套凭证对应一个云账号）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStoreConfig {
    pub id: String,
    pub name: String,
    pub backend: ObjectStoreBackend,
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub enabled: bool,
    pub config: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

// ---------------------------------------------------------------------------
// ObjectStoreBucket — 桶层（一套凭证下可有多个桶）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStoreBucket {
    pub id: String,
    pub config_id: String,
    pub name: String,
    pub bucket: String,
    pub root_path: Option<String>,
    pub access_policy: BucketAccessPolicy,
    pub custom_domain: Option<String>,
    pub is_default: bool,
    pub enabled: bool,
    pub config: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

// ---------------------------------------------------------------------------
// StoredObject — 存储文件元数据
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredObject {
    pub id: String,
    pub bucket_id: String,
    pub key: String,
    pub original_name: Option<String>,
    pub content_type: String,
    pub size: i64,
    pub checksum: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}
