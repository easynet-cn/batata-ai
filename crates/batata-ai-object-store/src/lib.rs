#[cfg(feature = "local")]
mod local;
mod oss;
mod s3;

#[cfg(feature = "local")]
pub use local::LocalFileStore;

#[cfg(feature = "oss")]
pub use oss::OssStore;

#[cfg(feature = "s3")]
pub use s3::S3Store;
