use std::sync::Arc;

use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation};
use serde::{Deserialize, Serialize};

/// JWT claims embedded in every access token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject — user ID.
    pub sub: String,
    /// Tenant ID.
    pub tenant_id: String,
    /// Username (convenience, not authoritative).
    pub username: String,
    /// JSON-encoded scopes.
    #[serde(default)]
    pub scopes: serde_json::Value,
    /// Issued at (seconds since UNIX epoch).
    pub iat: u64,
    /// Expires at (seconds since UNIX epoch).
    pub exp: u64,
}

/// Shared configuration for JWT operations.
#[derive(Clone)]
pub struct JwtConfig {
    pub encoding_key: Arc<EncodingKey>,
    pub decoding_key: Arc<DecodingKey>,
    /// Access token lifetime in seconds (default: 3600 = 1 hour).
    pub access_token_ttl: u64,
    /// Refresh token lifetime in seconds (default: 604800 = 7 days).
    pub refresh_token_ttl: u64,
}

impl JwtConfig {
    /// Create from a shared secret (HMAC-SHA256).
    pub fn from_secret(secret: &str) -> Self {
        Self {
            encoding_key: Arc::new(EncodingKey::from_secret(secret.as_bytes())),
            decoding_key: Arc::new(DecodingKey::from_secret(secret.as_bytes())),
            access_token_ttl: 3600,
            refresh_token_ttl: 604_800,
        }
    }

    /// Override access token lifetime.
    pub fn with_access_ttl(mut self, secs: u64) -> Self {
        self.access_token_ttl = secs;
        self
    }

    /// Override refresh token lifetime.
    pub fn with_refresh_ttl(mut self, secs: u64) -> Self {
        self.refresh_token_ttl = secs;
        self
    }

    /// Issue an access token for the given user.
    pub fn issue_access_token(
        &self,
        user_id: &str,
        tenant_id: &str,
        username: &str,
        scopes: serde_json::Value,
    ) -> Result<String, jsonwebtoken::errors::Error> {
        let now = chrono::Utc::now().timestamp() as u64;
        let claims = Claims {
            sub: user_id.to_string(),
            tenant_id: tenant_id.to_string(),
            username: username.to_string(),
            scopes,
            iat: now,
            exp: now + self.access_token_ttl,
        };
        encode(&Header::default(), &claims, &self.encoding_key)
    }

    /// Issue a refresh token (longer-lived, minimal claims).
    pub fn issue_refresh_token(
        &self,
        user_id: &str,
        tenant_id: &str,
    ) -> Result<String, jsonwebtoken::errors::Error> {
        let now = chrono::Utc::now().timestamp() as u64;
        let claims = Claims {
            sub: user_id.to_string(),
            tenant_id: tenant_id.to_string(),
            username: String::new(),
            scopes: serde_json::Value::Null,
            iat: now,
            exp: now + self.refresh_token_ttl,
        };
        encode(&Header::default(), &claims, &self.encoding_key)
    }

    /// Validate and decode a token.
    pub fn validate_token(&self, token: &str) -> Result<TokenData<Claims>, jsonwebtoken::errors::Error> {
        let mut validation = Validation::default();
        validation.validate_exp = true;
        decode::<Claims>(token, &self.decoding_key, &validation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> JwtConfig {
        JwtConfig::from_secret("test-secret-key-for-unit-tests")
    }

    #[test]
    fn issue_and_validate_access_token() {
        let config = test_config();
        let scopes = serde_json::json!(["read", "write"]);

        let token = config
            .issue_access_token("user-1", "tenant-1", "alice", scopes.clone())
            .expect("should issue token");

        let data = config.validate_token(&token).expect("should validate");
        assert_eq!(data.claims.sub, "user-1");
        assert_eq!(data.claims.tenant_id, "tenant-1");
        assert_eq!(data.claims.username, "alice");
        assert_eq!(data.claims.scopes, scopes);
    }

    #[test]
    fn issue_and_validate_refresh_token() {
        let config = test_config();

        let token = config
            .issue_refresh_token("user-2", "tenant-2")
            .expect("should issue refresh token");

        let data = config.validate_token(&token).expect("should validate");
        assert_eq!(data.claims.sub, "user-2");
        assert_eq!(data.claims.tenant_id, "tenant-2");
        assert_eq!(data.claims.username, "");
    }

    #[test]
    fn wrong_secret_rejects_token() {
        let config = test_config();
        let token = config
            .issue_access_token("u", "t", "n", serde_json::Value::Null)
            .unwrap();

        let other = JwtConfig::from_secret("different-secret");
        assert!(other.validate_token(&token).is_err());
    }

    #[test]
    fn expired_token_rejected() {
        let config = JwtConfig::from_secret("test");
        // Manually create a token with exp in the past
        let claims = Claims {
            sub: "u".to_string(),
            tenant_id: "t".to_string(),
            username: "n".to_string(),
            scopes: serde_json::Value::Null,
            iat: 1_000_000,
            exp: 1_000_001, // way in the past
        };
        let token = encode(&Header::default(), &claims, &config.encoding_key).unwrap();

        assert!(config.validate_token(&token).is_err());
    }

    #[test]
    fn custom_ttl() {
        let config = test_config().with_access_ttl(7200).with_refresh_ttl(86400);
        assert_eq!(config.access_token_ttl, 7200);
        assert_eq!(config.refresh_token_ttl, 86400);
    }
}
