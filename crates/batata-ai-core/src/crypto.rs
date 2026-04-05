use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, AeadCore, Nonce,
};
use base64::{engine::general_purpose::STANDARD, Engine};

use crate::error::{BatataError, Result};

const ENC_PREFIX: &str = "enc:";

/// Generate a random AES-256 master key and return it as a base64 string.
///
/// Use this to generate the value for `BATATA_MASTER_KEY`.
pub fn generate_master_key() -> String {
    let key = Aes256Gcm::generate_key(&mut OsRng);
    STANDARD.encode(key)
}

/// Generate an app key pair: `(app_key, app_secret_key)`.
///
/// - `app_key`: a readable identifier with `bat_` prefix (20 hex chars).
/// - `app_secret_key`: a high-entropy secret with `bsk_` prefix (64 hex chars / 32 bytes).
pub fn generate_app_key_pair() -> (String, String) {
    use aes_gcm::aead::rand_core::RngCore;

    let mut buf10 = [0u8; 10];
    OsRng.fill_bytes(&mut buf10);
    let app_key = format!("bat_{}", hex::encode(buf10));

    let mut buf32 = [0u8; 32];
    OsRng.fill_bytes(&mut buf32);
    let app_secret = format!("bsk_{}", hex::encode(buf32));

    (app_key, app_secret)
}

/// AES-256-GCM encryptor for sensitive fields (provider API keys, object store secrets).
///
/// Master key is loaded from the `BATATA_MASTER_KEY` environment variable (base64-encoded, 32 bytes).
#[derive(Clone)]
pub struct Encryptor {
    cipher: Aes256Gcm,
}

impl Encryptor {
    /// Create from a base64-encoded 32-byte key.
    pub fn from_base64(key_b64: &str) -> Result<Self> {
        let key_bytes = STANDARD
            .decode(key_b64)
            .map_err(|e| BatataError::Config(format!("invalid base64 master key: {e}")))?;
        let cipher = Aes256Gcm::new_from_slice(&key_bytes)
            .map_err(|_| BatataError::Config("master key must be exactly 32 bytes".into()))?;
        Ok(Self { cipher })
    }

    /// Create from the `BATATA_MASTER_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let key_b64 = std::env::var("BATATA_MASTER_KEY")
            .map_err(|_| BatataError::Config("BATATA_MASTER_KEY env not set".into()))?;
        Self::from_base64(&key_b64)
    }

    /// Encrypt plaintext → `"enc:<base64(nonce)>:<base64(ciphertext)>"`.
    pub fn encrypt(&self, plaintext: &str) -> Result<String> {
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = self
            .cipher
            .encrypt(&nonce, plaintext.as_bytes())
            .map_err(|e| BatataError::Config(format!("encrypt failed: {e}")))?;
        Ok(format!(
            "{}{}:{}",
            ENC_PREFIX,
            STANDARD.encode(nonce),
            STANDARD.encode(ciphertext)
        ))
    }

    /// Decrypt `"enc:<base64(nonce)>:<base64(ciphertext)>"` → plaintext.
    ///
    /// If the value does not start with `"enc:"`, it is treated as legacy plaintext
    /// and returned as-is to support migration.
    pub fn decrypt(&self, value: &str) -> Result<String> {
        let Some(payload) = value.strip_prefix(ENC_PREFIX) else {
            // Legacy plaintext — return as-is
            return Ok(value.to_string());
        };

        let (nonce_b64, ct_b64) = payload
            .split_once(':')
            .ok_or_else(|| BatataError::Config("invalid encrypted format".into()))?;

        let nonce_bytes = STANDARD
            .decode(nonce_b64)
            .map_err(|e| BatataError::Config(format!("invalid nonce base64: {e}")))?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = STANDARD
            .decode(ct_b64)
            .map_err(|e| BatataError::Config(format!("invalid ciphertext base64: {e}")))?;

        let plaintext = self
            .cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| BatataError::Config(format!("decrypt failed: {e}")))?;

        String::from_utf8(plaintext)
            .map_err(|e| BatataError::Config(format!("decrypted value is not valid UTF-8: {e}")))
    }

    /// Encrypt an optional field. `None` stays `None`.
    pub fn encrypt_opt(&self, value: &Option<String>) -> Result<Option<String>> {
        match value {
            Some(v) if !v.is_empty() => self.encrypt(v).map(Some),
            _ => Ok(value.clone()),
        }
    }

    /// Decrypt an optional field. `None` stays `None`.
    pub fn decrypt_opt(&self, value: &Option<String>) -> Result<Option<String>> {
        match value {
            Some(v) if !v.is_empty() => self.decrypt(v).map(Some),
            _ => Ok(value.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_encryptor() -> Encryptor {
        // 32 random bytes, base64-encoded
        let key = STANDARD.encode([0x42u8; 32]);
        Encryptor::from_base64(&key).unwrap()
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let enc = test_encryptor();
        let plaintext = "sk-test-key-12345";
        let encrypted = enc.encrypt(plaintext).unwrap();
        assert!(encrypted.starts_with(ENC_PREFIX));
        let decrypted = enc.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn decrypt_legacy_plaintext() {
        let enc = test_encryptor();
        let legacy = "sk-plain-api-key";
        let decrypted = enc.decrypt(legacy).unwrap();
        assert_eq!(decrypted, legacy);
    }

    #[test]
    fn encrypt_opt_none() {
        let enc = test_encryptor();
        assert_eq!(enc.encrypt_opt(&None).unwrap(), None);
    }

    #[test]
    fn encrypt_opt_some() {
        let enc = test_encryptor();
        let encrypted = enc.encrypt_opt(&Some("secret".into())).unwrap().unwrap();
        assert!(encrypted.starts_with(ENC_PREFIX));
        let decrypted = enc.decrypt_opt(&Some(encrypted)).unwrap().unwrap();
        assert_eq!(decrypted, "secret");
    }

    #[test]
    fn generate_master_key_is_valid() {
        let key = generate_master_key();
        let enc = Encryptor::from_base64(&key).unwrap();
        let ct = enc.encrypt("hello").unwrap();
        assert_eq!(enc.decrypt(&ct).unwrap(), "hello");
    }

    #[test]
    fn generate_app_key_pair_format() {
        let (app_key, app_secret) = generate_app_key_pair();
        assert!(app_key.starts_with("bat_"));
        assert!(app_secret.starts_with("bsk_"));
        assert_eq!(app_key.len(), 4 + 20); // "bat_" + 20 hex
        assert_eq!(app_secret.len(), 4 + 64); // "bsk_" + 64 hex
    }

    #[test]
    fn generate_app_key_pair_unique() {
        let (k1, s1) = generate_app_key_pair();
        let (k2, s2) = generate_app_key_pair();
        assert_ne!(k1, k2);
        assert_ne!(s1, s2);
    }
}
