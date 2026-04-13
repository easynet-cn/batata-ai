use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::guardrail::{Guardrail, GuardrailResult, Severity, Violation};

/// A guardrail that enforces maximum character length on input and output.
///
/// Returns `High` severity when the content exceeds the configured limit.
pub struct LengthLimit {
    /// Maximum allowed input length in characters.
    max_input_length: usize,
    /// Maximum allowed output length in characters.
    max_output_length: usize,
}

impl LengthLimit {
    /// Create a new `LengthLimit` with the given maximum lengths.
    pub fn new(max_input_length: usize, max_output_length: usize) -> Self {
        Self {
            max_input_length,
            max_output_length,
        }
    }
}

#[async_trait]
impl Guardrail for LengthLimit {
    fn name(&self) -> &str {
        "length_limit"
    }

    async fn check_input(&self, content: &str) -> Result<GuardrailResult> {
        let len = content.chars().count();
        if len > self.max_input_length {
            Ok(GuardrailResult::fail(vec![Violation {
                rule: "length_limit_input".to_string(),
                message: format!(
                    "Input length ({len} chars) exceeds maximum ({} chars)",
                    self.max_input_length
                ),
                severity: Severity::High,
            }]))
        } else {
            Ok(GuardrailResult::pass())
        }
    }

    async fn check_output(&self, content: &str) -> Result<GuardrailResult> {
        let len = content.chars().count();
        if len > self.max_output_length {
            Ok(GuardrailResult::fail(vec![Violation {
                rule: "length_limit_output".to_string(),
                message: format!(
                    "Output length ({len} chars) exceeds maximum ({} chars)",
                    self.max_output_length
                ),
                severity: Severity::High,
            }]))
        } else {
            Ok(GuardrailResult::pass())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn content_within_limits_passes() {
        let limit = LengthLimit::new(100, 200);
        let result = limit.check_input("short input").await.unwrap();
        assert!(result.passed);
        let result = limit.check_output("short output").await.unwrap();
        assert!(result.passed);
    }

    #[tokio::test]
    async fn input_over_limit_fails() {
        let limit = LengthLimit::new(5, 200);
        let result = limit.check_input("this is too long").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].rule, "length_limit_input");
        assert_eq!(result.violations[0].severity, Severity::High);
    }

    #[tokio::test]
    async fn output_over_limit_fails() {
        let limit = LengthLimit::new(200, 5);
        let result = limit.check_output("this is too long").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].rule, "length_limit_output");
        assert_eq!(result.violations[0].severity, Severity::High);
    }
}
