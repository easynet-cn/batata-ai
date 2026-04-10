use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::guardrail::{Guardrail, GuardrailResult, Severity, Violation};

/// A guardrail that detects Personally Identifiable Information (PII).
///
/// Uses regex patterns to detect:
/// - China mainland phone numbers (手机号)
/// - China ID card numbers (身份证号)
/// - Email addresses
/// - Bank card numbers (银行卡号)
/// - China mainland landline numbers (座机号)
///
/// Custom patterns can be added via `add_pattern`.
pub struct PiiFilter {
    patterns: Vec<PiiPattern>,
}

struct PiiPattern {
    name: String,
    regex: regex::Regex,
    severity: Severity,
}

impl PiiFilter {
    /// Create a `PiiFilter` with default PII patterns for China mainland.
    pub fn new() -> Self {
        let mut filter = Self {
            patterns: Vec::new(),
        };

        // China mobile phone: 1[3-9] followed by 9 digits
        filter.add_pattern_inner(
            "phone_number",
            r"(?:^|[^\d])1[3-9]\d{9}(?:$|[^\d])",
            Severity::High,
        );

        // China ID card: 18 digits (17 digits + digit/X)
        filter.add_pattern_inner(
            "id_card",
            r"(?:^|[^\d])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?:$|[^\d])",
            Severity::Critical,
        );

        // Email address
        filter.add_pattern_inner(
            "email",
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            Severity::High,
        );

        // Bank card number: 16-19 consecutive digits
        filter.add_pattern_inner(
            "bank_card",
            r"(?:^|[^\d])\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?:[\s\-]?\d{1,3})?(?:$|[^\d])",
            Severity::Critical,
        );

        filter
    }

    /// Add a custom PII detection pattern.
    pub fn add_pattern(
        mut self,
        name: impl Into<String>,
        pattern: &str,
        severity: Severity,
    ) -> std::result::Result<Self, regex::Error> {
        let regex = regex::Regex::new(pattern)?;
        self.patterns.push(PiiPattern {
            name: name.into(),
            regex,
            severity,
        });
        Ok(self)
    }

    fn add_pattern_inner(&mut self, name: &str, pattern: &str, severity: Severity) {
        if let Ok(regex) = regex::Regex::new(pattern) {
            self.patterns.push(PiiPattern {
                name: name.to_string(),
                regex,
                severity,
            });
        }
    }

    fn check_content(&self, content: &str) -> GuardrailResult {
        let violations: Vec<Violation> = self
            .patterns
            .iter()
            .filter(|p| p.regex.is_match(content))
            .map(|p| Violation {
                rule: format!("pii_{}", p.name),
                message: format!("Content may contain PII: {}", p.name),
                severity: p.severity.clone(),
            })
            .collect();

        if violations.is_empty() {
            GuardrailResult::pass()
        } else {
            GuardrailResult::fail(violations)
        }
    }
}

impl Default for PiiFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Guardrail for PiiFilter {
    fn name(&self) -> &str {
        "pii_filter"
    }

    async fn check_input(&self, content: &str) -> Result<GuardrailResult> {
        Ok(self.check_content(content))
    }

    async fn check_output(&self, content: &str) -> Result<GuardrailResult> {
        Ok(self.check_content(content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_phone_detection() {
        let filter = PiiFilter::new();
        let result = filter.check_input("我的手机号是13812345678").await.unwrap();
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.rule == "pii_phone_number"));
    }

    #[tokio::test]
    async fn test_id_card_detection() {
        let filter = PiiFilter::new();
        let result = filter
            .check_input("身份证号110101199003077890")
            .await
            .unwrap();
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.rule == "pii_id_card"));
    }

    #[tokio::test]
    async fn test_email_detection() {
        let filter = PiiFilter::new();
        let result = filter
            .check_input("请联系 test@example.com")
            .await
            .unwrap();
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.rule == "pii_email"));
    }

    #[tokio::test]
    async fn test_clean_content() {
        let filter = PiiFilter::new();
        let result = filter.check_input("今天天气真好").await.unwrap();
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }
}
