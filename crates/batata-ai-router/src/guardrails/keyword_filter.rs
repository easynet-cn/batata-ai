use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::guardrail::{Guardrail, GuardrailResult, Severity, Violation};

/// A guardrail that blocks content containing specified keywords.
///
/// Performs case-insensitive matching against a configurable list of blocked keywords.
/// Returns `Critical` severity for any match.
pub struct KeywordFilter {
    /// Blocked keywords stored in lowercase for case-insensitive comparison.
    blocked_keywords: Vec<String>,
}

impl KeywordFilter {
    /// Create a new `KeywordFilter` with the given blocked keywords.
    ///
    /// Keywords are stored in lowercase for case-insensitive matching.
    pub fn new(blocked_keywords: Vec<String>) -> Self {
        Self {
            blocked_keywords: blocked_keywords
                .into_iter()
                .map(|k| k.to_lowercase())
                .collect(),
        }
    }

    fn check_content(&self, content: &str) -> GuardrailResult {
        let lower = content.to_lowercase();
        let violations: Vec<Violation> = self
            .blocked_keywords
            .iter()
            .filter(|kw| lower.contains(kw.as_str()))
            .map(|kw| Violation {
                rule: "keyword_filter".to_string(),
                message: format!("Content contains blocked keyword: {kw}"),
                severity: Severity::Critical,
            })
            .collect();

        if violations.is_empty() {
            GuardrailResult::pass()
        } else {
            GuardrailResult::fail(violations)
        }
    }
}

#[async_trait]
impl Guardrail for KeywordFilter {
    fn name(&self) -> &str {
        "keyword_filter"
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
    async fn blocked_keyword_detected() {
        let filter = KeywordFilter::new(vec!["badword".to_string()]);
        let result = filter.check_input("this contains badword here").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].severity, Severity::Critical);
    }

    #[tokio::test]
    async fn case_insensitive_matching() {
        let filter = KeywordFilter::new(vec!["BadWord".to_string()]);
        let result = filter.check_input("BADWORD is here").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
    }

    #[tokio::test]
    async fn clean_content_passes() {
        let filter = KeywordFilter::new(vec!["badword".to_string()]);
        let result = filter.check_input("this is perfectly fine").await.unwrap();
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[tokio::test]
    async fn multiple_violations_collected() {
        let filter = KeywordFilter::new(vec!["bad".to_string(), "evil".to_string()]);
        let result = filter.check_input("bad and evil content").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 2);
    }
}
