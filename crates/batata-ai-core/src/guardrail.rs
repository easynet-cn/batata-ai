use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Result of a guardrail check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailResult {
    pub passed: bool,
    pub violations: Vec<Violation>,
}

impl GuardrailResult {
    pub fn pass() -> Self {
        Self {
            passed: true,
            violations: vec![],
        }
    }

    pub fn fail(violations: Vec<Violation>) -> Self {
        Self {
            passed: false,
            violations,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub rule: String,
    pub message: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Guardrail trait — checks input or output content.
#[async_trait]
pub trait Guardrail: Send + Sync {
    fn name(&self) -> &str;

    /// Check input (user messages) before sending to the model.
    async fn check_input(&self, content: &str) -> Result<GuardrailResult>;

    /// Check output (model response) before returning to the user.
    async fn check_output(&self, content: &str) -> Result<GuardrailResult>;
}

/// Pipeline that runs multiple guardrails in sequence.
pub struct GuardrailPipeline {
    guardrails: Vec<Box<dyn Guardrail>>,
}

impl GuardrailPipeline {
    pub fn new() -> Self {
        Self {
            guardrails: vec![],
        }
    }

    pub fn add(mut self, guardrail: Box<dyn Guardrail>) -> Self {
        self.guardrails.push(guardrail);
        self
    }

    /// Run all guardrails on input. Stops at first Critical violation.
    pub async fn check_input(&self, content: &str) -> Result<GuardrailResult> {
        let mut all_violations = vec![];
        for g in &self.guardrails {
            let result = g.check_input(content).await?;
            if !result.passed {
                for v in &result.violations {
                    if v.severity == Severity::Critical {
                        return Ok(GuardrailResult::fail(result.violations));
                    }
                }
                all_violations.extend(result.violations);
            }
        }
        if all_violations.is_empty() {
            Ok(GuardrailResult::pass())
        } else {
            Ok(GuardrailResult::fail(all_violations))
        }
    }

    /// Run all guardrails on output.
    pub async fn check_output(&self, content: &str) -> Result<GuardrailResult> {
        let mut all_violations = vec![];
        for g in &self.guardrails {
            let result = g.check_output(content).await?;
            if !result.passed {
                for v in &result.violations {
                    if v.severity == Severity::Critical {
                        return Ok(GuardrailResult::fail(result.violations));
                    }
                }
                all_violations.extend(result.violations);
            }
        }
        if all_violations.is_empty() {
            Ok(GuardrailResult::pass())
        } else {
            Ok(GuardrailResult::fail(all_violations))
        }
    }
}

impl Default for GuardrailPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock guardrail that always passes.
    struct PassGuardrail;

    #[async_trait]
    impl Guardrail for PassGuardrail {
        fn name(&self) -> &str {
            "pass"
        }
        async fn check_input(&self, _content: &str) -> crate::error::Result<GuardrailResult> {
            Ok(GuardrailResult::pass())
        }
        async fn check_output(&self, _content: &str) -> crate::error::Result<GuardrailResult> {
            Ok(GuardrailResult::pass())
        }
    }

    /// A mock guardrail that returns a non-critical violation.
    struct WarnGuardrail;

    #[async_trait]
    impl Guardrail for WarnGuardrail {
        fn name(&self) -> &str {
            "warn"
        }
        async fn check_input(&self, _content: &str) -> crate::error::Result<GuardrailResult> {
            Ok(GuardrailResult::fail(vec![Violation {
                rule: "warn_rule".to_string(),
                message: "warning".to_string(),
                severity: Severity::Medium,
            }]))
        }
        async fn check_output(&self, _content: &str) -> crate::error::Result<GuardrailResult> {
            Ok(GuardrailResult::fail(vec![Violation {
                rule: "warn_rule".to_string(),
                message: "warning".to_string(),
                severity: Severity::Medium,
            }]))
        }
    }

    /// A mock guardrail that returns a critical violation.
    struct CriticalGuardrail;

    #[async_trait]
    impl Guardrail for CriticalGuardrail {
        fn name(&self) -> &str {
            "critical"
        }
        async fn check_input(&self, _content: &str) -> crate::error::Result<GuardrailResult> {
            Ok(GuardrailResult::fail(vec![Violation {
                rule: "critical_rule".to_string(),
                message: "critical".to_string(),
                severity: Severity::Critical,
            }]))
        }
        async fn check_output(&self, _content: &str) -> crate::error::Result<GuardrailResult> {
            Ok(GuardrailResult::fail(vec![Violation {
                rule: "critical_rule".to_string(),
                message: "critical".to_string(),
                severity: Severity::Critical,
            }]))
        }
    }

    #[tokio::test]
    async fn empty_pipeline_passes() {
        let pipeline = GuardrailPipeline::new();
        let result = pipeline.check_input("anything").await.unwrap();
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[tokio::test]
    async fn pipeline_with_passing_guardrail_passes() {
        let pipeline = GuardrailPipeline::new().add(Box::new(PassGuardrail));
        let result = pipeline.check_input("anything").await.unwrap();
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[tokio::test]
    async fn non_critical_violation_collects_violations() {
        let pipeline = GuardrailPipeline::new().add(Box::new(WarnGuardrail));
        let result = pipeline.check_input("anything").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].severity, Severity::Medium);
    }

    #[tokio::test]
    async fn critical_violation_returns_immediately() {
        // Add a pass guardrail after critical to ensure pipeline stops early
        let pipeline = GuardrailPipeline::new()
            .add(Box::new(CriticalGuardrail))
            .add(Box::new(WarnGuardrail));
        let result = pipeline.check_input("anything").await.unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].severity, Severity::Critical);
    }
}
