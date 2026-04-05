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
