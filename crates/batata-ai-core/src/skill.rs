use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillOutput {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[async_trait]
pub trait Skill: Send + Sync {
    fn name(&self) -> &str;

    fn description(&self) -> &str;

    /// JSON Schema describing the parameters this skill accepts
    fn parameters_schema(&self) -> serde_json::Value;

    async fn execute(&self, params: serde_json::Value) -> Result<SkillOutput>;
}

/// Registry that manages available skills
pub struct SkillRegistry {
    skills: std::collections::HashMap<String, Box<dyn Skill>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: std::collections::HashMap::new(),
        }
    }

    pub fn register(&mut self, skill: Box<dyn Skill>) {
        let name = skill.name().to_string();
        self.skills.insert(name, skill);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Skill> {
        self.skills.get(name).map(|s| s.as_ref())
    }

    pub fn list(&self) -> Vec<(&str, &str)> {
        self.skills
            .values()
            .map(|s| (s.name(), s.description()))
            .collect()
    }

    /// Export all skills as JSON Schema (for MCP tool registration)
    pub fn to_tool_schemas(&self) -> Vec<serde_json::Value> {
        self.skills
            .values()
            .map(|s| {
                serde_json::json!({
                    "name": s.name(),
                    "description": s.description(),
                    "inputSchema": s.parameters_schema(),
                })
            })
            .collect()
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}
