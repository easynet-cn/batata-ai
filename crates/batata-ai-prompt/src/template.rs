use std::collections::HashMap;

use batata_ai_core::error::{BatataError, Result};
use serde::{Deserialize, Serialize};

/// A prompt template with variable substitution support.
///
/// Variables are denoted by `{{variable_name}}` in the template string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub name: String,
    pub description: String,
    pub template: String,
    #[serde(default)]
    pub variables: Vec<String>,
}

impl PromptTemplate {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        template: impl Into<String>,
    ) -> Self {
        let template = template.into();
        let variables = Self::extract_variables(&template);
        Self {
            name: name.into(),
            description: description.into(),
            template,
            variables,
        }
    }

    /// Render the template with the given variable values
    pub fn render(&self, vars: &HashMap<String, String>) -> Result<String> {
        let mut result = self.template.clone();
        for var in &self.variables {
            let placeholder = format!("{{{{{var}}}}}");
            let value = vars.get(var).ok_or_else(|| {
                BatataError::Prompt(format!("missing variable: {var}"))
            })?;
            result = result.replace(&placeholder, value);
        }
        Ok(result)
    }

    fn extract_variables(template: &str) -> Vec<String> {
        let mut vars = Vec::new();
        let mut rest = template;
        while let Some(start) = rest.find("{{") {
            if let Some(end) = rest[start..].find("}}") {
                let var = rest[start + 2..start + end].trim().to_string();
                if !vars.contains(&var) {
                    vars.push(var);
                }
                rest = &rest[start + end + 2..];
            } else {
                break;
            }
        }
        vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_template() {
        let tpl = PromptTemplate::new(
            "greeting",
            "A greeting prompt",
            "Hello, {{name}}! You are a {{role}}.",
        );

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("role".to_string(), "developer".to_string());

        let result = tpl.render(&vars).unwrap();
        assert_eq!(result, "Hello, Alice! You are a developer.");
    }

    #[test]
    fn test_missing_variable() {
        let tpl = PromptTemplate::new("test", "test", "Hello, {{name}}!");
        let vars = HashMap::new();
        assert!(tpl.render(&vars).is_err());
    }
}
