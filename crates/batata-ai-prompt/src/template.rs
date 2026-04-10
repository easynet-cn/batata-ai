use std::collections::HashMap;

use batata_ai_core::error::{BatataError, Result};
use serde::{Deserialize, Serialize};
use tera::{Context, Tera, Value};

/// A prompt template with Tera-based rendering support.
///
/// Supports Jinja2/Tera syntax:
/// - Variables: `{{ variable_name }}`
/// - Conditionals: `{% if condition %}...{% endif %}`
/// - Loops: `{% for item in items %}...{% endfor %}`
/// - Filters: `{{ name | upper }}`, `{{ text | truncate(length=100) }}`
/// - Defaults: `{{ name | default(value="World") }}`
/// - Template inheritance: `{% extends "base" %}` / `{% block name %}...{% endblock %}`
/// - Includes: `{% include "partial" %}`
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

    /// Render the template with the given variable values (string-only).
    pub fn render(&self, vars: &HashMap<String, String>) -> Result<String> {
        let mut context = Context::new();
        for (key, value) in vars {
            context.insert(key, value);
        }
        self.render_with_context(&context)
    }

    /// Render the template with a Tera context (supports complex types).
    ///
    /// Use this when you need to pass arrays, objects, or nested data
    /// for loops and conditionals.
    pub fn render_with_context(&self, context: &Context) -> Result<String> {
        let mut tera = Tera::default();
        register_ai_filters(&mut tera);
        tera.add_raw_template(&self.name, &self.template)
            .map_err(|e| BatataError::Prompt(format!("template parse error: {e}")))?;
        tera.render(&self.name, context)
            .map_err(|e| BatataError::Prompt(format!("template render error: {e}")))
    }

    /// Render with a JSON value as the context root.
    ///
    /// Convenient for passing structured data from API requests:
    /// ```ignore
    /// let data = serde_json::json!({"user": "Alice", "items": ["a", "b"]});
    /// tpl.render_json(&data)?;
    /// ```
    pub fn render_json(&self, value: &serde_json::Value) -> Result<String> {
        let context = Context::from_value(value.clone())
            .map_err(|e| BatataError::Prompt(format!("invalid context: {e}")))?;
        self.render_with_context(&context)
    }

    /// Validate the template syntax without rendering.
    pub fn validate(&self) -> Result<()> {
        let mut tera = Tera::default();
        tera.add_raw_template(&self.name, &self.template)
            .map_err(|e| BatataError::Prompt(format!("template validation error: {e}")))?;
        Ok(())
    }

    fn extract_variables(template: &str) -> Vec<String> {
        let mut vars = Vec::new();
        let mut rest = template;
        while let Some(start) = rest.find("{{") {
            if let Some(end) = rest[start..].find("}}") {
                let raw = rest[start + 2..start + end].trim();
                // Extract the variable name (before any filter pipe)
                let var = raw.split('|').next().unwrap_or(raw).trim();
                // Skip Tera keywords and expressions
                if !var.is_empty()
                    && !var.contains('(')
                    && !var.contains(' ')
                    && !vars.iter().any(|v: &String| v == var)
                {
                    vars.push(var.to_string());
                }
                rest = &rest[start + end + 2..];
            } else {
                break;
            }
        }
        vars
    }
}

/// Register AI-specific Tera filters.
pub(crate) fn register_ai_filters(tera: &mut Tera) {
    tera.register_filter("system_msg", system_msg_filter);
    tera.register_filter("user_msg", user_msg_filter);
    tera.register_filter("assistant_msg", assistant_msg_filter);
    tera.register_filter("json_encode", json_encode_filter);
}

/// Format text as a system message: `[SYSTEM] content`
fn system_msg_filter(
    value: &Value,
    _args: &HashMap<String, Value>,
) -> tera::Result<Value> {
    let s = value
        .as_str()
        .ok_or_else(|| tera::Error::msg("system_msg filter expects a string"))?;
    Ok(Value::String(format!("[SYSTEM] {s}")))
}

/// Format text as a user message: `[USER] content`
fn user_msg_filter(
    value: &Value,
    _args: &HashMap<String, Value>,
) -> tera::Result<Value> {
    let s = value
        .as_str()
        .ok_or_else(|| tera::Error::msg("user_msg filter expects a string"))?;
    Ok(Value::String(format!("[USER] {s}")))
}

/// Format text as an assistant message: `[ASSISTANT] content`
fn assistant_msg_filter(
    value: &Value,
    _args: &HashMap<String, Value>,
) -> tera::Result<Value> {
    let s = value
        .as_str()
        .ok_or_else(|| tera::Error::msg("assistant_msg filter expects a string"))?;
    Ok(Value::String(format!("[ASSISTANT] {s}")))
}

/// Encode a value as a JSON string.
fn json_encode_filter(
    value: &Value,
    _args: &HashMap<String, Value>,
) -> tera::Result<Value> {
    let json = serde_json::to_string(value)
        .map_err(|e| tera::Error::msg(format!("json_encode error: {e}")))?;
    Ok(Value::String(json))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_render() {
        let tpl = PromptTemplate::new(
            "greeting",
            "A greeting prompt",
            "Hello, {{ name }}! You are a {{ role }}.",
        );

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("role".to_string(), "developer".to_string());

        let result = tpl.render(&vars).unwrap();
        assert_eq!(result, "Hello, Alice! You are a developer.");
    }

    #[test]
    fn test_missing_variable() {
        let tpl = PromptTemplate::new("test", "test", "Hello, {{ name }}!");
        let vars = HashMap::new();
        assert!(tpl.render(&vars).is_err());
    }

    #[test]
    fn test_default_value() {
        let tpl = PromptTemplate::new(
            "default",
            "test defaults",
            "Hello, {{ name | default(value=\"World\") }}!",
        );
        let vars = HashMap::new();
        let result = tpl.render(&vars).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_conditional() {
        let tpl = PromptTemplate::new(
            "conditional",
            "test conditionals",
            "{% if formal %}Dear {{ name }},{% else %}Hey {{ name }}!{% endif %}",
        );

        let mut context = Context::new();
        context.insert("name", "Bob");
        context.insert("formal", &true);
        assert_eq!(tpl.render_with_context(&context).unwrap(), "Dear Bob,");

        let mut context = Context::new();
        context.insert("name", "Bob");
        context.insert("formal", &false);
        assert_eq!(tpl.render_with_context(&context).unwrap(), "Hey Bob!");
    }

    #[test]
    fn test_loop() {
        let tpl = PromptTemplate::new(
            "few_shot",
            "few-shot examples",
            "Examples:\n{% for ex in examples %}- {{ ex }}\n{% endfor %}Now answer:",
        );

        let data = serde_json::json!({
            "examples": ["The sky is blue.", "Water is wet."]
        });
        let result = tpl.render_json(&data).unwrap();
        assert_eq!(
            result,
            "Examples:\n- The sky is blue.\n- Water is wet.\nNow answer:"
        );
    }

    #[test]
    fn test_filters() {
        let tpl = PromptTemplate::new(
            "filters",
            "test filters",
            "{{ name | upper }} - {{ bio | truncate(length=10) }}",
        );

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "alice".to_string());
        vars.insert("bio".to_string(), "A very long biography text here".to_string());

        let result = tpl.render(&vars).unwrap();
        assert!(result.starts_with("ALICE - A very l"));
    }

    #[test]
    fn test_system_msg_filter() {
        let tpl = PromptTemplate::new(
            "sys",
            "system message",
            "{{ instruction | system_msg }}",
        );

        let mut vars = HashMap::new();
        vars.insert("instruction".to_string(), "You are a helpful assistant.".to_string());
        let result = tpl.render(&vars).unwrap();
        assert_eq!(result, "[SYSTEM] You are a helpful assistant.");
    }

    #[test]
    fn test_json_encode_filter() {
        let tpl = PromptTemplate::new(
            "json",
            "json encode",
            "Data: {{ data | json_encode }}",
        );

        let mut context = Context::new();
        context.insert("data", &vec!["a", "b", "c"]);
        let result = tpl.render_with_context(&context).unwrap();
        assert_eq!(result, "Data: [\"a\",\"b\",\"c\"]");
    }

    #[test]
    fn test_render_json() {
        let tpl = PromptTemplate::new(
            "json_ctx",
            "json context",
            "Hello {{ user.name }}, you have {{ user.tasks | length }} tasks.",
        );

        let data = serde_json::json!({
            "user": {
                "name": "Alice",
                "tasks": ["task1", "task2", "task3"]
            }
        });
        let result = tpl.render_json(&data).unwrap();
        assert_eq!(result, "Hello Alice, you have 3 tasks.");
    }

    #[test]
    fn test_validate_ok() {
        let tpl = PromptTemplate::new("v", "valid", "{% if x %}yes{% endif %}");
        assert!(tpl.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_syntax() {
        let tpl = PromptTemplate::new("v", "invalid", "{% if %}broken{% endif %}");
        assert!(tpl.validate().is_err());
    }

    #[test]
    fn test_complex_ai_prompt() {
        let template = r#"{{ system_prompt | system_msg }}

{% if examples %}Few-shot examples:
{% for ex in examples %}
Q: {{ ex.question }}
A: {{ ex.answer }}
{% endfor %}{% endif %}
{{ user_query | user_msg }}"#;

        let tpl = PromptTemplate::new("complex", "complex AI prompt", template);

        let data = serde_json::json!({
            "system_prompt": "You are a helpful assistant.",
            "examples": [
                {"question": "What is 1+1?", "answer": "2"},
                {"question": "What is 2+2?", "answer": "4"}
            ],
            "user_query": "What is 3+3?"
        });

        let result = tpl.render_json(&data).unwrap();
        assert!(result.contains("[SYSTEM] You are a helpful assistant."));
        assert!(result.contains("Q: What is 1+1?"));
        assert!(result.contains("A: 2"));
        assert!(result.contains("[USER] What is 3+3?"));
    }
}
