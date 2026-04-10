use std::collections::HashMap;
use std::path::Path;

use batata_ai_core::error::{BatataError, Result};
use tera::{Context, Tera};

use crate::template::PromptTemplate;

/// Prompt template store with Tera-based rendering.
///
/// Templates registered in the store can reference each other via
/// `{% extends "base_template" %}` and `{% include "partial" %}`.
pub struct PromptStore {
    templates: HashMap<String, PromptTemplate>,
    tera: Tera,
}

impl PromptStore {
    pub fn new() -> Self {
        let mut tera = Tera::default();
        crate::template::register_ai_filters(&mut tera);
        Self {
            templates: HashMap::new(),
            tera,
        }
    }

    /// Add a template to the store and register it with the Tera engine.
    pub fn add(&mut self, template: PromptTemplate) -> Result<()> {
        self.tera
            .add_raw_template(&template.name, &template.template)
            .map_err(|e| BatataError::Prompt(format!("template parse error: {e}")))?;
        self.templates.insert(template.name.clone(), template);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<PromptTemplate> {
        let removed = self.templates.remove(name);
        if removed.is_some() {
            self.rebuild_tera();
        }
        removed
    }

    /// Rebuild the Tera instance from current templates.
    fn rebuild_tera(&mut self) {
        let mut tera = Tera::default();
        crate::template::register_ai_filters(&mut tera);
        for (name, tpl) in &self.templates {
            // Errors should not occur since templates were validated on add
            let _ = tera.add_raw_template(name, &tpl.template);
        }
        self.tera = tera;
    }

    pub fn list(&self) -> Vec<&PromptTemplate> {
        self.templates.values().collect()
    }

    /// Render a named template with a string variable map.
    pub fn render(
        &self,
        name: &str,
        vars: &HashMap<String, String>,
    ) -> Result<String> {
        let mut context = Context::new();
        for (key, value) in vars {
            context.insert(key, value);
        }
        self.render_with_context(name, &context)
    }

    /// Render a named template with a Tera context.
    ///
    /// This uses the shared Tera instance, so templates can reference
    /// each other via `{% extends %}` and `{% include %}`.
    pub fn render_with_context(&self, name: &str, context: &Context) -> Result<String> {
        if !self.templates.contains_key(name) {
            return Err(BatataError::Prompt(format!("template not found: {name}")));
        }
        self.tera
            .render(name, context)
            .map_err(|e| BatataError::Prompt(format!("template render error: {e}")))
    }

    /// Render a named template with JSON data.
    pub fn render_json(
        &self,
        name: &str,
        value: &serde_json::Value,
    ) -> Result<String> {
        let context = Context::from_value(value.clone())
            .map_err(|e| BatataError::Prompt(format!("invalid context: {e}")))?;
        self.render_with_context(name, &context)
    }

    /// Load templates from a JSON file.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let templates: Vec<PromptTemplate> =
            serde_json::from_str(&content).map_err(|e| BatataError::Prompt(e.to_string()))?;

        let mut store = Self::new();
        for tpl in templates {
            store.add(tpl)?;
        }
        Ok(store)
    }

    /// Save all templates to a JSON file.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let templates: Vec<&PromptTemplate> = self.templates.values().collect();
        let content = serde_json::to_string_pretty(&templates)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl Default for PromptStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_render() {
        let mut store = PromptStore::new();
        store
            .add(PromptTemplate::new(
                "greet",
                "greeting",
                "Hello, {{ name }}!",
            ))
            .unwrap();

        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        assert_eq!(store.render("greet", &vars).unwrap(), "Hello, Alice!");
    }

    #[test]
    fn test_store_template_not_found() {
        let store = PromptStore::new();
        let vars = HashMap::new();
        assert!(store.render("nonexistent", &vars).is_err());
    }

    #[test]
    fn test_store_include() {
        let mut store = PromptStore::new();
        store
            .add(PromptTemplate::new(
                "header",
                "header partial",
                "System: You are a {{ role }}.",
            ))
            .unwrap();
        store
            .add(PromptTemplate::new(
                "full_prompt",
                "full prompt with include",
                "{% include \"header\" %}\n\nUser: {{ question }}",
            ))
            .unwrap();

        let data = serde_json::json!({
            "role": "helpful assistant",
            "question": "What is Rust?"
        });
        let result = store.render_json("full_prompt", &data).unwrap();
        assert_eq!(
            result,
            "System: You are a helpful assistant.\n\nUser: What is Rust?"
        );
    }

    #[test]
    fn test_store_extends() {
        let mut store = PromptStore::new();
        store
            .add(PromptTemplate::new(
                "base",
                "base template",
                "{% block system %}Default system prompt{% endblock %}\n---\n{% block content %}{% endblock %}",
            ))
            .unwrap();
        store
            .add(PromptTemplate::new(
                "qa_prompt",
                "QA prompt extending base",
                "{% extends \"base\" %}{% block system %}You are a QA expert.{% endblock %}{% block content %}Question: {{ question }}{% endblock %}",
            ))
            .unwrap();

        let data = serde_json::json!({"question": "How to test Rust code?"});
        let result = store.render_json("qa_prompt", &data).unwrap();
        assert_eq!(
            result,
            "You are a QA expert.\n---\nQuestion: How to test Rust code?"
        );
    }
}
