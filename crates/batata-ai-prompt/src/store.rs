use std::collections::HashMap;
use std::path::Path;

use batata_ai_core::error::{BatataError, Result};

use crate::template::PromptTemplate;

/// In-memory prompt template store with optional file-based persistence
pub struct PromptStore {
    templates: HashMap<String, PromptTemplate>,
}

impl PromptStore {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn add(&mut self, template: PromptTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<PromptTemplate> {
        self.templates.remove(name)
    }

    pub fn list(&self) -> Vec<&PromptTemplate> {
        self.templates.values().collect()
    }

    /// Load templates from a JSON file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let templates: Vec<PromptTemplate> =
            serde_json::from_str(&content).map_err(|e| BatataError::Prompt(e.to_string()))?;

        let mut store = Self::new();
        for tpl in templates {
            store.add(tpl);
        }
        Ok(store)
    }

    /// Save all templates to a JSON file
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
