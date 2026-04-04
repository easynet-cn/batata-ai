use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    pub id: String,
    pub model_id: String,
    pub provider_id: String,
    pub input_cost_per_1k: f64,
    pub output_cost_per_1k: f64,
    pub currency: String,
    pub effective_from: NaiveDateTime,
    pub created_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}
