use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "request_logs")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub provider_id: String,
    pub provider_name: String,
    pub model_identifier: String,
    pub routing_policy: String,
    pub status: String,
    pub latency_ms: i64,
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
    pub total_tokens: Option<i32>,
    pub estimated_cost: Option<f64>,
    #[sea_orm(column_type = "Text", nullable)]
    pub error_message: Option<String>,
    pub metadata: Option<Json>,
    pub created_at: ChronoDateTime,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
