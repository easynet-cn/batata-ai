use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "kb_documents")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub kb_id: String,
    pub tenant_id: Option<String>,
    pub source_uri: String,
    pub title: Option<String>,
    pub mime: Option<String>,
    pub status: String,
    #[sea_orm(column_type = "Text", nullable)]
    pub error: Option<String>,
    pub chunk_count: i32,
    pub metadata: Option<Json>,
    pub created_at: ChronoDateTime,
    pub updated_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
