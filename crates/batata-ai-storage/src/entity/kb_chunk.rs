use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "kb_chunks")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub kb_id: String,
    pub doc_id: String,
    pub ord: i32,
    #[sea_orm(column_type = "Text")]
    pub text: String,
    #[sea_orm(column_type = "Blob")]
    pub embedding: Vec<u8>,
    pub metadata: Option<Json>,
    pub created_at: ChronoDateTime,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
