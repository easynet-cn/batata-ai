use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "skills")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    #[sea_orm(unique)]
    pub name: String,
    #[sea_orm(column_type = "Text")]
    pub description: String,
    pub parameters_schema: Json,
    pub skill_type: String,
    pub config: Option<Json>,
    pub enabled: bool,
    pub version: i32,
    pub created_at: ChronoDateTime,
    pub updated_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
