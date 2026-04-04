use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "routing_policies")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    #[sea_orm(unique)]
    pub name: String,
    pub policy_type: String,
    pub config: Json,
    pub enabled: bool,
    pub priority: i32,
    pub created_at: ChronoDateTime,
    pub updated_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
