use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "models")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    #[sea_orm(unique)]
    pub name: String,
    pub model_type: String,
    pub context_length: Option<i32>,
    pub description: Option<String>,
    pub metadata: Option<Json>,
    pub enabled: bool,
    pub created_at: ChronoDateTime,
    pub updated_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl Related<super::provider::Entity> for Entity {
    fn to() -> RelationDef {
        super::model_provider::Relation::Provider.def()
    }

    fn via() -> Option<RelationDef> {
        Some(super::model_provider::Relation::Model.def().rev())
    }
}

impl ActiveModelBehavior for ActiveModel {}
