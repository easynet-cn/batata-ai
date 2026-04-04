use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "providers")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    #[sea_orm(unique)]
    pub name: String,
    pub provider_type: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub config: Option<Json>,
    pub enabled: bool,
    pub created_at: ChronoDateTime,
    pub updated_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl Related<super::model::Entity> for Entity {
    fn to() -> RelationDef {
        super::model_provider::Relation::Model.def()
    }

    fn via() -> Option<RelationDef> {
        Some(super::model_provider::Relation::Provider.def().rev())
    }
}

impl ActiveModelBehavior for ActiveModel {}
