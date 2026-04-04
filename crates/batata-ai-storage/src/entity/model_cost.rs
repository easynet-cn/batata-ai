use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "model_costs")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub model_id: String,
    pub provider_id: String,
    pub input_cost_per_1k: f64,
    pub output_cost_per_1k: f64,
    pub currency: String,
    pub effective_from: ChronoDateTime,
    pub created_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::model::Entity",
        from = "Column::ModelId",
        to = "super::model::Column::Id",
        on_delete = "Cascade"
    )]
    Model,
    #[sea_orm(
        belongs_to = "super::provider::Entity",
        from = "Column::ProviderId",
        to = "super::provider::Column::Id",
        on_delete = "Cascade"
    )]
    Provider,
}

impl Related<super::model::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Model.def()
    }
}

impl Related<super::provider::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Provider.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
