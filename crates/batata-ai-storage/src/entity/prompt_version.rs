use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "prompt_versions")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub prompt_id: String,
    pub version: i32,
    #[sea_orm(column_type = "Text")]
    pub template: String,
    pub variables: Json,
    #[sea_orm(column_type = "Text")]
    pub description: String,
    #[sea_orm(column_type = "Text", nullable)]
    pub change_message: Option<String>,
    pub changed_by: Option<String>,
    pub created_at: ChronoDateTime,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::prompt::Entity",
        from = "Column::PromptId",
        to = "super::prompt::Column::Id",
        on_delete = "Cascade"
    )]
    Prompt,
}

impl Related<super::prompt::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Prompt.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
