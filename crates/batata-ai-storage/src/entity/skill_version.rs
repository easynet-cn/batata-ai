use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "skill_versions")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub skill_id: String,
    pub version: i32,
    #[sea_orm(column_type = "Text")]
    pub description: String,
    pub parameters_schema: Json,
    pub config: Option<Json>,
    #[sea_orm(column_type = "Text", nullable)]
    pub change_message: Option<String>,
    pub changed_by: Option<String>,
    pub created_at: ChronoDateTime,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::skill::Entity",
        from = "Column::SkillId",
        to = "super::skill::Column::Id",
        on_delete = "Cascade"
    )]
    Skill,
}

impl Related<super::skill::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Skill.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
