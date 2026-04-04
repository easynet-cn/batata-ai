use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "object_store_buckets")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub config_id: String,
    pub tenant_id: Option<String>,
    pub name: String,
    pub bucket: String,
    pub root_path: Option<String>,
    pub access_policy: String,
    pub custom_domain: Option<String>,
    pub is_default: bool,
    pub enabled: bool,
    pub config: Option<Json>,
    pub created_at: ChronoDateTime,
    pub updated_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::object_store_config::Entity",
        from = "Column::ConfigId",
        to = "super::object_store_config::Column::Id",
        on_delete = "Cascade"
    )]
    ObjectStoreConfig,
}

impl Related<super::object_store_config::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ObjectStoreConfig.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
