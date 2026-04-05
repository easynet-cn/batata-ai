use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "api_keys")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub tenant_id: String,
    pub name: String,
    #[sea_orm(unique)]
    pub key_hash: String,
    pub key_prefix: String,
    #[sea_orm(unique)]
    pub app_key: Option<String>,
    pub app_secret_hash: Option<String>,
    pub scopes: Json,
    pub rate_limit: Option<i32>,
    pub expires_at: Option<ChronoDateTime>,
    pub enabled: bool,
    pub last_used_at: Option<ChronoDateTime>,
    pub created_at: ChronoDateTime,
    pub updated_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::tenant::Entity",
        from = "Column::TenantId",
        to = "super::tenant::Column::Id",
        on_delete = "Cascade"
    )]
    Tenant,
}

impl Related<super::tenant::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Tenant.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
