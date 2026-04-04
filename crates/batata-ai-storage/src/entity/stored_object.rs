use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "stored_objects")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub bucket_id: String,
    #[sea_orm(unique)]
    pub key: String,
    pub original_name: Option<String>,
    pub content_type: String,
    pub size: i64,
    pub checksum: Option<String>,
    pub metadata: Option<Json>,
    pub created_at: ChronoDateTime,
    pub deleted_at: Option<ChronoDateTime>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::object_store_bucket::Entity",
        from = "Column::BucketId",
        to = "super::object_store_bucket::Column::Id",
        on_delete = "Cascade"
    )]
    ObjectStoreBucket,
}

impl Related<super::object_store_bucket::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ObjectStoreBucket.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
