/// Helper functions for soft delete operations.
/// These are called from the Repository trait implementations.

use sea_orm::*;

use batata_ai_core::error::{BatataError, Result};

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

/// Soft delete: set deleted_at to now.
pub async fn soft_delete<E>(db: &DatabaseConnection, id: &str) -> Result<bool>
where
    E: EntityTrait,
    <E::Column as std::str::FromStr>::Err: std::fmt::Debug,
{
    let col_id: E::Column = "id".parse().map_err(|e| {
        BatataError::Storage(format!("cannot parse 'id' column: {:?}", e))
    })?;
    let col_deleted: E::Column = "deleted_at".parse().map_err(|e| {
        BatataError::Storage(format!("cannot parse 'deleted_at' column: {:?}", e))
    })?;

    let now = chrono::Utc::now().naive_utc();
    let result = E::update_many()
        .col_expr(
            col_deleted,
            sea_orm::sea_query::Expr::value(sea_orm::Value::ChronoDateTime(Some(Box::new(now)))),
        )
        .filter(col_id.eq(id))
        .filter(col_deleted.is_null())
        .exec(db)
        .await
        .map_err(map_db_err)?;
    Ok(result.rows_affected > 0)
}

/// Hard delete: permanently remove.
pub async fn hard_delete<E>(db: &DatabaseConnection, id: &str) -> Result<bool>
where
    E: EntityTrait,
    <E::PrimaryKey as PrimaryKeyTrait>::ValueType: From<String>,
{
    let result = E::delete_by_id(id.to_string())
        .exec(db)
        .await
        .map_err(map_db_err)?;
    Ok(result.rows_affected > 0)
}

/// Restore: clear deleted_at.
pub async fn restore<E>(db: &DatabaseConnection, id: &str) -> Result<bool>
where
    E: EntityTrait,
    <E::Column as std::str::FromStr>::Err: std::fmt::Debug,
{
    let col_id: E::Column = "id".parse().map_err(|e| {
        BatataError::Storage(format!("cannot parse 'id' column: {:?}", e))
    })?;
    let col_deleted: E::Column = "deleted_at".parse().map_err(|e| {
        BatataError::Storage(format!("cannot parse 'deleted_at' column: {:?}", e))
    })?;

    let result = E::update_many()
        .col_expr(
            col_deleted,
            sea_orm::sea_query::Expr::value(sea_orm::Value::ChronoDateTime(None)),
        )
        .filter(col_id.eq(id))
        .filter(col_deleted.is_not_null())
        .exec(db)
        .await
        .map_err(map_db_err)?;
    Ok(result.rows_affected > 0)
}
