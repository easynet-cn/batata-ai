//! Sea-ORM backed implementation of `batata_ai_core::rag::VectorStore`.
//!
//! Embeddings are stored as little-endian f32 bytes in a BLOB column.
//! Search is a brute-force cosine scan over all chunks inside a KB — fine
//! for up to ~100k chunks per KB. Swap for `sqlite-vec` / `pgvector` in M5.

use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::rag::{RagChunk, RagHit, RagSearchQuery, VectorStore};

use crate::entity::kb_chunk;

pub struct SeaOrmVectorStore {
    db: DatabaseConnection,
}

impl SeaOrmVectorStore {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn model_to_chunk(m: kb_chunk::Model) -> RagChunk {
    RagChunk {
        id: m.id,
        doc_id: m.doc_id,
        kb_id: m.kb_id,
        ord: m.ord as u32,
        text: m.text,
        embedding: decode_embedding(&m.embedding),
        metadata: m.metadata.unwrap_or(serde_json::Value::Null),
    }
}

#[async_trait]
impl VectorStore for SeaOrmVectorStore {
    async fn upsert(&self, chunks: Vec<RagChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let now = chrono::Utc::now().naive_utc();
        let rows: Vec<kb_chunk::ActiveModel> = chunks
            .into_iter()
            .map(|c| kb_chunk::ActiveModel {
                id: Set(c.id),
                kb_id: Set(c.kb_id),
                doc_id: Set(c.doc_id),
                ord: Set(c.ord as i32),
                text: Set(c.text),
                embedding: Set(encode_embedding(&c.embedding)),
                metadata: Set(if c.metadata.is_null() {
                    None
                } else {
                    Some(c.metadata)
                }),
                created_at: Set(now),
            })
            .collect();
        kb_chunk::Entity::insert_many(rows)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(())
    }

    async fn delete_by_doc(&self, kb_id: &str, doc_id: &str) -> Result<()> {
        kb_chunk::Entity::delete_many()
            .filter(kb_chunk::Column::KbId.eq(kb_id))
            .filter(kb_chunk::Column::DocId.eq(doc_id))
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(())
    }

    async fn delete_by_kb(&self, kb_id: &str) -> Result<()> {
        kb_chunk::Entity::delete_many()
            .filter(kb_chunk::Column::KbId.eq(kb_id))
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(())
    }

    async fn search(&self, q: RagSearchQuery) -> Result<Vec<RagHit>> {
        // Brute-force: load all chunks in the KB, score in Rust.
        // Acceptable up to a few hundred-k rows; swap for ANN later.
        let rows = kb_chunk::Entity::find()
            .filter(kb_chunk::Column::KbId.eq(&q.kb_id))
            .all(&self.db)
            .await
            .map_err(map_db_err)?;

        let mut hits: Vec<RagHit> = rows
            .into_iter()
            .map(model_to_chunk)
            .map(|chunk| RagHit {
                score: cosine(&q.embedding, &chunk.embedding),
                chunk,
            })
            .filter(|h| q.min_score.map_or(true, |m| h.score >= m))
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(q.top_k);
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sea_orm::Database;
    use sea_orm_migration::MigratorTrait;

    async fn setup_db() -> DatabaseConnection {
        // Stub master key so the encrypt_secrets migration doesn't fail.
        if std::env::var("BATATA_MASTER_KEY").is_err() {
            // SAFETY: process-local test fixture.
            unsafe {
                std::env::set_var(
                    "BATATA_MASTER_KEY",
                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                );
            }
        }
        let db = Database::connect("sqlite::memory:").await.unwrap();
        crate::migration::Migrator::up(&db, None).await.unwrap();
        db
    }

    fn dummy_embedding(seed: u8) -> Vec<f32> {
        (0..8).map(|i| (i as f32 + seed as f32) * 0.1).collect()
    }

    #[tokio::test]
    async fn upsert_and_search() {
        let db = setup_db().await;
        let store = SeaOrmVectorStore::new(db);

        let chunks = vec![
            RagChunk {
                id: "c1".into(),
                doc_id: "d1".into(),
                kb_id: "kb".into(),
                ord: 0,
                text: "alpha".into(),
                embedding: dummy_embedding(1),
                metadata: serde_json::json!({"tag": "a"}),
            },
            RagChunk {
                id: "c2".into(),
                doc_id: "d2".into(),
                kb_id: "kb".into(),
                ord: 0,
                text: "beta".into(),
                embedding: dummy_embedding(5),
                metadata: serde_json::Value::Null,
            },
        ];
        store.upsert(chunks).await.unwrap();

        let hits = store
            .search(RagSearchQuery {
                kb_id: "kb".into(),
                embedding: dummy_embedding(1),
                top_k: 1,
                min_score: None,
                filter: serde_json::Value::Null,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.id, "c1");
        assert!(hits[0].score > 0.99);
    }

    #[tokio::test]
    async fn delete_by_kb_wipes_target_only() {
        let db = setup_db().await;
        let store = SeaOrmVectorStore::new(db);
        store
            .upsert(vec![
                RagChunk {
                    id: "a".into(),
                    doc_id: "d1".into(),
                    kb_id: "kb_wipe".into(),
                    ord: 0,
                    text: "one".into(),
                    embedding: dummy_embedding(1),
                    metadata: serde_json::Value::Null,
                },
                RagChunk {
                    id: "c".into(),
                    doc_id: "d3".into(),
                    kb_id: "kb_keep".into(),
                    ord: 0,
                    text: "three".into(),
                    embedding: dummy_embedding(3),
                    metadata: serde_json::Value::Null,
                },
            ])
            .await
            .unwrap();

        store.delete_by_kb("kb_wipe").await.unwrap();

        let wiped = store
            .search(RagSearchQuery {
                kb_id: "kb_wipe".into(),
                embedding: dummy_embedding(1),
                top_k: 10,
                min_score: None,
                filter: serde_json::Value::Null,
            })
            .await
            .unwrap();
        assert!(wiped.is_empty());

        let kept = store
            .search(RagSearchQuery {
                kb_id: "kb_keep".into(),
                embedding: dummy_embedding(3),
                top_k: 10,
                min_score: None,
                filter: serde_json::Value::Null,
            })
            .await
            .unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].chunk.id, "c");
    }

    #[tokio::test]
    async fn delete_by_doc_removes_rows() {
        let db = setup_db().await;
        let store = SeaOrmVectorStore::new(db);
        store
            .upsert(vec![RagChunk {
                id: "c1".into(),
                doc_id: "d1".into(),
                kb_id: "kb".into(),
                ord: 0,
                text: "t".into(),
                embedding: dummy_embedding(2),
                metadata: serde_json::Value::Null,
            }])
            .await
            .unwrap();
        store.delete_by_doc("kb", "d1").await.unwrap();
        let hits = store
            .search(RagSearchQuery {
                kb_id: "kb".into(),
                embedding: dummy_embedding(2),
                top_k: 10,
                min_score: None,
                filter: serde_json::Value::Null,
            })
            .await
            .unwrap();
        assert!(hits.is_empty());
    }
}
