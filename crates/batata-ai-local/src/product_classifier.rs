//! Zero-shot product classifier based on CLIP embeddings.
//!
//! Uses pre-computed text embeddings for product names and matches them
//! against image embeddings using cosine similarity. No training required —
//! just provide a list of product names/descriptions.

use serde::{Deserialize, Serialize};
use tracing::info;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::multimodal::ImageData;

use crate::clip::ClipModel;

/// A product entry in the classifier's catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductEntry {
    /// Unique product identifier (SKU)
    pub sku_id: String,
    /// Display name
    pub name: String,
    /// Price (optional, for display)
    pub price: Option<f64>,
    /// Barcode (optional, for cross-validation)
    pub barcode: Option<String>,
    /// Category (e.g., "fruit", "dairy", "beverage")
    pub category: Option<String>,
}

/// A recognition result with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionResult {
    pub sku_id: String,
    pub name: String,
    pub confidence: f32,
    pub price: Option<f64>,
    pub category: Option<String>,
}

/// Cached text embedding for a product
struct ProductEmbedding {
    entry: ProductEntry,
    embedding: Vec<f32>,
}

/// Zero-shot product classifier using CLIP
pub struct ProductClassifier {
    products: Vec<ProductEmbedding>,
}

impl ProductClassifier {
    /// Build a new classifier by pre-computing text embeddings for all products.
    ///
    /// Each product name is prefixed with "a photo of " to improve CLIP
    /// zero-shot accuracy (following OpenAI's recommendation).
    pub fn new(clip: &ClipModel, products: Vec<ProductEntry>) -> Result<Self> {
        info!(
            "building product classifier with {} products",
            products.len()
        );

        let texts: Vec<String> = products
            .iter()
            .map(|p| format!("a photo of {}", p.name))
            .collect();

        let embeddings = clip.embed_texts(&texts)?;

        let product_embeddings = products
            .into_iter()
            .zip(embeddings)
            .map(|(entry, embedding)| ProductEmbedding { entry, embedding })
            .collect();

        info!("product classifier ready");

        Ok(Self {
            products: product_embeddings,
        })
    }

    /// Recognize a product from an image. Returns top-k candidates
    /// sorted by confidence (highest first).
    pub fn recognize(
        &self,
        clip: &ClipModel,
        image: &ImageData,
        top_k: usize,
    ) -> Result<Vec<RecognitionResult>> {
        if self.products.is_empty() {
            return Ok(Vec::new());
        }

        let image_embedding = clip.embed_image(image)?;

        let mut scores: Vec<(usize, f32)> = self
            .products
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let sim = ClipModel::cosine_similarity(&image_embedding, &p.embedding);
                (i, sim)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        let results = scores
            .into_iter()
            .map(|(i, confidence)| {
                let p = &self.products[i].entry;
                RecognitionResult {
                    sku_id: p.sku_id.clone(),
                    name: p.name.clone(),
                    confidence,
                    price: p.price,
                    category: p.category.clone(),
                }
            })
            .collect();

        Ok(results)
    }

    /// Add a new product to the catalog at runtime.
    pub fn add_product(&mut self, clip: &ClipModel, product: ProductEntry) -> Result<()> {
        let text = format!("a photo of {}", product.name);
        let embeddings = clip.embed_texts(&[text])?;
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| BatataError::Inference("failed to compute embedding".into()))?;

        self.products.push(ProductEmbedding {
            entry: product,
            embedding,
        });

        Ok(())
    }

    /// Remove a product by SKU ID.
    pub fn remove_product(&mut self, sku_id: &str) -> bool {
        let before = self.products.len();
        self.products.retain(|p| p.entry.sku_id != sku_id);
        self.products.len() < before
    }

    /// Number of products in the catalog.
    pub fn product_count(&self) -> usize {
        self.products.len()
    }

    /// List all product entries (without embeddings).
    pub fn list_products(&self) -> Vec<&ProductEntry> {
        self.products.iter().map(|p| &p.entry).collect()
    }
}
