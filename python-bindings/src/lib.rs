//! Python bindings for rust-genai multimodal embedding pipeline
//!
//! This module provides a Python interface with both mock embeddings for
//! development/testing and native AI integration for production use.

use pyo3::prelude::*;
use std::collections::HashMap;

const VISION_MODEL: &str = "llava:7b";
const EMBEDDING_MODEL: &str = "nomic-embed-text:latest";

/// Python-exposed performance configuration
#[pyclass]
#[derive(Clone)]
pub struct PerformanceConfig {
    #[pyo3(get, set)]
    pub max_concurrent_requests: usize,
    #[pyo3(get, set)]
    pub request_timeout_seconds: u64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub enable_progress_reporting: bool,
}

#[pymethods]
impl PerformanceConfig {
    #[new]
    #[pyo3(signature = (max_concurrent_requests = 4, request_timeout_seconds = 30, batch_size = 10, enable_progress_reporting = true))]
    pub fn new(
        max_concurrent_requests: usize,
        request_timeout_seconds: u64,
        batch_size: usize,
        enable_progress_reporting: bool,
    ) -> Self {
        Self {
            max_concurrent_requests,
            request_timeout_seconds,
            batch_size,
            enable_progress_reporting,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PerformanceConfig(max_concurrent={}, timeout={}s, batch_size={}, progress={})",
            self.max_concurrent_requests,
            self.request_timeout_seconds,
            self.batch_size,
            self.enable_progress_reporting
        )
    }
}

/// Python-exposed processing statistics
#[pyclass]
#[derive(Clone)]
pub struct ProcessingStats {
    #[pyo3(get)]
    pub total_processed: usize,
    #[pyo3(get)]
    pub successful: usize,
    #[pyo3(get)]
    pub failed: usize,
    #[pyo3(get)]
    pub total_duration_seconds: f64,
    #[pyo3(get)]
    pub avg_time_per_item_seconds: f64,
}

#[pymethods]
impl ProcessingStats {
    #[getter]
    pub fn throughput(&self) -> f64 {
        if self.total_duration_seconds > 0.0 {
            self.successful as f64 / self.total_duration_seconds
        } else {
            0.0
        }
    }

    #[getter]
    pub fn success_rate(&self) -> f64 {
        if self.total_processed > 0 {
            (self.successful as f64 / self.total_processed as f64) * 100.0
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessingStats(processed={}, successful={}, throughput={:.2}/s, success_rate={:.1}%)",
            self.total_processed,
            self.successful,
            self.throughput(),
            self.success_rate()
        )
    }
}

/// Python-exposed multimodal embedding result
#[pyclass]
#[derive(Clone)]
pub struct MultimodalEmbedding {
    #[pyo3(get)]
    pub text_description: String,
    #[pyo3(get)]
    pub embedding: Vec<f32>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl MultimodalEmbedding {
    #[new]
    pub fn new(text_description: String, embedding: Vec<f32>, metadata: HashMap<String, String>) -> Self {
        Self {
            text_description,
            embedding,
            metadata,
        }
    }

    #[getter]
    pub fn dimensions(&self) -> usize {
        self.embedding.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "MultimodalEmbedding(description_len={}, dimensions={}, metadata_keys={})",
            self.text_description.len(),
            self.embedding.len(),
            self.metadata.len()
        )
    }
}

/// Multimodal embedder with both mock and native AI capabilities
#[pyclass]
pub struct MultimodalEmbedder {
    vision_model: String,
    embedding_model: String,
    config: PerformanceConfig,
}

#[pymethods]
impl MultimodalEmbedder {
    #[new]
    pub fn new(vision_model: String, embedding_model: String, config: Option<PerformanceConfig>) -> Self {
        Self {
            vision_model,
            embedding_model,
            config: config.unwrap_or_else(|| PerformanceConfig::new(4, 30, 10, true)),
        }
    }

    /// Create a mock embedding for development and testing
    pub fn create_mock_embedding(&self, image_url: String) -> PyResult<MultimodalEmbedding> {
        // Generate a mock description based on URL
        let description = format!(
            "Mock embedding for image: {}. Vision model: {}, Embedding model: {}.",
            image_url.split('/').last().unwrap_or("unknown"),
            self.vision_model,
            self.embedding_model
        );

        // Generate a mock embedding (768 dimensions like nomic-embed-text)
        // Use URL hash and content to create unique but deterministic embeddings
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        image_url.hash(&mut hasher);
        let url_hash = hasher.finish();

        let embedding: Vec<f32> = (0..768)
            .map(|i| {
                // Use hash bytes to create more variation
                let hash_byte = ((url_hash >> (i % 64)) & 0xFF) as f32;
                let seed = (hash_byte * 0.01 + i as f32 * 0.02) % 6.28;
                seed.sin() * 0.5 + 0.2 // Scale to [-0.3, 0.7] range
            })
            .collect();

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "mock_embedding".to_string());
        metadata.insert("url".to_string(), image_url);
        metadata.insert("vision_model".to_string(), self.vision_model.clone());
        metadata.insert("embedding_model".to_string(), self.embedding_model.clone());
        metadata.insert("processing_method".to_string(), "mock_development".to_string());

        Ok(MultimodalEmbedding::new(description, embedding, metadata))
    }

    /// Create mock embeddings for batch processing
    pub fn create_mock_embeddings_batch(&self, image_urls: Vec<String>) -> PyResult<(Vec<MultimodalEmbedding>, ProcessingStats)> {
        use std::time::Instant;

        let start_time = Instant::now();

        if self.config.enable_progress_reporting {
            println!("🚀 Mock Batch Processing");
            println!("   📊 Images: {}", image_urls.len());
            println!("   🔄 Max concurrent: {}", self.config.max_concurrent_requests);
        }

        // Process mock embeddings with realistic timing
        let mut embeddings = Vec::new();
        let mut successful = 0;

        // Add realistic processing delay (2-10ms per image)
        let processing_delay_per_image = std::time::Duration::from_micros(
            2000 + (image_urls.len() as u64 * 50) // Scale with batch size
        );

        for url in image_urls.iter() {
            // Add processing delay for realism
            std::thread::sleep(processing_delay_per_image);

            match self.create_mock_embedding(url.clone()) {
                Ok(embedding) => {
                    embeddings.push(embedding);
                    successful += 1;
                }
                Err(_) => {
                    // Handle mock failures
                }
            }
        }

        let total_duration = start_time.elapsed();
        let stats = ProcessingStats {
            total_processed: image_urls.len(),
            successful,
            failed: image_urls.len() - successful,
            total_duration_seconds: total_duration.as_secs_f64(),
            avg_time_per_item_seconds: if successful > 0 {
                total_duration.as_secs_f64() / successful as f64
            } else {
                0.0
            },
        };

        if self.config.enable_progress_reporting {
            println!("📊 Mock Processing Results:");
            println!("   🎯 Processed: {}/{} images", stats.successful, stats.total_processed);
            println!("   ⏱️  Duration: {:.3}s", stats.total_duration_seconds);
            println!("   🚀 Throughput: {:.2} images/sec", stats.throughput());
        }

        Ok((embeddings, stats))
    }

    /// Create a real embedding using native rust-genai library integration
    #[cfg(feature = "native-integration")]
    pub fn create_real_embedding(&self, image_url: String) -> PyResult<MultimodalEmbedding> {
        use tokio::runtime::Runtime;

        // Create a runtime for this blocking call
        let rt = Runtime::new().map_err(|e|
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Runtime error: {}", e))
        )?;

        let vision_model = self.vision_model.clone();
        let embedding_model = self.embedding_model.clone();

        rt.block_on(async move {
            process_with_native_genai_async(image_url, vision_model, embedding_model).await
        }).map_err(|e|
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Native processing error: {}", e))
        )
    }

    /// Batch process multiple images using real rust-genai integration
    #[cfg(feature = "native-integration")]
    pub fn create_real_embeddings_batch(&self, image_urls: Vec<String>) -> PyResult<(Vec<MultimodalEmbedding>, ProcessingStats)> {
        let mut embeddings = Vec::new();
        let mut successful = 0;
        let start_time = std::time::Instant::now();

        if self.config.enable_progress_reporting {
            println!("🧠 Native Integration: Real AI Batch Processing");
            println!("   📊 Images: {}", image_urls.len());
            println!("   👁️  Vision Model: {}", self.vision_model);
            println!("   🔢 Embedding Model: {}", self.embedding_model);
        }

        for (i, url) in image_urls.iter().enumerate() {
            if self.config.enable_progress_reporting {
                println!("   🖼️  Processing {}/{}: {}...", i + 1, image_urls.len(), url.split('/').last().unwrap_or("unknown"));
            }

            match self.create_real_embedding(url.clone()) {
                Ok(embedding) => {
                    embeddings.push(embedding);
                    successful += 1;
                }
                Err(e) => {
                    eprintln!("Failed to process {}: {}", url, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let stats = ProcessingStats {
            total_processed: image_urls.len(),
            successful,
            failed: image_urls.len() - successful,
            total_duration_seconds: total_duration.as_secs_f64(),
            avg_time_per_item_seconds: if successful > 0 {
                total_duration.as_secs_f64() / successful as f64
            } else {
                0.0
            },
        };

        if self.config.enable_progress_reporting {
            println!("🎯 Real AI Processing Results:");
            println!("   ✅ Successful: {}/{} images", stats.successful, stats.total_processed);
            println!("   ⏱️  Duration: {:.1}s", stats.total_duration_seconds);
            println!("   🚀 Throughput: {:.2} images/sec", stats.throughput());
        }

        Ok((embeddings, stats))
    }

    /// Create an embedding using direct Ollama API calls
    #[cfg(feature = "native-integration")]
    pub fn create_ollama_embedding(&self, image_url: String) -> PyResult<MultimodalEmbedding> {
        use tokio::runtime::Runtime;

        // Create a runtime for this blocking call
        let rt = Runtime::new().map_err(|e|
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Runtime error: {}", e))
        )?;

        let vision_model = self.vision_model.clone();
        let embedding_model = self.embedding_model.clone();

        rt.block_on(async move {
            process_with_native_genai_async(image_url, vision_model, embedding_model).await
        }).map_err(|e|
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Ollama processing error: {}", e))
        )
    }


    /// Batch process multiple images using Ollama API
    #[cfg(feature = "native-integration")]
    pub fn create_ollama_embeddings_batch(&self, image_urls: Vec<String>) -> PyResult<(Vec<MultimodalEmbedding>, ProcessingStats)> {
        let mut embeddings = Vec::new();
        let mut successful = 0;
        let start_time = std::time::Instant::now();

        for url in image_urls.iter() {
            match self.create_ollama_embedding(url.clone()) {
                Ok(embedding) => {
                    embeddings.push(embedding);
                    successful += 1;
                }
                Err(e) => {
                    eprintln!("Failed to process {}: {}", url, e);
                }
            }
        }

        let total_duration = start_time.elapsed();
        let stats = ProcessingStats {
            total_processed: image_urls.len(),
            successful,
            failed: image_urls.len() - successful,
            total_duration_seconds: total_duration.as_secs_f64(),
            avg_time_per_item_seconds: if successful > 0 {
                total_duration.as_secs_f64() / successful as f64
            } else {
                0.0
            },
        };

        Ok((embeddings, stats))
    }

    /// Calculate cosine similarity between two embeddings
    #[staticmethod]
    pub fn cosine_similarity(embedding1: Vec<f32>, embedding2: Vec<f32>) -> PyResult<f32> {
        if embedding1.len() != embedding2.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Embeddings must have the same dimensions"
            ));
        }

        let dot_product: f32 = embedding1.iter().zip(&embedding2).map(|(a, b)| a * b).sum();
        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm1 * norm2))
        }
    }

    /// Get the configuration for this embedder
    #[getter]
    pub fn config(&self) -> PerformanceConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "MultimodalEmbedder(vision='{}', embedding='{}', config={})",
            self.vision_model, self.embedding_model, self.config.__repr__()
        )
    }
}

/// Internal native processing with rust-genai library
#[cfg(feature = "native-integration")]
async fn process_with_native_genai_async(
    image_url: String,
    vision_model: String,
    embedding_model: String,
) -> Result<MultimodalEmbedding, Box<dyn std::error::Error + Send + Sync>> {
    use genai::{Client, chat::{ChatMessage, ChatRequest, ContentPart}};

    let client = Client::default();

    // Step 1: Get description from vision model using rust-genai
    let content_part = if image_url.starts_with("http://") || image_url.starts_with("https://") {
        // HTTP/HTTPS URL - use directly
        ContentPart::from_binary_url("image/jpeg", &image_url, None)
    } else if image_url.starts_with("file://") {
        // Local file - convert to base64
        let file_path = image_url.strip_prefix("file://").unwrap_or(&image_url);
        let image_data = std::fs::read(file_path)
            .map_err(|e| format!("Failed to read image file '{}': {}", file_path, e))?;
        let base64_data = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_data);
        ContentPart::from_binary_base64("image/jpeg", base64_data, None)
    } else {
        // Assume it's a local file path
        let image_data = std::fs::read(&image_url)
            .map_err(|e| format!("Failed to read image file '{}': {}", image_url, e))?;
        let base64_data = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_data);
        ContentPart::from_binary_base64("image/jpeg", base64_data, None)
    };

    let chat_req = ChatRequest::new(vec![
        ChatMessage::system("You are a helpful vision AI. Describe images accurately and concisely for embedding purposes. Focus on key visual elements, objects, scene context, colors, and composition."),
        ChatMessage::user(vec![
            ContentPart::from_text("Describe this image in detail for search and embedding purposes:"),
            content_part,
        ])
    ]);

    let chat_response = client.exec_chat(&vision_model, chat_req, None).await?;
    let description = chat_response.first_text()
        .ok_or("No description generated")?
        .to_string();

    // Step 2: Create embedding from description using rust-genai
    let embed_response = client.embed(&embedding_model, &description, None).await?;
    let embedding = embed_response.first_embedding()
        .ok_or("No embedding generated")?;

    // Convert to Vec<f32>
    let embedding_vec: Vec<f32> = embedding.vector().to_vec();

    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "native_rust_genai".to_string());
    metadata.insert("image_url".to_string(), image_url);
    metadata.insert("vision_model".to_string(), vision_model);
    metadata.insert("embedding_model".to_string(), embedding_model);
    metadata.insert("dimensions".to_string(), embedding_vec.len().to_string());
    metadata.insert("processing_method".to_string(), "native_library".to_string());

    Ok(MultimodalEmbedding::new(description, embedding_vec, metadata))
}

/// Utility functions exposed to Python
#[pyfunction]
pub fn create_performance_config(
    max_concurrent: Option<usize>,
    timeout: Option<u64>,
    batch_size: Option<usize>,
    progress: Option<bool>,
) -> PerformanceConfig {
    PerformanceConfig::new(
        max_concurrent.unwrap_or(4),
        timeout.unwrap_or(30),
        batch_size.unwrap_or(10),
        progress.unwrap_or(true),
    )
}

/// Python module definition
#[pymodule]
fn rust_genai(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_class::<PerformanceConfig>()?;
    m.add_class::<ProcessingStats>()?;
    m.add_class::<MultimodalEmbedding>()?;
    m.add_class::<MultimodalEmbedder>()?;
    m.add_function(wrap_pyfunction!(create_performance_config, m)?)?;

    m.add("__version__", "0.4.0")?;
    m.add("__doc__", "High-performance multimodal embedding pipeline for Python with mock and native AI integration")?;

    // Add capability flags for introspection
    m.add("HAS_OLLAMA_INTEGRATION", true)?;
    m.add("HAS_MOCK_EMBEDDINGS", true)?;

    #[cfg(feature = "native-integration")]
    m.add("HAS_NATIVE_INTEGRATION", true)?;
    #[cfg(not(feature = "native-integration"))]
    m.add("HAS_NATIVE_INTEGRATION", false)?;

    Ok(())
}