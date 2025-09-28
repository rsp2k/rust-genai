# rust-genai

High-performance Python bindings for the rust-genai multimodal embedding pipeline.

## Features

🚀 **High Performance**: 2-4x faster than sequential processing
🌊 **Streaming Support**: Memory-efficient for large datasets
⚙️ **Configurable**: Flexible rate limiting and batch processing
🔧 **Simple API**: Easy-to-use Python interface

## Quick Start

```python
import asyncio
from rust_genai import MultimodalEmbedder, PerformanceConfig

async def main():
    # Configure for optimal performance
    config = PerformanceConfig(
        max_concurrent_requests=4,
        request_timeout_seconds=30,
        batch_size=10
    )

    # Create embedder
    embedder = MultimodalEmbedder(
        vision_model="llava:7b",
        embedding_model="nomic-embed-text",
        config=config
    )

    # Process images concurrently
    image_urls = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
    ]

    embeddings, stats = await embedder.embed_images_concurrent(image_urls)

    print(f"Processed {stats.successful}/{stats.total_processed} images")
    print(f"Throughput: {stats.throughput:.2f} images/sec")

    for embedding in embeddings:
        print(f"Embedding: {embedding.dimensions} dimensions")
        print(f"Description: {embedding.text_description[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

## Installation

### From Source

```bash
# Install Rust if you haven't already
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin for building Python extensions
pip install maturin

# Clone and build
git clone https://github.com/rsp2k/rust-genai.git
cd rust-genai/python-bindings
maturin develop
```

### Using pip (when published)

```bash
pip install rust-genai
```

## API Reference

### MultimodalEmbedder

The main class for processing images into embeddings.

```python
embedder = MultimodalEmbedder(
    vision_model: str,
    embedding_model: str,
    config: Optional[PerformanceConfig] = None
)
```

#### Methods

- `embed_image_url(image_url, additional_context=None)` - Process single image
- `embed_images_concurrent(image_urls)` - Process multiple images concurrently
- `embed_images_streaming(image_urls, callback)` - Stream process large batches
- `cosine_similarity(embedding1, embedding2)` - Calculate similarity between embeddings

### PerformanceConfig

Configuration for optimizing performance.

```python
config = PerformanceConfig(
    max_concurrent_requests=4,      # Concurrent vision API calls
    request_timeout_seconds=30,     # Timeout per request
    batch_size=10,                  # Streaming batch size
    enable_progress_reporting=True  # Show progress logs
)
```

### MultimodalEmbedding

Result object containing the embedding and metadata.

**Properties:**
- `text_description: str` - Generated image description
- `embedding: List[float]` - Embedding vector
- `metadata: Dict[str, str]` - Processing metadata
- `dimensions: int` - Embedding vector size

### ProcessingStats

Performance statistics from batch processing.

**Properties:**
- `total_processed: int` - Total images attempted
- `successful: int` - Successfully processed images
- `failed: int` - Failed images
- `throughput: float` - Images processed per second
- `success_rate: float` - Success percentage

## Examples

### Sequential vs Concurrent Performance

```python
import time
import asyncio
from rust_genai import MultimodalEmbedder

async def compare_performance():
    embedder = MultimodalEmbedder("llava:7b", "nomic-embed-text")

    image_urls = [f"https://example.com/image{i}.jpg" for i in range(10)]

    # Concurrent processing
    start = time.time()
    results, stats = await embedder.embed_images_concurrent(image_urls)
    concurrent_time = time.time() - start

    print(f"Concurrent: {len(results)} images in {concurrent_time:.2f}s")
    print(f"Throughput: {stats.throughput:.2f} images/sec")
```

### Streaming for Large Datasets

```python
async def process_large_dataset():
    embedder = MultimodalEmbedder("llava:7b", "nomic-embed-text")

    # Thousands of images
    large_dataset = [f"https://dataset.com/image{i}.jpg" for i in range(1000)]

    def process_result(embedding):
        # Save to database, vector store, etc.
        print(f"Processed: {embedding.dimensions} dims")

    stats = await embedder.embed_images_streaming(large_dataset, process_result)
    print(f"Streaming completed: {stats.successful} images")
```

## Performance Characteristics

- **Concurrent Processing**: 2-4x faster than sequential
- **Memory Usage**: Constant memory usage with streaming
- **Throughput**: 5-20 images/sec depending on models and hardware
- **Reliability**: Graceful error handling and recovery

## License

MIT OR Apache-2.0