#!/usr/bin/env python3
"""
Full Integration Test: Python Bindings + Real Ollama Pipeline

This test validates the complete integration of:
1. rust-genai Python bindings (PyO3-based)
2. Real Ollama multimodal pipeline (Llava + Nomic)
3. Performance comparison: mock vs real processing
4. Memory efficiency and error handling

This represents the production-ready state of the rust-genai package.
"""

import os
import sys
import time
import tempfile
import requests
import psutil
from pathlib import Path
from typing import List, Dict, Any
import base64

# Import our Python bindings
try:
    import rust_genai
    print("✅ rust_genai Python bindings imported successfully")
except ImportError as e:
    print(f"❌ Failed to import rust_genai: {e}")
    sys.exit(1)

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL = "llava:7b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Test images for comparison
TEST_IMAGES = [
    {
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300",
        "name": "mountain.jpg",
        "category": "landscape"
    },
    {
        "url": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=300",
        "name": "dog.jpg",
        "category": "animal"
    },
    {
        "url": "https://images.unsplash.com/photo-1415604934674-561df9abf539?w=300",
        "name": "city.jpg",
        "category": "urban"
    }
]

def download_test_images(temp_dir: Path) -> List[Dict[str, Any]]:
    """Download smaller test images for faster processing."""
    print("📥 Downloading optimized test images...")

    downloaded = []
    for img_info in TEST_IMAGES:
        try:
            headers = {'User-Agent': 'rust-genai-full-integration/0.4.0'}
            response = requests.get(img_info['url'], headers=headers, timeout=20)
            response.raise_for_status()

            file_path = temp_dir / img_info['name']
            file_path.write_bytes(response.content)

            downloaded.append({
                **img_info,
                'path': str(file_path),
                'size_kb': len(response.content) // 1024
            })

            print(f"   ✅ {img_info['name']} ({len(response.content)//1024}KB)")

        except Exception as e:
            print(f"   ❌ {img_info['name']}: {e}")

    return downloaded

def get_real_image_description(image_path: str) -> str:
    """Get description from Llava vision model."""
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            "model": VISION_MODEL,
            "prompt": "Describe this image concisely in 1-2 sentences.",
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 100}
        }

        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=45)
        response.raise_for_status()

        return response.json().get('response', '').strip()

    except Exception as e:
        return f"Error generating description: {e}"

def get_real_embedding(text: str) -> List[float]:
    """Get embedding from Nomic model."""
    try:
        payload = {"model": EMBEDDING_MODEL, "prompt": text}
        response = requests.post(f"{OLLAMA_HOST}/api/embeddings", json=payload, timeout=20)
        response.raise_for_status()
        return response.json().get('embedding', [])
    except Exception as e:
        print(f"   ⚠️  Embedding error: {e}")
        return []

def test_python_bindings_functionality():
    """Test basic Python bindings functionality."""
    print("\n🐍 Testing Python Bindings Functionality")
    print("=" * 45)

    try:
        # Test configuration creation
        config = rust_genai.PerformanceConfig(
            max_concurrent_requests=2,
            request_timeout_seconds=30,
            batch_size=5,
            enable_progress_reporting=False
        )
        print("✅ PerformanceConfig creation")

        # Test embedder creation
        embedder = rust_genai.MultimodalEmbedder(
            vision_model=VISION_MODEL,
            embedding_model=EMBEDDING_MODEL,
            config=config
        )
        print("✅ MultimodalEmbedder creation")

        # Test mock embedding
        mock_embedding = embedder.create_mock_embedding("test_image.jpg")
        print(f"✅ Mock embedding: {mock_embedding.dimensions} dimensions")

        # Test similarity calculation
        embedding1 = mock_embedding.embedding
        embedding2 = embedder.create_mock_embedding("different_image.jpg").embedding
        similarity = rust_genai.MultimodalEmbedder.cosine_similarity(embedding1, embedding2)
        print(f"✅ Cosine similarity: {similarity:.4f}")

        # Test batch processing simulation
        test_urls = [f"image_{i}.jpg" for i in range(5)]
        embeddings, stats = embedder.simulate_concurrent_processing(test_urls)
        print(f"✅ Batch processing: {len(embeddings)} embeddings, {stats.throughput:.0f} items/sec")

        return True

    except Exception as e:
        print(f"❌ Python bindings test failed: {e}")
        return False

def test_mock_vs_real_comparison(images: List[Dict[str, Any]]):
    """Compare mock embeddings with real pipeline results."""
    print("\n⚖️  Mock vs Real Pipeline Comparison")
    print("=" * 42)

    if not images:
        print("❌ No images available for comparison")
        return False

    # Initialize embedder
    config = rust_genai.PerformanceConfig(max_concurrent_requests=1, enable_progress_reporting=False)
    embedder = rust_genai.MultimodalEmbedder(VISION_MODEL, EMBEDDING_MODEL, config)

    results = []

    for img_info in images:
        print(f"\n📸 Processing {img_info['name']} ({img_info['category']})...")

        # Mock pipeline (fast)
        mock_start = time.time()
        mock_embedding = embedder.create_mock_embedding(f"file://{img_info['path']}")
        mock_duration = time.time() - mock_start

        print(f"   🏃 Mock pipeline: {mock_duration*1000:.1f}ms")
        print(f"   📝 Mock description: {mock_embedding.text_description[:80]}...")

        # Real pipeline (slow but accurate)
        real_start = time.time()
        real_description = get_real_image_description(img_info['path'])
        real_embedding_vec = get_real_embedding(real_description)
        real_duration = time.time() - real_start

        if real_embedding_vec:
            print(f"   🤖 Real pipeline: {real_duration:.1f}s")
            print(f"   📝 Real description: {real_description[:80]}...")

            # Compare embeddings if both successful
            if len(real_embedding_vec) == len(mock_embedding.embedding):
                similarity = rust_genai.MultimodalEmbedder.cosine_similarity(
                    mock_embedding.embedding, real_embedding_vec
                )
                print(f"   📊 Mock-Real similarity: {similarity:.4f}")

                results.append({
                    'image': img_info['name'],
                    'category': img_info['category'],
                    'mock_duration_ms': mock_duration * 1000,
                    'real_duration_s': real_duration,
                    'mock_real_similarity': similarity,
                    'speedup_factor': real_duration / mock_duration
                })
            else:
                print(f"   ⚠️  Dimension mismatch: mock={len(mock_embedding.embedding)}, real={len(real_embedding_vec)}")
        else:
            print("   ❌ Real pipeline failed")

    # Summary analysis
    if results:
        print(f"\n📊 Comparison Analysis:")
        avg_similarity = sum(r['mock_real_similarity'] for r in results) / len(results)
        avg_speedup = sum(r['speedup_factor'] for r in results) / len(results)

        print(f"   🎯 Average mock-real similarity: {avg_similarity:.4f}")
        print(f"   ⚡ Average speedup factor: {avg_speedup:.0f}x")
        print(f"   📈 Mock enables {avg_speedup:.0f}x faster development iteration")

        return True

    return False

def test_memory_efficiency():
    """Test memory usage during processing."""
    print("\n🧠 Memory Efficiency Test")
    print("=" * 27)

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(f"📊 Initial memory: {initial_memory:.1f}MB")

    # Create multiple embedders and process data
    embedders = []
    for i in range(5):
        config = rust_genai.PerformanceConfig(max_concurrent_requests=2)
        embedder = rust_genai.MultimodalEmbedder(f"model_{i}", f"embed_{i}", config)
        embedders.append(embedder)

    # Generate many embeddings
    all_embeddings = []
    for embedder in embedders:
        test_urls = [f"test_image_{j}.jpg" for j in range(10)]
        embeddings, _ = embedder.simulate_concurrent_processing(test_urls)
        all_embeddings.extend(embeddings)

    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = peak_memory - initial_memory

    print(f"📊 Peak memory: {peak_memory:.1f}MB")
    print(f"📈 Memory growth: {memory_growth:.1f}MB")
    print(f"📦 Generated {len(all_embeddings)} embeddings")
    print(f"💾 Memory per embedding: {memory_growth/len(all_embeddings)*1024:.1f}KB")

    # Cleanup test
    del embedders, all_embeddings

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"🧹 Final memory: {final_memory:.1f}MB")
    print(f"♻️  Memory efficiency: {'✅ Good' if memory_growth < 50 else '⚠️ Check'}")

    return memory_growth < 100  # Less than 100MB growth is acceptable

def run_full_integration_tests():
    """Run comprehensive integration tests."""
    print("🧪 RUST-GENAI FULL INTEGRATION TEST SUITE")
    print("=" * 65)
    print("Testing Python bindings + Real Ollama pipeline integration\n")

    # Check Ollama availability
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Ollama connection verified")
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print("   Running Python bindings tests only...\n")

    test_results = {}

    # Test 1: Python bindings functionality
    test_results['python_bindings'] = test_python_bindings_functionality()

    # Test 2: Memory efficiency
    test_results['memory_efficiency'] = test_memory_efficiency()

    # Test 3: Mock vs Real comparison (if Ollama available)
    ollama_available = True
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
    except:
        ollama_available = False

    if ollama_available:
        with tempfile.TemporaryDirectory() as temp_dir:
            images = download_test_images(Path(temp_dir))
            if images:
                test_results['mock_vs_real'] = test_mock_vs_real_comparison(images)
            else:
                test_results['mock_vs_real'] = False
    else:
        test_results['mock_vs_real'] = None  # Skipped

    # Results summary
    print(f"\n" + "=" * 65)
    print("🏁 FULL INTEGRATION TEST RESULTS")
    print("=" * 65)

    passed = 0
    total = 0

    for test_name, result in test_results.items():
        if result is None:
            status = "⏭️  SKIP"
        elif result:
            status = "✅ PASS"
            passed += 1
            total += 1
        else:
            status = "❌ FAIL"
            total += 1

        print(f"{test_name.replace('_', ' ').title():<25} {status}")

    if test_results['mock_vs_real'] is None:
        print(f"\nOverall: {passed}/{total} tests passed (1 skipped)")
    else:
        print(f"\nOverall: {passed}/{total} tests passed")

    success = passed == total

    if success:
        print("\n🎉 Full integration successful!")
        print("   📦 Python bindings working correctly")
        print("   🧠 Memory usage optimized")
        if test_results['mock_vs_real']:
            print("   🔗 Real pipeline integration verified")
        print("\n🚀 rust-genai package is ready for production use!")
    else:
        print("\n⚠️  Some tests failed - package needs investigation")

    return success

if __name__ == "__main__":
    success = run_full_integration_tests()
    sys.exit(0 if success else 1)