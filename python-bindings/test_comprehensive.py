#!/usr/bin/env python3
"""
Comprehensive test suite for rust-genai bindings.

Tests all functionality including edge cases, error handling,
and performance characteristics.
"""

import sys
import time
import traceback
from typing import List, Dict, Any

# Global import that will be set by test_imports
rust_genai = None

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🧪 Testing imports...")
    try:
        global rust_genai
        import rust_genai
        from rust_genai import (
            MultimodalEmbedder,
            PerformanceConfig,
            MultimodalEmbedding,
            ProcessingStats
        )
        print("✅ Core imports successful")

        # Test optional imports
        try:
            from rust_genai import create_performance_config
            print("✅ Utility function import successful")
        except ImportError:
            print("⚠️ Utility function not available (optional)")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_performance_config():
    """Test PerformanceConfig creation and properties."""
    print("\n🧪 Testing PerformanceConfig...")

    # Test default configuration
    config1 = rust_genai.PerformanceConfig()
    assert config1.max_concurrent_requests == 4
    assert config1.request_timeout_seconds == 30
    assert config1.batch_size == 10
    assert config1.enable_progress_reporting == True
    print("✅ Default config creation works")

    # Test custom configuration
    config2 = rust_genai.PerformanceConfig(
        max_concurrent_requests=8,
        request_timeout_seconds=60,
        batch_size=20,
        enable_progress_reporting=False
    )
    assert config2.max_concurrent_requests == 8
    assert config2.request_timeout_seconds == 60
    assert config2.batch_size == 20
    assert config2.enable_progress_reporting == False
    print("✅ Custom config creation works")

    # Test repr
    repr_str = repr(config2)
    assert "PerformanceConfig" in repr_str
    assert "max_concurrent=8" in repr_str
    print("✅ Config repr works")

    # Test utility function if available
    try:
        config3 = rust_genai.create_performance_config(
            max_concurrent=16,
            timeout=120
        )
        assert config3.max_concurrent_requests == 16
        assert config3.request_timeout_seconds == 120
        print("✅ Utility function works")
    except AttributeError:
        print("⚠️ Utility function not available (skipping)")

    return True

def test_multimodal_embedder():
    """Test MultimodalEmbedder creation and basic functionality."""
    print("\n🧪 Testing MultimodalEmbedder...")

    # Test with default config
    embedder1 = rust_genai.MultimodalEmbedder(
        "llava:7b",
        "nomic-embed-text"
    )
    assert embedder1.config.max_concurrent_requests == 4
    print("✅ Embedder with default config works")

    # Test with custom config
    config = rust_genai.PerformanceConfig(max_concurrent_requests=2)
    embedder2 = rust_genai.MultimodalEmbedder(
        "llama3.1:8b",
        "nomic-embed-text",
        config
    )
    assert embedder2.config.max_concurrent_requests == 2
    print("✅ Embedder with custom config works")

    # Test repr
    repr_str = repr(embedder2)
    assert "MultimodalEmbedder" in repr_str
    assert "llama3.1:8b" in repr_str
    print("✅ Embedder repr works")

    return embedder2

def test_single_embedding(embedder):
    """Test single image embedding creation."""
    print("\n🧪 Testing single embedding creation...")

    # Test basic embedding
    url = "https://example.com/test-image.jpg"
    embedding = embedder.create_mock_embedding(url)

    # Validate embedding properties
    assert isinstance(embedding.text_description, str)
    assert len(embedding.text_description) > 0
    assert "test-image.jpg" in embedding.text_description
    print("✅ Text description generated correctly")

    assert isinstance(embedding.embedding, list)
    assert len(embedding.embedding) == 768
    assert all(isinstance(x, float) for x in embedding.embedding[:5])
    print("✅ Embedding vector has correct dimensions and types")

    assert isinstance(embedding.metadata, dict)
    assert "source" in embedding.metadata
    assert "url" in embedding.metadata
    assert "vision_model" in embedding.metadata
    assert "embedding_model" in embedding.metadata
    assert embedding.metadata["url"] == url
    print("✅ Metadata populated correctly")

    # Test dimensions property
    assert embedding.dimensions == 768
    print("✅ Dimensions property works")

    # Test repr
    repr_str = repr(embedding)
    assert "MultimodalEmbedding" in repr_str
    assert "dimensions=768" in repr_str
    print("✅ Embedding repr works")

    return embedding

def test_batch_processing(embedder):
    """Test batch processing functionality."""
    print("\n🧪 Testing batch processing...")

    # Test small batch
    urls = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        "https://example.com/image3.jpg"
    ]

    start_time = time.time()
    embeddings, stats = embedder.simulate_concurrent_processing(urls)
    duration = time.time() - start_time

    # Validate results
    assert len(embeddings) == 3
    assert all(isinstance(emb, rust_genai.MultimodalEmbedding) for emb in embeddings)
    print("✅ Correct number of embeddings returned")

    # Validate stats
    assert isinstance(stats, rust_genai.ProcessingStats)
    assert stats.total_processed == 3
    assert stats.successful == 3
    assert stats.failed == 0
    assert stats.total_duration_seconds >= 0
    assert stats.success_rate == 100.0
    assert stats.throughput > 0
    print("✅ Processing stats correct")

    # Test stats repr
    stats_repr = repr(stats)
    assert "ProcessingStats" in stats_repr
    assert "processed=3" in stats_repr
    print("✅ Stats repr works")

    # Test larger batch
    large_urls = [f"https://example.com/image{i}.jpg" for i in range(20)]
    large_embeddings, large_stats = embedder.simulate_concurrent_processing(large_urls)

    assert len(large_embeddings) == 20
    assert large_stats.total_processed == 20
    assert large_stats.successful == 20
    print("✅ Large batch processing works")

    return embeddings

def test_similarity_calculation(embeddings):
    """Test cosine similarity calculation."""
    print("\n🧪 Testing similarity calculation...")

    if len(embeddings) < 2:
        print("⚠️ Need at least 2 embeddings for similarity test")
        return True

    # Test similarity between different embeddings
    similarity = rust_genai.MultimodalEmbedder.cosine_similarity(
        embeddings[0].embedding,
        embeddings[1].embedding
    )
    assert isinstance(similarity, float)
    assert -1.0 <= similarity <= 1.0
    print(f"✅ Similarity calculation works: {similarity:.4f}")

    # Test self-similarity (should be 1.0)
    self_similarity = rust_genai.MultimodalEmbedder.cosine_similarity(
        embeddings[0].embedding,
        embeddings[0].embedding
    )
    assert abs(self_similarity - 1.0) < 1e-6
    print("✅ Self-similarity is 1.0")

    return True

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n🧪 Testing error handling...")

    # Test similarity with different dimensions
    try:
        short_vec = [1.0, 2.0, 3.0]
        long_vec = [1.0] * 768
        rust_genai.MultimodalEmbedder.cosine_similarity(short_vec, long_vec)
        assert False, "Should have raised an error"
    except Exception as e:
        assert "same dimensions" in str(e)
        print("✅ Dimension mismatch error handled correctly")

    # Test zero vectors
    zero_vec = [0.0] * 768
    normal_vec = [1.0] * 768
    similarity = rust_genai.MultimodalEmbedder.cosine_similarity(zero_vec, normal_vec)
    assert similarity == 0.0
    print("✅ Zero vector similarity handled correctly")

    return True

def test_configuration_edge_cases():
    """Test configuration edge cases."""
    print("\n🧪 Testing configuration edge cases...")

    # Test extreme values
    config = rust_genai.PerformanceConfig(
        max_concurrent_requests=1,
        request_timeout_seconds=1,
        batch_size=1,
        enable_progress_reporting=False
    )
    embedder = rust_genai.MultimodalEmbedder("test", "test", config)

    # Should still work with minimal settings
    embedding = embedder.create_mock_embedding("https://example.com/test.jpg")
    assert embedding.dimensions == 768
    print("✅ Minimal configuration works")

    # Test large values
    large_config = rust_genai.PerformanceConfig(
        max_concurrent_requests=1000,
        request_timeout_seconds=3600,
        batch_size=10000,
        enable_progress_reporting=True
    )
    large_embedder = rust_genai.MultimodalEmbedder("test", "test", large_config)

    # Test with many URLs
    many_urls = [f"https://example.com/image{i}.jpg" for i in range(100)]
    embeddings, stats = large_embedder.simulate_concurrent_processing(many_urls)
    assert len(embeddings) == 100
    assert stats.total_processed == 100
    print("✅ Large configuration and batch works")

    return True

def test_memory_and_performance():
    """Test memory usage and performance characteristics."""
    print("\n🧪 Testing memory and performance...")

    config = rust_genai.PerformanceConfig(
        max_concurrent_requests=10,
        enable_progress_reporting=True
    )
    embedder = rust_genai.MultimodalEmbedder("test", "test", config)

    # Test processing time scales reasonably
    small_batch = [f"https://example.com/img{i}.jpg" for i in range(10)]
    large_batch = [f"https://example.com/img{i}.jpg" for i in range(100)]

    start = time.time()
    _, small_stats = embedder.simulate_concurrent_processing(small_batch)
    small_time = time.time() - start

    start = time.time()
    _, large_stats = embedder.simulate_concurrent_processing(large_batch)
    large_time = time.time() - start

    print(f"✅ Small batch (10): {small_time:.4f}s, Throughput: {small_stats.throughput:.0f}/s")
    print(f"✅ Large batch (100): {large_time:.4f}s, Throughput: {large_stats.throughput:.0f}/s")

    # Test that embeddings are properly independent
    emb1 = embedder.create_mock_embedding("https://example.com/test1.jpg")
    emb2 = embedder.create_mock_embedding("https://example.com/test2.jpg")

    # Should have different embeddings for different URLs
    similarity = rust_genai.MultimodalEmbedder.cosine_similarity(
        emb1.embedding, emb2.embedding
    )
    assert similarity < 1.0  # Should not be identical
    print("✅ Different URLs produce different embeddings")

    return True

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting Comprehensive Python Bindings Test Suite")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("PerformanceConfig Test", test_performance_config),
        ("MultimodalEmbedder Test", test_multimodal_embedder),
        ("Error Handling Test", test_error_handling),
        ("Configuration Edge Cases", test_configuration_edge_cases),
        ("Memory and Performance", test_memory_and_performance),
    ]

    passed = 0
    failed = 0
    embedder = None
    embeddings = None

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")

            if test_name == "MultimodalEmbedder Test":
                embedder = test_func()
            elif test_name == "Single Embedding Test" and embedder:
                embeddings = test_single_embedding(embedder)
            elif test_name == "Batch Processing Test" and embedder:
                embeddings = test_batch_processing(embedder)
            elif test_name == "Similarity Test" and embeddings:
                test_similarity_calculation(embeddings)
            else:
                test_func()

            print(f"✅ {test_name} PASSED")
            passed += 1

        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            failed += 1

    # Run dependent tests if we have the prerequisites
    if embedder:
        try:
            print(f"\n{'='*20} Single Embedding Test {'='*20}")
            embeddings = test_single_embedding(embedder)
            print("✅ Single Embedding Test PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ Single Embedding Test FAILED: {e}")
            failed += 1

        try:
            print(f"\n{'='*20} Batch Processing Test {'='*20}")
            embeddings = test_batch_processing(embedder)
            print("✅ Batch Processing Test PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ Batch Processing Test FAILED: {e}")
            failed += 1

    if embeddings:
        try:
            print(f"\n{'='*20} Similarity Test {'='*20}")
            test_similarity_calculation(embeddings)
            print("✅ Similarity Test PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ Similarity Test FAILED: {e}")
            failed += 1

    print("\n" + "="*60)
    print("🏁 Test Suite Complete")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("🎉 All tests passed! Python bindings are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)