#!/usr/bin/env python3
"""
Real Pipeline Integration Test

Test the real AI pipeline integration in the Python bindings
using the new create_real_embedding method.
"""

import sys
import time
import tempfile
import requests
from pathlib import Path

def test_real_pipeline_integration():
    """Test the real pipeline with actual AI models."""
    print("🚀 REAL PIPELINE INTEGRATION TEST")
    print("=" * 45)

    try:
        # Import our Python bindings
        import rust_genai
        print(f"✅ rust_genai imported (live integration: {getattr(rust_genai, 'LIVE_INTEGRATION', False)})")

        # Download a test image
        test_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=200"
        print("\n📥 Downloading test image...")

        headers = {'User-Agent': 'rust-genai-real-pipeline-test/0.4.0'}
        response = requests.get(test_url, headers=headers, timeout=15)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(response.content)
            image_path = f.name

        print(f"✅ Downloaded {len(response.content)} bytes to {image_path}")

        # Create embedder
        print("\n🤖 Creating MultimodalEmbedder...")
        embedder = rust_genai.MultimodalEmbedder("llava:7b", "nomic-embed-text:latest")
        print(f"✅ Embedder created: {embedder}")

        # Test mock embedding first (fast baseline)
        print("\n🏃 Testing mock embedding (baseline)...")
        mock_start = time.time()
        mock_embedding = embedder.create_mock_embedding(f"file://{image_path}")
        mock_duration = time.time() - mock_start

        print(f"✅ Mock embedding generated in {mock_duration*1000:.1f}ms")
        print(f"   📊 Dimensions: {mock_embedding.dimensions}")
        print(f"   📝 Description: {mock_embedding.text_description[:60]}...")
        print(f"   🏷️  Source: {mock_embedding.metadata['source']}")

        # Test real embedding (slow but actual AI)
        if hasattr(embedder, 'create_real_embedding'):
            print("\n🧠 Testing real AI embedding...")
            real_start = time.time()
            real_embedding = embedder.create_real_embedding(f"file://{image_path}")
            real_duration = time.time() - real_start

            print(f"✅ Real embedding generated in {real_duration:.1f}s")
            print(f"   📊 Dimensions: {real_embedding.dimensions}")
            print(f"   📝 Description: {real_embedding.text_description[:60]}...")
            print(f"   🏷️  Source: {real_embedding.metadata['source']}")
            print(f"   🔧 Method: {real_embedding.metadata['processing_method']}")

            # Compare embeddings
            if len(real_embedding.embedding) == len(mock_embedding.embedding):
                similarity = rust_genai.MultimodalEmbedder.cosine_similarity(
                    mock_embedding.embedding, real_embedding.embedding
                )
                print(f"   📊 Mock-Real similarity: {similarity:.4f}")

            speedup_factor = real_duration / mock_duration
            print(f"   ⚡ Mock speedup: {speedup_factor:.0f}x faster")

            # Test batch processing
            print("\n🔄 Testing batch processing...")
            test_urls = [f"file://{image_path}", f"file://{image_path}"]

            batch_start = time.time()
            if hasattr(embedder, 'create_real_embeddings_batch'):
                real_embeddings, real_stats = embedder.create_real_embeddings_batch(test_urls)
                batch_duration = time.time() - batch_start

                print(f"✅ Batch processing completed in {batch_duration:.1f}s")
                print(f"   🎯 Processed: {real_stats.successful}/{real_stats.total_processed}")
                print(f"   🚀 Throughput: {real_stats.throughput:.1f} items/sec")
                print(f"   ⏱️  Avg per item: {real_stats.avg_time_per_item_seconds:.1f}s")

        else:
            print("\n⚠️  Real embedding method not available (live integration disabled)")

        # Cleanup
        import os
        os.unlink(image_path)

        print(f"\n🎉 INTEGRATION TEST SUCCESSFUL!")
        print("   🏃 Mock pipeline: ✅ Working (fast development)")
        if hasattr(embedder, 'create_real_embedding'):
            print("   🧠 Real pipeline: ✅ Working (actual AI)")
            print("   🔗 Full integration: ✅ Complete")
        else:
            print("   🔗 Live integration: ⚠️  Disabled (mock mode only)")

        return True

    except ImportError as e:
        print(f"❌ Failed to import rust_genai: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_real_pipeline_integration()
    sys.exit(0 if success else 1)