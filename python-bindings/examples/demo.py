#!/usr/bin/env python3
"""
Example script for rust-genai Python bindings.

This script shows the Python API with mock embeddings for development.
Run this after building the Python extension.
"""

def main():
    try:
        # Import our Rust-based Python module
        import rust_genai

        print("🚀 Rust-Python Multimodal Embedding Example")
        print("===========================================\n")

        # Create performance configuration
        config = rust_genai.PerformanceConfig(
            max_concurrent_requests=8,
            request_timeout_seconds=45,
            batch_size=15,
            enable_progress_reporting=True
        )

        print(f"📊 Configuration: {config}")

        # Create embedder
        embedder = rust_genai.MultimodalEmbedder(
            vision_model="llava:7b",
            embedding_model="nomic-embed-text",
            config=config
        )

        print(f"🤖 Embedder: {embedder}")
        print(f"🔧 Config: {embedder.config}\n")

        # Test single embedding
        print("🖼️  Testing single embedding:")
        test_url = "https://example.com/nature-boardwalk.jpg"
        embedding = embedder.create_mock_embedding(test_url)

        print(f"   📏 Dimensions: {embedding.dimensions}")
        print(f"   📝 Description: {embedding.text_description[:80]}...")
        print(f"   📋 Metadata keys: {list(embedding.metadata.keys())}\n")

        # Test batch processing
        print("🚀 Testing batch processing:")
        test_images = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg",
            "https://example.com/image4.jpg",
            "https://example.com/image5.jpg",
        ]

        embeddings, stats = embedder.create_mock_embeddings_batch(test_images)

        print(f"\n📊 Processing Results:")
        print(f"   {stats}")
        print(f"   📈 Throughput: {stats.throughput:.0f} images/sec")
        print(f"   ✅ Success Rate: {stats.success_rate:.1f}%\n")

        # Test similarity calculation
        if len(embeddings) >= 2:
            print("🔍 Testing similarity calculation:")
            similarity = rust_genai.MultimodalEmbedder.cosine_similarity(
                embeddings[0].embedding,
                embeddings[1].embedding
            )
            print(f"   📊 Similarity between first two embeddings: {similarity:.4f}\n")

        # Show individual embedding details
        print("📋 Embedding Details:")
        for i, emb in enumerate(embeddings[:2]):  # Show first 2
            print(f"   {i+1}. {emb}")
            print(f"      🔗 URL: {emb.metadata.get('url', 'unknown')}")
            print(f"      🤖 Vision: {emb.metadata.get('vision_model', 'unknown')}")
            print(f"      📊 Embedding: {emb.metadata.get('embedding_model', 'unknown')}")

        print("\n✅ Python bindings example completed successfully!")
        print("🎯 Ready for integration with live Rust pipeline!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 To build the module, run: maturin develop")
        print("   Make sure you have maturin installed: pip install maturin")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()