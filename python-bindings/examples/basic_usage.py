#!/usr/bin/env python3
"""
Basic usage example for genai-python multimodal embeddings.

This example demonstrates the simplest way to get started with
high-performance multimodal embeddings in Python.
"""

import asyncio
import os
from rust_genai import MultimodalEmbedder, PerformanceConfig

def main():
    print("🚀 Python Multimodal Embedding Example")
    print("======================================\n")

    # Set up remote Ollama (optional)
    if not os.getenv("OLLAMA_HOST"):
        print("💡 Tip: Set OLLAMA_HOST=https://ollama.l.supported.systems for remote testing")

    # Create performance configuration
    config = PerformanceConfig(
        max_concurrent_requests=2,  # Conservative for demo
        request_timeout_seconds=45,
        batch_size=5,
        enable_progress_reporting=True
    )

    # Initialize embedder
    embedder = MultimodalEmbedder(
        vision_model="llama3.1:8b",
        embedding_model="nomic-embed-text",
        config=config
    )

    print(f"📊 Configuration: {config}")
    print(f"🤖 Embedder: {embedder}\n")

    # Test images
    test_images = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/800px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Bucephala-albeola-010.jpg/400px-Bucephala-albeola-010.jpg",
    ]

    print(f"📷 Processing {len(test_images)} test images...\n")

    try:
        # Process images concurrently using mock simulation
        embeddings, stats = embedder.simulate_concurrent_processing(test_images)

        print(f"\n✅ Results:")
        print(f"   📊 Stats: {stats}")
        print(f"   🎯 Success Rate: {stats.success_rate:.1f}%")
        print(f"   🚀 Throughput: {stats.throughput:.2f} images/sec")

        # Show embedding details
        for i, embedding in enumerate(embeddings):
            print(f"\n🖼️  Image {i+1}:")
            print(f"   📏 Dimensions: {embedding.dimensions}")
            print(f"   📝 Description: {embedding.text_description[:100]}...")

            # Show metadata
            print(f"   📋 Metadata:")
            for key, value in embedding.metadata.items():
                print(f"      {key}: {value}")

        # Demonstrate similarity calculation
        if len(embeddings) >= 2:
            similarity = MultimodalEmbedder.cosine_similarity(
                embeddings[0].embedding,
                embeddings[1].embedding
            )
            print(f"\n🔍 Similarity between images: {similarity:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("   💡 Make sure Ollama is running and models are available")

if __name__ == "__main__":
    # Set environment for remote Ollama if desired
    # os.environ["OLLAMA_HOST"] = "https://ollama.l.supported.systems"

    main()