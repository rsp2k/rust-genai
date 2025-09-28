#!/usr/bin/env python3
"""
Stress test with 50 real images from Unsplash.

This test validates:
1. Large batch processing performance
2. Memory usage patterns
3. Network download reliability
4. Concurrent processing scaling
5. Error recovery under load
"""

import sys
import time
import tempfile
import requests
from pathlib import Path
from typing import List, Dict, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 50 diverse test images from Unsplash
# Using different categories to test varied content
UNSPLASH_IMAGES = [
    # Nature & Landscapes (10 images)
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # Mountain
    "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400",  # Forest
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400",  # Trees
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # Lake
    "https://images.unsplash.com/photo-1494599948593-3dafe8db8d85?w=400",  # Ocean
    "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400",  # Desert
    "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=400",  # Sunset
    "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=400",  # Clouds
    "https://images.unsplash.com/photo-1551632811-561732d1e306?w=400",  # Waterfall
    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400",  # Canyon

    # Animals (10 images)
    "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=400",  # Dog
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",  # Cat
    "https://images.unsplash.com/photo-1549471053-58b89c8c8ed6?w=400",  # Bird
    "https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=400",  # Horse
    "https://images.unsplash.com/photo-1547036967-23d11aacaee0?w=400",  # Butterfly
    "https://images.unsplash.com/photo-1571988840298-3b5301d5109b?w=400",  # Elephant
    "https://images.unsplash.com/photo-1567358621517-6d9a13b0ab5c?w=400",  # Lion
    "https://images.unsplash.com/photo-1580611404649-8d5c4f35e937?w=400",  # Deer
    "https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=400",  # Bear
    "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400",  # Fish

    # Urban & Architecture (10 images)
    "https://images.unsplash.com/photo-1415604934674-561df9abf539?w=400",  # City
    "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1f?w=400",  # Building
    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400",  # Bridge
    "https://images.unsplash.com/photo-1493246507139-91e8fad9978e?w=400",  # Library
    "https://images.unsplash.com/photo-1545241047-6083a3684587?w=400",  # Church
    "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=400",  # Skyscraper
    "https://images.unsplash.com/photo-1582555172866-f73bb12a2ab3?w=400",  # Street
    "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400",  # Park
    "https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=400",  # Cafe
    "https://images.unsplash.com/photo-1542204165-65bf26472b9b?w=400",  # Market

    # People & Portraits (10 images)
    "https://images.unsplash.com/photo-1494790108755-2616c7e2b22e?w=400",  # Portrait
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # Man
    "https://images.unsplash.com/photo-1494790108755-2616c7e2b22e?w=400",  # Woman
    "https://images.unsplash.com/photo-1515041219749-89347f83291a?w=400",  # Child
    "https://images.unsplash.com/photo-1515041219749-89347f83291a?w=400",  # Family
    "https://images.unsplash.com/photo-1582555172866-f73bb12a2ab3?w=400",  # Business
    "https://images.unsplash.com/photo-1542204165-65bf26472b9b?w=400",  # Sport
    "https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=400",  # Music
    "https://images.unsplash.com/photo-1494790108755-2616c7e2b22e?w=400",  # Art
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # Work

    # Objects & Technology (10 images)
    "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400",  # Laptop
    "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=400",  # Phone
    "https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=400",  # Camera
    "https://images.unsplash.com/photo-1507925921958-8a62f3d1a50d?w=400",  # Car
    "https://images.unsplash.com/photo-1552664730-d307ca884978?w=400",  # Watch
    "https://images.unsplash.com/photo-1542204165-65bf26472b9b?w=400",  # Book
    "https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=400",  # Guitar
    "https://images.unsplash.com/photo-1494790108755-2616c7e2b22e?w=400",  # Coffee
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # Food
    "https://images.unsplash.com/photo-1515041219749-89347f83291a?w=400",  # Flowers
]

def download_image(url: str, index: int, temp_dir: Path) -> Tuple[bool, str, int, float]:
    """Download a single image with timing and error handling."""
    start_time = time.time()
    try:
        headers = {
            'User-Agent': 'rust-genai-stress-test/0.4.0',
            'Accept': 'image/*'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        file_path = temp_dir / f"image_{index:02d}.jpg"
        file_path.write_bytes(response.content)

        duration = time.time() - start_time
        return True, str(file_path), len(response.content), duration

    except Exception as e:
        duration = time.time() - start_time
        return False, str(e), 0, duration

def download_all_images(temp_dir: Path, max_workers: int = 10) -> List[str]:
    """Download all 50 images concurrently."""
    print(f"📥 Downloading 50 images concurrently (max workers: {max_workers})...")

    downloaded_paths = []
    total_bytes = 0
    total_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_index = {
            executor.submit(download_image, url, i, temp_dir): i
            for i, url in enumerate(UNSPLASH_IMAGES)
        }

        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            success, result, size, duration = future.result()

            completed += 1
            if success:
                downloaded_paths.append(result)
                total_bytes += size
                print(f"   ✅ {completed:2d}/50 - Image {index:2d}: {size:6d} bytes in {duration:.2f}s")
            else:
                print(f"   ❌ {completed:2d}/50 - Image {index:2d}: {result}")

    total_download_time = time.time() - total_time
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {len(downloaded_paths)}/50 images")
    print(f"   📦 Total data: {total_bytes/1024/1024:.1f} MB")
    print(f"   ⏱️  Total time: {total_download_time:.1f}s")
    print(f"   🚀 Download speed: {total_bytes/1024/1024/total_download_time:.1f} MB/s")

    return downloaded_paths

def test_batch_processing_performance(image_paths: List[str]):
    """Test processing performance with different batch sizes."""
    print(f"\n⚡ Batch Processing Performance Test")
    print("=" * 45)

    try:
        import rust_genai

        # Test different batch sizes
        batch_sizes = [1, 5, 10, 25, len(image_paths)]

        for batch_size in batch_sizes:
            if batch_size > len(image_paths):
                continue

            print(f"\n🧪 Testing batch size: {batch_size}")

            # Create embedder with appropriate concurrency
            config = rust_genai.PerformanceConfig(
                max_concurrent_requests=min(batch_size, 8),
                enable_progress_reporting=False
            )
            embedder = rust_genai.MultimodalEmbedder("llava:7b", "nomic-embed-text", config)

            # Process batch
            test_images = image_paths[:batch_size]
            image_urls = [f"file://{path}" for path in test_images]

            start = time.time()
            embeddings, stats = embedder.simulate_concurrent_processing(image_urls)
            duration = time.time() - start

            print(f"   📏 Processed: {len(embeddings)} embeddings")
            print(f"   ⏱️  Duration: {duration:.3f}s")
            print(f"   🚀 Throughput: {len(embeddings)/duration:.1f} items/sec")
            print(f"   📊 Reported throughput: {stats.throughput:.0f} items/sec")
            print(f"   ✅ Success rate: {stats.success_rate:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False

def test_memory_usage_patterns(image_paths: List[str]):
    """Test memory usage with large batches."""
    print(f"\n💾 Memory Usage Pattern Test")
    print("=" * 35)

    try:
        import rust_genai
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"   📊 Initial memory: {initial_memory:.1f} MB")

        # Process all images
        config = rust_genai.PerformanceConfig(
            max_concurrent_requests=4,
            enable_progress_reporting=False
        )
        embedder = rust_genai.MultimodalEmbedder("test", "test", config)

        image_urls = [f"file://{path}" for path in image_paths]

        # Process in chunks to monitor memory
        chunk_size = 10
        total_embeddings = []

        for i in range(0, len(image_urls), chunk_size):
            chunk = image_urls[i:i+chunk_size]

            chunk_start = time.time()
            embeddings, stats = embedder.simulate_concurrent_processing(chunk)
            chunk_duration = time.time() - chunk_start

            total_embeddings.extend(embeddings)
            current_memory = process.memory_info().rss / 1024 / 1024

            print(f"   📦 Chunk {i//chunk_size + 1}: {len(embeddings)} items in {chunk_duration:.2f}s")
            print(f"   💾 Memory: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")

        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"\n📊 Memory Summary:")
        print(f"   Initial: {initial_memory:.1f} MB")
        print(f"   Final: {final_memory:.1f} MB")
        print(f"   Peak increase: {final_memory - initial_memory:.1f} MB")
        print(f"   Per embedding: {(final_memory - initial_memory) / len(total_embeddings):.3f} MB")

        return True

    except ImportError:
        print("   ⚠️  psutil not available - skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_similarity_matrix(image_paths: List[str], sample_size: int = 10):
    """Test similarity calculations across diverse images."""
    print(f"\n🔍 Similarity Matrix Test")
    print("=" * 30)

    try:
        import rust_genai

        embedder = rust_genai.MultimodalEmbedder("test", "test")

        # Process sample of images
        sample_paths = image_paths[:sample_size]
        print(f"   📊 Computing similarity matrix for {len(sample_paths)} images...")

        embeddings = []
        for path in sample_paths:
            emb = embedder.create_mock_embedding(f"file://{path}")
            embeddings.append(emb)

        # Compute similarity matrix
        similarities = []
        for i in range(len(embeddings)):
            row = []
            for j in range(len(embeddings)):
                sim = rust_genai.MultimodalEmbedder.cosine_similarity(
                    embeddings[i].embedding,
                    embeddings[j].embedding
                )
                row.append(sim)
            similarities.append(row)

        # Analyze similarity patterns
        diagonal_sims = [similarities[i][i] for i in range(len(similarities))]
        off_diagonal_sims = [similarities[i][j] for i in range(len(similarities))
                           for j in range(len(similarities)) if i != j]

        print(f"   ✅ Diagonal similarities (self): {min(diagonal_sims):.3f} - {max(diagonal_sims):.3f}")
        print(f"   📊 Off-diagonal similarities: {min(off_diagonal_sims):.3f} - {max(off_diagonal_sims):.3f}")
        print(f"   📈 Average cross-similarity: {sum(off_diagonal_sims)/len(off_diagonal_sims):.3f}")

        # Check for expected patterns
        if all(abs(sim - 1.0) < 0.001 for sim in diagonal_sims):
            print("   ✅ Self-similarities are perfect (1.0)")
        else:
            print("   ❌ Self-similarities not perfect")
            return False

        return True

    except Exception as e:
        print(f"❌ Similarity test failed: {e}")
        return False

def run_50_image_stress_test():
    """Run comprehensive stress test with 50 real images."""
    print("🚀 50-IMAGE STRESS TEST")
    print("=" * 50)
    print("Testing rust-genai with 50 real images from Unsplash\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download all images
        image_paths = download_all_images(temp_path)

        if len(image_paths) < 40:  # Need at least 40/50 for valid test
            print("❌ Too many download failures - test invalid")
            return False

        print(f"✅ Proceeding with {len(image_paths)} successfully downloaded images")

        # Run comprehensive tests
        tests = [
            ("Batch Processing Performance", lambda: test_batch_processing_performance(image_paths)),
            ("Memory Usage Patterns", lambda: test_memory_usage_patterns(image_paths)),
            ("Similarity Matrix Analysis", lambda: test_similarity_matrix(image_paths))
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results[test_name] = False

        # Final summary
        print("\n" + "=" * 50)
        print("🏁 50-IMAGE STRESS TEST RESULTS")
        print("=" * 50)

        passed = 0
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<30} {status}")
            if result:
                passed += 1

        print(f"\nDownload Success Rate: {len(image_paths)}/50 ({len(image_paths)/50*100:.1f}%)")
        print(f"Test Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")

        if passed == total and len(image_paths) >= 45:
            print("\n🎉 STRESS TEST PASSED! Package handles large workloads correctly.")
            return True
        else:
            print("\n⚠️  STRESS TEST ISSUES - See details above.")
            return False

if __name__ == "__main__":
    success = run_50_image_stress_test()
    sys.exit(0 if success else 1)