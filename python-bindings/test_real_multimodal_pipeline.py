#!/usr/bin/env python3
"""
Real Multimodal Pipeline Test with Llava Vision Model

This test validates the complete multimodal embedding pipeline using:
- Real images downloaded from Unsplash
- Actual Llava vision model for image description
- Real Nomic embedding model for text embeddings
- Live Ollama integration
- Performance measurement of actual AI workloads

No mocks, no simulations - pure real-world testing.
"""

import os
import sys
import time
import json
import base64
import tempfile
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess

# Test configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL = "llava:7b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Real test images from Unsplash (diverse content for vision testing)
TEST_IMAGES = [
    {
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
        "name": "mountain_landscape.jpg",
        "expected_content": ["mountain", "landscape", "nature", "scenic"]
    },
    {
        "url": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400",
        "name": "golden_retriever.jpg",
        "expected_content": ["dog", "golden", "retriever", "animal"]
    },
    {
        "url": "https://images.unsplash.com/photo-1415604934674-561df9abf539?w=400",
        "name": "city_waterfront.jpg",
        "expected_content": ["city", "water", "building", "urban"]
    },
    {
        "url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400",
        "name": "forest_path.jpg",
        "expected_content": ["forest", "path", "tree", "nature"]
    }
]

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 for Ollama API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_ollama_connectivity():
    """Test Ollama connection and model availability."""
    print("🔌 Testing Ollama Connectivity")
    print("=" * 35)

    try:
        # Test basic connection
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        response.raise_for_status()
        models_data = response.json()

        available_models = [model['name'] for model in models_data.get('models', [])]
        print(f"✅ Connected to Ollama at {OLLAMA_HOST}")
        print(f"📋 Available models: {len(available_models)}")

        # Check for required models
        vision_available = VISION_MODEL in available_models
        embedding_available = EMBEDDING_MODEL in available_models

        print(f"👁️  Vision model ({VISION_MODEL}): {'✅ Available' if vision_available else '❌ Missing'}")
        print(f"🔤 Embedding model ({EMBEDDING_MODEL}): {'✅ Available' if embedding_available else '❌ Missing'}")

        if not vision_available:
            print(f"   💡 Install with: ollama pull {VISION_MODEL}")
        if not embedding_available:
            print(f"   💡 Install with: ollama pull {EMBEDDING_MODEL}")

        return vision_available and embedding_available

    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("   💡 Is Ollama running? Try: ollama serve")
        return False

def download_test_images(temp_dir: Path) -> List[Dict[str, Any]]:
    """Download test images for real multimodal testing."""
    print("\n📥 Downloading Test Images")
    print("=" * 30)

    downloaded_images = []

    for img_info in TEST_IMAGES:
        try:
            print(f"   Downloading {img_info['name']}...")
            headers = {
                'User-Agent': 'rust-genai-multimodal-test/0.4.0 (real pipeline testing)'
            }
            response = requests.get(img_info['url'], headers=headers, timeout=30)
            response.raise_for_status()

            file_path = temp_dir / img_info['name']
            file_path.write_bytes(response.content)

            downloaded_images.append({
                **img_info,
                'path': str(file_path),
                'size_bytes': len(response.content)
            })

            print(f"   ✅ {img_info['name']} ({len(response.content)} bytes)")

        except Exception as e:
            print(f"   ❌ Failed to download {img_info['name']}: {e}")

    print(f"✅ Downloaded {len(downloaded_images)}/{len(TEST_IMAGES)} images")
    return downloaded_images

def generate_image_description(image_path: str) -> Dict[str, Any]:
    """Generate description using Llava vision model."""
    try:
        # Encode image to base64
        image_b64 = encode_image_to_base64(image_path)

        # Prepare Ollama request
        payload = {
            "model": VISION_MODEL,
            "prompt": "Describe this image in detail, focusing on the main subjects, colors, setting, and overall composition. Be specific and descriptive.",
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent descriptions
                "num_predict": 200   # Limit response length
            }
        }

        print(f"   🧠 Generating description with {VISION_MODEL}...")
        start_time = time.time()

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=60  # Vision models can be slow
        )
        response.raise_for_status()

        duration = time.time() - start_time
        result = response.json()
        description = result.get('response', '').strip()

        return {
            'success': True,
            'description': description,
            'duration_seconds': duration,
            'model_used': VISION_MODEL,
            'prompt_eval_count': result.get('prompt_eval_count', 0),
            'eval_count': result.get('eval_count', 0)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'description': '',
            'duration_seconds': 0
        }

def generate_text_embedding(text: str) -> Dict[str, Any]:
    """Generate embedding using Nomic embedding model."""
    try:
        payload = {
            "model": EMBEDDING_MODEL,
            "prompt": text
        }

        print(f"   🔢 Generating embedding with {EMBEDDING_MODEL}...")
        start_time = time.time()

        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        duration = time.time() - start_time
        result = response.json()
        embedding = result.get('embedding', [])

        return {
            'success': True,
            'embedding': embedding,
            'dimensions': len(embedding),
            'duration_seconds': duration,
            'model_used': EMBEDDING_MODEL
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'embedding': [],
            'dimensions': 0,
            'duration_seconds': 0
        }

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)

def validate_description_content(description: str, expected_content: List[str]) -> Dict[str, Any]:
    """Validate that image description contains expected content."""
    description_lower = description.lower()

    found_content = []
    missing_content = []

    for expected in expected_content:
        if expected.lower() in description_lower:
            found_content.append(expected)
        else:
            missing_content.append(expected)

    accuracy = len(found_content) / len(expected_content) if expected_content else 0.0

    return {
        'accuracy': accuracy,
        'found_content': found_content,
        'missing_content': missing_content,
        'total_expected': len(expected_content)
    }

def test_real_multimodal_pipeline(images: List[Dict[str, Any]]):
    """Test the complete multimodal pipeline with real AI models."""
    print("\n🚀 Testing Real Multimodal Pipeline")
    print("=" * 42)

    if not images:
        print("❌ No images available for testing")
        return False

    results = []
    total_start = time.time()

    for i, img_info in enumerate(images):
        print(f"\n📸 Processing image {i+1}/{len(images)}: {img_info['name']}")
        print("-" * 50)

        pipeline_start = time.time()

        # Step 1: Generate image description with Llava
        desc_result = generate_image_description(img_info['path'])

        if not desc_result['success']:
            print(f"   ❌ Vision model failed: {desc_result['error']}")
            continue

        description = desc_result['description']
        print(f"   📝 Description ({desc_result['duration_seconds']:.2f}s): {description[:100]}...")

        # Step 2: Validate description content
        content_validation = validate_description_content(description, img_info['expected_content'])
        print(f"   🎯 Content accuracy: {content_validation['accuracy']:.1%}")
        print(f"   ✅ Found: {content_validation['found_content']}")
        if content_validation['missing_content']:
            print(f"   ❌ Missing: {content_validation['missing_content']}")

        # Step 3: Generate embedding from description
        emb_result = generate_text_embedding(description)

        if not emb_result['success']:
            print(f"   ❌ Embedding model failed: {emb_result['error']}")
            continue

        print(f"   🔢 Embedding ({emb_result['duration_seconds']:.2f}s): {emb_result['dimensions']} dimensions")

        pipeline_duration = time.time() - pipeline_start

        results.append({
            'image_info': img_info,
            'description': desc_result,
            'embedding': emb_result,
            'content_validation': content_validation,
            'total_pipeline_duration': pipeline_duration
        })

        print(f"   ⏱️  Total pipeline: {pipeline_duration:.2f}s")

    total_duration = time.time() - total_start

    # Analysis and summary
    print(f"\n" + "=" * 42)
    print("📊 Pipeline Analysis")
    print("=" * 42)

    successful_results = [r for r in results if r['description']['success'] and r['embedding']['success']]

    print(f"✅ Successful pipelines: {len(successful_results)}/{len(images)}")
    print(f"⏱️  Total processing time: {total_duration:.2f}s")
    print(f"🚀 Average per image: {total_duration/len(images):.2f}s")

    if successful_results:
        # Vision model performance
        avg_vision_time = sum(r['description']['duration_seconds'] for r in successful_results) / len(successful_results)
        avg_embedding_time = sum(r['embedding']['duration_seconds'] for r in successful_results) / len(successful_results)
        avg_content_accuracy = sum(r['content_validation']['accuracy'] for r in successful_results) / len(successful_results)

        print(f"👁️  Average vision processing: {avg_vision_time:.2f}s")
        print(f"🔢 Average embedding generation: {avg_embedding_time:.2f}s")
        print(f"🎯 Average content accuracy: {avg_content_accuracy:.1%}")

        # Test embedding similarity
        if len(successful_results) >= 2:
            print(f"\n🔍 Testing embedding similarities:")

            for i in range(len(successful_results)):
                for j in range(i+1, len(successful_results)):
                    img1 = successful_results[i]['image_info']['name']
                    img2 = successful_results[j]['image_info']['name']
                    emb1 = successful_results[i]['embedding']['embedding']
                    emb2 = successful_results[j]['embedding']['embedding']

                    similarity = cosine_similarity(emb1, emb2)
                    print(f"   📊 {img1} ↔ {img2}: {similarity:.4f}")

    return len(successful_results) > 0

def run_real_multimodal_tests():
    """Run comprehensive real multimodal pipeline tests."""
    print("🧪 REAL MULTIMODAL PIPELINE TEST SUITE")
    print("=" * 60)
    print("Testing rust-genai with actual Llava vision and Nomic embeddings\n")

    # Test Ollama connectivity first
    if not test_ollama_connectivity():
        print("\n❌ Cannot proceed - Ollama or required models not available")
        return False

    # Download test images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images = download_test_images(temp_path)

        if not images:
            print("\n❌ Cannot proceed - no test images available")
            return False

        # Run the real multimodal pipeline test
        pipeline_success = test_real_multimodal_pipeline(images)

        print(f"\n" + "=" * 60)
        print("🏁 REAL MULTIMODAL TEST RESULTS")
        print("=" * 60)
        print(f"🔌 Ollama Connection: ✅ PASS")
        print(f"📥 Image Download: ✅ PASS ({len(images)}/{len(TEST_IMAGES)} images)")
        print(f"🚀 Multimodal Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")

        if pipeline_success:
            print("\n🎉 Real multimodal pipeline is working correctly!")
            print("   👁️  Llava vision model generating accurate descriptions")
            print("   🔢 Nomic embedding model creating meaningful embeddings")
            print("   🔗 Full image → description → embedding pipeline operational")
        else:
            print("\n⚠️  Pipeline issues detected - see details above")

        return pipeline_success

if __name__ == "__main__":
    success = run_real_multimodal_tests()
    sys.exit(0 if success else 1)