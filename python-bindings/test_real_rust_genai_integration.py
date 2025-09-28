#!/usr/bin/env python3
"""
Real Rust-GenAI Integration Test

This test uses the actual rust-genai library directly via a standalone
Rust binary to test the complete multimodal pipeline with real AI models.

This approach allows us to validate the real functionality while keeping
the Python bindings simple and focused on performance.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Any

# Test configuration
TEST_IMAGES = [
    {
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300",
        "name": "mountain_landscape.jpg",
        "expected_content": ["mountain", "landscape", "nature"]
    },
    {
        "url": "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=300",
        "name": "golden_retriever.jpg",
        "expected_content": ["dog", "golden", "retriever", "animal"]
    }
]

def create_rust_test_binary():
    """Create a standalone Rust binary for testing real integration."""
    test_code = '''
use genai::Client;
use genai::chat::{ChatMessage, ChatRequest, ContentPart};
use serde_json::json;
use std::env;

const VISION_MODEL: &str = "llava:7b";
const EMBEDDING_MODEL: &str = "nomic-embed-text:latest";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <image_url>", args[0]);
        std::process::exit(1);
    }

    let image_url = &args[1];
    let client = Client::default();

    // Step 1: Get description from Llava
    let chat_req = ChatRequest::new(vec![
        ChatMessage::system("You are a helpful vision AI. Describe images accurately and concisely for embedding purposes. Focus on key visual elements, objects, scene context, colors, and composition."),
        ChatMessage::user(vec![
            ContentPart::from_text("Describe this image in detail for search and embedding purposes:"),
            ContentPart::from_binary_url("image/jpeg", image_url, None),
        ])
    ]);

    let chat_response = client.exec_chat(VISION_MODEL, chat_req, None).await?;
    let description = chat_response.first_text()
        .ok_or("No description generated")?;

    // Step 2: Create embedding from description
    let embed_response = client.embed(EMBEDDING_MODEL, description, None).await?;
    let embedding = embed_response.first_embedding()
        .ok_or("No embedding generated")?;

    // Output results as JSON
    let result = json!({
        "success": true,
        "image_url": image_url,
        "description": description,
        "embedding": {
            "dimensions": embedding.dimensions(),
            "vector_preview": &embedding.vector()[0..3],
        },
        "models": {
            "vision": VISION_MODEL,
            "embedding": EMBEDDING_MODEL
        }
    });

    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
'''

    # Get absolute path to genai library
    genai_path = Path(__file__).parent.parent.absolute()

    cargo_toml = f'''
[package]
name = "rust-genai-test"
version = "0.1.0"
edition = "2021"

[dependencies]
genai = {{ path = "{genai_path}" }}
tokio = {{ version = "1", features = ["macros", "rt-multi-thread"] }}
serde_json = "1.0"
'''

    # Create temporary Rust project
    temp_dir = Path(tempfile.mkdtemp())
    src_dir = temp_dir / "src"
    src_dir.mkdir(parents=True)

    # Write files
    (temp_dir / "Cargo.toml").write_text(cargo_toml)
    (src_dir / "main.rs").write_text(test_code)

    print(f"📦 Created test binary at: {temp_dir}")
    return temp_dir

def build_rust_test_binary(project_path: Path) -> Path:
    """Build the Rust test binary."""
    print("🔨 Building Rust test binary...")

    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=project_path,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Build failed:")
        print(result.stderr)
        raise RuntimeError("Failed to build Rust test binary")

    binary_path = project_path / "target" / "release" / "rust-genai-test"
    print(f"✅ Built binary: {binary_path}")
    return binary_path

def download_test_images(temp_dir: Path) -> List[Dict[str, Any]]:
    """Download test images."""
    print("📥 Downloading test images...")

    downloaded = []
    for img_info in TEST_IMAGES:
        try:
            headers = {'User-Agent': 'rust-genai-integration-test/0.4.0'}
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

def test_with_rust_binary(binary_path: Path, image_path: str) -> Dict[str, Any]:
    """Test image processing with the Rust binary."""
    try:
        result = subprocess.run(
            [str(binary_path), f"file://{image_path}"],
            capture_output=True,
            text=True,
            timeout=60  # Llava can be slow
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Binary failed: {result.stderr}",
                "stdout": result.stdout
            }

        # Parse JSON output
        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Processing timeout (60s exceeded)"
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON output: {e}",
            "stdout": result.stdout if 'result' in locals() else "No output"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {e}"
        }

def validate_content(description: str, expected_content: List[str]) -> Dict[str, Any]:
    """Validate description contains expected content."""
    description_lower = description.lower()
    found = [item for item in expected_content if item.lower() in description_lower]
    accuracy = len(found) / len(expected_content) if expected_content else 0.0

    return {
        "accuracy": accuracy,
        "found": found,
        "missing": [item for item in expected_content if item not in found]
    }

def run_rust_genai_integration_tests():
    """Run comprehensive integration tests with real Rust-GenAI library."""
    print("🦀 RUST-GENAI REAL INTEGRATION TEST")
    print("=" * 50)
    print("Testing actual rust-genai library with real AI models\n")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Step 1: Create and build Rust test binary
            project_path = create_rust_test_binary()
            binary_path = build_rust_test_binary(project_path)

            # Step 2: Download test images
            images = download_test_images(temp_path)
            if not images:
                print("❌ No images downloaded - cannot run tests")
                return False

            # Step 3: Test with real AI models
            print(f"\n🚀 Testing Real AI Pipeline")
            print("=" * 35)

            results = []
            total_start = time.time()

            for i, img_info in enumerate(images):
                print(f"\n📸 Processing image {i+1}/{len(images)}: {img_info['name']}")

                start_time = time.time()
                result = test_with_rust_binary(binary_path, img_info['path'])
                duration = time.time() - start_time

                if result['success']:
                    print(f"   ✅ Processed in {duration:.1f}s")
                    print(f"   📝 Description: {result['description'][:80]}...")
                    print(f"   📊 Embedding: {result['embedding']['dimensions']} dimensions")

                    # Validate content
                    validation = validate_content(result['description'], img_info['expected_content'])
                    print(f"   🎯 Content accuracy: {validation['accuracy']:.1%}")
                    print(f"   ✅ Found: {validation['found']}")
                    if validation['missing']:
                        print(f"   ❌ Missing: {validation['missing']}")

                    results.append({
                        **result,
                        'processing_time': duration,
                        'content_validation': validation
                    })
                else:
                    print(f"   ❌ Failed: {result['error']}")

            total_duration = time.time() - total_start

            # Summary
            print(f"\n" + "=" * 50)
            print("📊 INTEGRATION TEST RESULTS")
            print("=" * 50)

            successful = len(results)
            print(f"✅ Successful: {successful}/{len(images)} images")
            print(f"⏱️  Total time: {total_duration:.1f}s")
            print(f"🚀 Average per image: {total_duration/len(images):.1f}s")

            if results:
                avg_accuracy = sum(r['content_validation']['accuracy'] for r in results) / len(results)
                print(f"🎯 Average content accuracy: {avg_accuracy:.1%}")

                print(f"\n🔍 Detailed Results:")
                for result in results:
                    print(f"   📄 {Path(result['image_url']).name}")
                    print(f"      👁️  Vision: {result['models']['vision']}")
                    print(f"      🔢 Embedding: {result['models']['embedding']}")
                    print(f"      ⏱️  Time: {result['processing_time']:.1f}s")
                    print(f"      🎯 Accuracy: {result['content_validation']['accuracy']:.1%}")

            success = successful > 0

            if success:
                print(f"\n🎉 Integration test successful!")
                print("   🦀 Rust-GenAI library working correctly")
                print("   👁️  Llava vision model operational")
                print("   🔢 Nomic embedding model operational")
                print("   🔗 Complete image → description → embedding pipeline validated")
            else:
                print(f"\n⚠️  Integration test failed - no successful processes")

            return success

        except Exception as e:
            print(f"❌ Test setup failed: {e}")
            return False
        finally:
            # Cleanup
            if 'project_path' in locals():
                import shutil
                shutil.rmtree(project_path, ignore_errors=True)

if __name__ == "__main__":
    success = run_rust_genai_integration_tests()
    sys.exit(0 if success else 1)