# Contributing: Native Image Embedding Support

This document outlines how to implement native image embedding support in rust-genai when providers add multimodal embedding capabilities.

## Current Status (2025)

**🔍 Provider Landscape:**
- **❌ Ollama**: No native image embedding API ([Issue #4296](https://github.com/ollama/ollama/issues/4296))
- **❌ OpenAI**: Text-only embedding models (text-embedding-3-*)
- **❌ Anthropic**: No embedding API
- **❌ Groq**: No embedding API
- **✅ Voyage AI**: `voyage-multimodal-3` supports images (not yet integrated)
- **✅ Jina AI**: `jina-clip-v2` supports images (not yet integrated)

**🎯 Goal**: Position rust-genai as the first library ready for multimodal embeddings when the ecosystem matures.

## Architecture Overview

### 1. Future-Ready Types

Located in `src/embed/multimodal_input.rs`:

```rust
pub enum MultimodalEmbedInput {
    Text(String),                           // ✅ Current support
    TextBatch(Vec<String>),                 // ✅ Current support
    Multimodal(Vec<ContentPart>),           // 🔮 Future support
    MultimodalBatch(Vec<Vec<ContentPart>>), // 🔮 Future support
    MixedBatch(Vec<MultimodalEmbedInput>),  // 🔮 Future support
}
```

**Key Features:**
- **Backward Compatible**: Converts to current `EmbedInput` seamlessly
- **Provider Detection**: `is_currently_supported()` checks capability
- **Flexible Content**: Supports text, images, and mixed batches

### 2. Provider Capability System

Track what each provider supports:

```rust
pub struct ProviderCapabilities {
    pub supports_image_embeddings: bool,
    pub supports_multimodal_batch: bool,
    pub max_batch_size: usize,
    pub supported_formats: Vec<String>,
}
```

## Implementation Path

### Phase 1: Foundation (✅ Complete)

1. **Hybrid Architecture**: Working image-to-text-to-embedding pipeline
2. **Future Types**: `MultimodalEmbedInput` with provider detection
3. **Example Implementation**: `e04-future-image-embeddings.rs` shows API design
4. **Documentation**: Clear path for contributors

### Phase 2: Provider Integration (🔄 In Progress)

When providers add image embedding support:

#### 2.1 Add Provider Support

1. **Update Provider Capabilities**:
   ```rust
   // In src/adapter/adapters/ollama/adapter_impl.rs
   impl Adapter for OllamaAdapter {
       fn supports_image_embeddings() -> bool {
           true // ← Update when Ollama adds support
       }
   }
   ```

2. **Extend Embed Request Data**:
   ```rust
   fn to_embed_request_data(
       service_target: ServiceTarget,
       embed_req: EmbedRequest,
       options_set: EmbedOptionsSet<'_, '_>,
   ) -> Result<WebRequestData> {
       // Add image handling logic
       if embed_req.has_images() {
           // Build multimodal request
       } else {
           // Existing text-only logic
       }
   }
   ```

#### 2.2 Update EmbedRequest

1. **Add Image Support**:
   ```rust
   // In src/embed/embed_request.rs
   impl EmbedRequest {
       pub fn new_multimodal(content: Vec<ContentPart>) -> Self {
           Self {
               input: EmbedInput::Multimodal(content),
           }
       }
   }
   ```

2. **Provider-Specific Serialization**:
   ```rust
   // Different providers use different formats:
   // - Ollama: {"model": "...", "input": [...], "images": [...]}
   // - Voyage: {"model": "...", "input": [{"type": "text|image", "content": "..."}]}
   // - Jina: {"model": "...", "inputs": [{"text": "...", "image": "..."}]}
   ```

#### 2.3 Update Client API

```rust
impl Client {
    /// Embed images directly (when provider supports it)
    pub async fn embed_images(
        &self,
        model: &str,
        images: Vec<ContentPart>,
        options: Option<&EmbedOptions>,
    ) -> Result<EmbedResponse> {
        let multimodal_input = MultimodalEmbedInput::Multimodal(images);

        if !multimodal_input.is_currently_supported() {
            return Err(Error::ImageEmbeddingsNotSupported {
                model: model.to_string()
            });
        }

        // Use native image embedding API
        self.exec_embed_multimodal(model, multimodal_input, options).await
    }
}
```

### Phase 3: Provider-Specific Implementation

#### Ollama Implementation (when available)

Track: [ollama/ollama#4296](https://github.com/ollama/ollama/issues/4296)

```rust
// Expected API format (based on GitHub discussions):
{
    "model": "clip-vit-base-patch32",
    "input": ["text description"],
    "images": ["base64_encoded_image_data"]
}
```

**Implementation Steps:**
1. Monitor Ollama releases for image embedding support
2. Add capability detection in `OllamaAdapter`
3. Implement request/response serialization
4. Add integration tests
5. Update documentation

#### Voyage AI Integration

```rust
// Voyage API format:
{
    "model": "voyage-multimodal-3",
    "input": [
        {"type": "text", "text": "description"},
        {"type": "image", "image": "base64_data"}
    ]
}
```

#### Jina AI Integration

```rust
// Jina API format:
{
    "model": "jina-clip-v2",
    "inputs": [
        {"text": "description", "image": "image_url"}
    ]
}
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_multimodal_input_detection() {
        let text_input = MultimodalEmbedInput::from_text("hello");
        assert!(text_input.is_currently_supported());

        let image_input = MultimodalEmbedInput::from_image_url("http://example.com/img.jpg", None);
        assert!(!image_input.is_currently_supported()); // Until providers add support
    }
}
```

### Integration Tests

```rust
// In tests/tests_p_ollama_images.rs (when support is added)
#[tokio::test]
async fn test_ollama_image_embedding() {
    let client = Client::default();
    let image_content = vec![
        ContentPart::from_text("A beautiful landscape"),
        ContentPart::from_binary_url("image/jpeg", TEST_IMAGE_URL, None),
    ];

    let response = client.embed_images("clip-model", image_content, None).await?;
    assert!(response.first_embedding().is_some());
}
```

## Migration Guide

### For Users

**Current (Hybrid) Approach:**
```rust
// Works today with any vision model + text embeddings
let description = describe_image_with_vision(&client, image_url).await?;
let embedding = client.embed("text-embedding-3-small", &description, None).await?;
```

**Future (Native) Approach:**
```rust
// Will work when providers add support
let embedding = client.embed_images("voyage-multimodal-3", vec![
    ContentPart::from_text("Context about the image"),
    ContentPart::from_binary_url("image/jpeg", image_url, None),
], None).await?;
```

### For Contributors

1. **Monitor Provider Releases**: Watch GitHub issues and release notes
2. **Capability Detection**: Use provider capability system
3. **Graceful Fallback**: Always support hybrid approach as backup
4. **Consistent API**: Maintain same patterns as existing adapters

## Performance Considerations

### Native vs Hybrid Approach

**Native Image Embeddings** (when available):
- ✅ Single API call
- ✅ Optimized multimodal representations
- ✅ Lower latency
- ❌ Provider-dependent quality

**Hybrid Approach** (current):
- ✅ Works with any provider
- ✅ High-quality vision models
- ✅ Flexible model combinations
- ❌ Two API calls required

### Optimization Strategies

1. **Caching**: Cache image descriptions to avoid re-processing
2. **Batch Processing**: Group requests to minimize API calls
3. **Model Selection**: Choose optimal vision/embedding model pairs
4. **Format Optimization**: Use efficient image formats (WebP, AVIF)

## Community Involvement

### How to Help

1. **Monitor Provider Progress**:
   - 👀 Watch Ollama issue #4296
   - 📧 Subscribe to OpenAI developer updates
   - 🐦 Follow provider announcements

2. **Contribute Provider Support**:
   - 🔌 Add Voyage AI adapter
   - 🔌 Add Jina AI adapter
   - 🧪 Write integration tests
   - 📚 Update documentation

3. **Improve Hybrid Approach**:
   - 🎯 Optimize vision model prompts
   - ⚡ Add caching mechanisms
   - 🔄 Improve batch processing
   - 📊 Add performance metrics

### Pull Request Guidelines

1. **Provider Detection**: Always check capabilities before using native APIs
2. **Backward Compatibility**: Maintain existing API surface
3. **Error Handling**: Graceful fallback to hybrid approach
4. **Documentation**: Update examples and README files
5. **Testing**: Add both unit and integration tests

## Timeline Expectations

**Q2 2025**: Ollama may add image embedding support
**Q3 2025**: OpenAI might announce multimodal embeddings
**Q4 2025**: Widespread provider adoption expected

**rust-genai will be ready on day one!** 🚀

## Resources

- [Ollama Image Embedding Issue](https://github.com/ollama/ollama/issues/4296)
- [Voyage AI Multimodal Docs](https://docs.voyageai.com/docs/multimodal-embeddings)
- [Jina AI CLIP Models](https://jina.ai/embeddings/)
- [OpenAI Embedding Updates](https://openai.com/index/new-embedding-models-and-api-updates/)

---

**Ready to contribute?** Start by monitoring provider releases and testing our future-ready architecture with the hybrid approach!