# API Embedder Integration Guide

## Overview

ck now supports external embedding API endpoints, including vLLM servers with hosted open source models like Jina embeddings. This allows you to:

- Use centralized embedding infrastructure
- Leverage GPU-accelerated embedding servers
- Share embeddings across multiple tools
- Use the latest embedding models without local storage

## Quick Start with vLLM

### 1. Start your vLLM server with Jina embeddings

```bash
# Example: Start vLLM with Jina embeddings model
vllm serve jinaai/jina-embeddings-v2-base-en --port 8000
```

### 2. Index your codebase using the API

```bash
# Option 1: Using CLI flags
ck --index . \
  --embedding-api "http://localhost:8000/v1/embeddings" \
  --embedding-model "jinaai/jina-embeddings-v2-base-en" \
  --embedding-dim 768

# Option 2: Using environment variables (recommended for persistent config)
export CK_EMBEDDING_API="http://localhost:8000/v1/embeddings"
export CK_EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-en"
export CK_EMBEDDING_DIM=768

ck --index .
```

### 3. Search as normal

```bash
# Once indexed, search works the same way
ck --sem "authentication logic" .
ck --hybrid "database connection" .
```

## Configuration Options

### CLI Flags

- `--embedding-api URL`: API endpoint URL (OpenAI-compatible format)
- `--embedding-model NAME`: Model name to request from the API
- `--embedding-dim N`: Embedding dimensions (default: 768)
- `--embedding-api-key-file PATH`: Read API key from a local file (contents are trimmed). Recommended for OpenAI/HuggingFace tokens because it keeps secrets out of the process list.

### Environment Variables

- `CK_EMBEDDING_API`: API endpoint URL
- `CK_EMBEDDING_MODEL`: Model name
- `CK_EMBEDDING_DIM`: Embedding dimensions
- `CK_EMBEDDING_API_KEY`: API key (if required)

Environment variables take precedence over CLI flags if both are set.
Avoid putting API keys directly on the command line—other local users can read `/proc/<pid>/cmdline`. Use environment variables or `--embedding-api-key-file` instead.

## OpenAI-Compatible API Format

ck expects the standard OpenAI embeddings API format:

**Request:**
```json
{
  "input": ["text1", "text2"],
  "model": "model-name"
}
```

**Response:**
```json
{
  "data": [
    {
      "embedding": [0.1, 0.2, ...],
      "index": 0
    },
    {
      "embedding": [0.3, 0.4, ...],
      "index": 1
    }
  ]
}
```

## Supported API Providers

### vLLM (Recommended for local hosting)
```bash
export CK_EMBEDDING_API="http://localhost:8000/v1/embeddings"
export CK_EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-en"
export CK_EMBEDDING_DIM=768
```

### OpenAI
```bash
export CK_EMBEDDING_API="https://api.openai.com/v1/embeddings"
export CK_EMBEDDING_API_KEY="sk-..."
export CK_EMBEDDING_MODEL="text-embedding-3-small"
export CK_EMBEDDING_DIM=1536
```

### HuggingFace Inference API
```bash
export CK_EMBEDDING_API="https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
export CK_EMBEDDING_API_KEY="hf_..."
export CK_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export CK_EMBEDDING_DIM=384
```

### Custom OpenAI-compatible endpoints
Any service that implements the OpenAI embeddings API format will work.

## Use Cases for Local Coding Agents

### Integration with OpenHands

1. **Start vLLM server** with your preferred embedding model:
   ```bash
   vllm serve jinaai/jina-embeddings-v2-base-code --port 8000
   ```

2. **Configure ck** to use the same server:
   ```bash
   export CK_EMBEDDING_API="http://localhost:8000/v1/embeddings"
   export CK_EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-code"
   export CK_EMBEDDING_DIM=768
   ```

3. **Index your workspace**:
   ```bash
   ck --index /path/to/workspace
   ```

4. **OpenHands can now use ck** for semantic code search:
   ```python
   # In OpenHands agent code
   result = subprocess.run(
       ["ck", "--jsonl", "--sem", "authentication logic", workspace_path],
       capture_output=True
   )
   matches = [json.loads(line) for line in result.stdout.split('\n') if line]
   ```

### Benefits

- **Shared Infrastructure**: Both OpenHands and ck use the same embedding server
- **GPU Acceleration**: vLLM can leverage GPU for faster embeddings
- **Consistency**: Same embeddings across your entire agent workflow
- **Resource Efficiency**: Single embedding model serves multiple tools
- **Flexibility**: Easy to swap models by restarting vLLM

## Performance Considerations

### Local Model vs API

| Aspect | Local (FastEmbed) | External API |
|--------|------------------|--------------|
| First-time setup | Downloads ~300MB model | No download needed |
| Network dependency | None | Required |
| Latency | 0ms network latency | Depends on network |
| GPU usage | CPU-only (ONNX) | Can use GPU via vLLM |
| Shared use | Per-machine | Centralized server |
| Privacy | Fully offline | Data sent to server |

### Recommended Setup

**For single-user, local development:**
- Use local FastEmbed models (default)
- Fast, private, no network dependency

**For team environments or GPU workstations:**
- Deploy vLLM server on GPU machine
- All developers connect to shared server
- Centralized infrastructure, consistent embeddings

## Troubleshooting

### "API request failed with status 404"
Check that your vLLM server is running and the endpoint URL is correct.

### "Response missing 'data' field"
Ensure your API server returns OpenAI-compatible format. vLLM should do this automatically.

### "Embedding value is not a number"
Check that `--embedding-dim` matches your model's actual output dimensions.

### API key not working
For vLLM (local), you typically don't need an API key. For hosted services, ensure `CK_EMBEDDING_API_KEY` is set correctly.

## Example: Complete Workflow

```bash
# Terminal 1: Start vLLM server
vllm serve jinaai/jina-embeddings-v2-base-en \
  --port 8000 \
  --dtype float16

# Terminal 2: Configure and use ck
export CK_EMBEDDING_API="http://localhost:8000/v1/embeddings"
export CK_EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-en"
export CK_EMBEDDING_DIM=768

# Index your codebase
cd /path/to/your/project
ck --index .

# Search semantically
ck --sem "error handling patterns" src/
ck --hybrid "async retry logic" .
ck --jsonl --sem "database connection" src/ | jq
```

## Implementation Details

The API embedder feature is implemented in:
- `ck-embed/src/lib.rs`: ApiEmbedder struct with OpenAI-compatible HTTP client
- `ck-cli/src/main.rs`: CLI flags and environment variable handling
- `ck-embed/Cargo.toml`: reqwest dependency with blocking feature

The implementation uses synchronous reqwest blocking client to match ck's existing sync Embedder trait, avoiding the need to refactor the entire codebase to async.

## Contributing

If you encounter issues or want to add support for other API formats, please open an issue or PR at:
https://github.com/BeaconBay/ck

## See Also

- [Main README](README.md) - Full ck documentation
- [Examples Guide](EXAMPLES.md) - Usage examples
- [Model Selection](README.md#model-selection) - Choosing the right embedding model
