# Embedding Model Evaluation Guide

## Representative Models
- Qwen3-Embedding (0.6B/4B/8B): Leads MTEB leaderboard
- bge-m3 (BAAI): Dense + sparse + multi-vector in one model
- stella_en_1.5B_v5: Matryoshka support, compact
- nomic-embed-text-v1.5/v2: Long context (8192 tokens)
- sentence-transformers/all-MiniLM-L6-v2: Fast baseline

## Framework
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_id, trust_remote_code=True)
embeddings = model.encode(
    ["query: what is MTEB?", "passage: MTEB is a benchmark for embeddings"],
    normalize_embeddings=True,
)
```

For bge-m3 (multi-vector):
```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel(model_id, use_fp16=True)
output = model.encode(sentences, return_dense=True, return_sparse=True)
```

## Smoke Test
- Load model
- Encode two similar sentences and two unrelated ones
- Verify cosine similarity: similar > unrelated

```python
import numpy as np

embs = model.encode([
    "The cat sat on the mat",
    "A feline was resting on the rug",
    "Stock prices rose sharply today",
])
# Cosine similarity: embs[0] @ embs[1] should be >> embs[0] @ embs[2]
sim_01 = np.dot(embs[0], embs[1])
sim_02 = np.dot(embs[0], embs[2])
assert sim_01 > sim_02 + 0.1, f"Similarity sanity check failed: {sim_01} vs {sim_02}"
```

## Quality Check — Retrieval Task

Build a small retrieval test:
```python
corpus = [
    "Python is a programming language",
    "Tokyo is the capital of Japan",
    "Machine learning uses data to make predictions",
    "The Eiffel Tower is in Paris",
    "Neural networks have layers of neurons",
]
query = "What is deep learning?"

corpus_embs = model.encode(corpus, normalize_embeddings=True)
query_emb = model.encode([query], normalize_embeddings=True)

scores = query_emb @ corpus_embs.T
ranked = sorted(zip(scores[0], corpus), reverse=True)
# Top results should be ML-related sentences
```

Test with at least 3 queries covering different domains.

### Instruction Prefix Check
Many models require prefixes for asymmetric retrieval:
- `"query: "` for queries, `"passage: "` for documents
- Omitting prefixes can drop performance 10-20%
- Check model card for required prefixes

## Performance Measurement

```python
import time

sentences = ["Test sentence number {i}" for i in range(100)]

# Warmup
model.encode(sentences[:10])

times = []
for _ in range(3):
    start = time.perf_counter()
    model.encode(sentences, batch_size=32, normalize_embeddings=True)
    elapsed = time.perf_counter() - start
    times.append(len(sentences) / elapsed)

embeddings_per_sec = sorted(times)[len(times) // 2]
```

Key metrics:
- **embeddings/sec** (batch of 100 sentences)
- **embedding dimension** (128, 384, 768, 1024, 4096)
- **max sequence length** (usually 512; some models support 8192)

## Standard Benchmarks (for reference)
- **MTEB**: 7 task types (retrieval, clustering, classification, STS, etc.)
  - Filter by your use case; overall average is misleading
- **MMTEB**: Multilingual extension (250+ languages)

## Common Issues
- **Instruction prefixes required**: Omitting "query:" / "passage:" silently degrades results.
- **Normalization matters**: Always `normalize_embeddings=True` for cosine similarity.
- **Max sequence truncation**: Most models cap at 512 tokens. Long text is silently truncated.
- **MTEB overall score is misleading**: A mediocre all-rounder can outscore a retrieval specialist.
- **Domain gap**: MTEB scores don't predict performance on domain-specific data (bio, legal, code).
- **Matryoshka support**: Some models support truncatable dims (256/512/1024); useful for storage.
