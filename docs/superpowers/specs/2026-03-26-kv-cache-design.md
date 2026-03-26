# KV Cache Module Design Spec

**Date:** 2026-03-26
**Goal:** Add a KV cache compressor module that enables full attention computation on TurboQuant-compressed key-value pairs, plus supporting additions (error handling, serde).

## Overview

The KV cache module bridges TurboQuant from a quantization algorithm to a practical LLM inference tool. It compresses key-value vectors on-the-fly, computes attention scores directly on compressed data (no decompression for keys), and performs full attention (softmax + weighted value decoding).

## Scope

**New files:**
- `src/error.rs` — `TurboQuantError` enum (used only by new code)
- `src/kv.rs` — `KvCacheCompressor` with full attention support

**Modified files:**
- `src/lib.rs` — add `pub mod error; pub mod kv;`
- `src/turbo_mse.rs` — add `Serialize, Deserialize` derives to `MseQuantized`
- `src/turbo_prod.rs` — add `Serialize, Deserialize` derives to `ProdQuantized`
- `src/codebook.rs` — add `Serialize, Deserialize` derives to `Codebook`
- `Cargo.toml` — add `serde`, `thiserror` dependencies; `serde_json` dev-dependency

**Not changed:** Existing function signatures, error handling patterns (panics stay in existing modules).

## Dependencies (new)

```toml
serde = { version = "1", features = ["derive"] }
thiserror = "1"

[dev-dependencies]
serde_json = "1"
```

## Error Types (`error.rs`)

```rust
#[derive(Debug, thiserror::Error)]
pub enum TurboQuantError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid bit width: {0} (must be 2..=4 for TurboProd, 1..=4 for TurboMse)")]
    InvalidBitWidth(u8),

    #[error("Empty cache: no tokens stored")]
    EmptyCache,
}
```

## KV Cache Module (`kv.rs`)

### Configuration

```rust
pub struct KvCacheConfig {
    pub head_dim: usize,    // dimension per attention head (must be >= 3 for Lloyd-Max)
    pub key_bits: u8,       // total bits for key quantization (TurboProd, 2..=4)
    pub value_bits: u8,     // bits for value quantization (TurboMse, 1..=4)
    pub seed: u64,          // deterministic seed
}
```

### Core Types

```rust
pub struct KvCacheCompressor {
    config: KvCacheConfig,
    key_quantizer: TurboProd,     // unbiased inner products for attention scores
    value_quantizer: TurboMse,    // good reconstruction for weighted sum
    tokens: Vec<CompressedToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedToken {
    pub key: ProdQuantized,
    pub value: MseQuantized,
}
```

### Seed Strategy

- Key quantizer (TurboProd): uses `seed` directly
- Value quantizer (TurboMse): uses `seed.wrapping_add(0x1234_5678)`
- Ensures independent rotation matrices and QJL projections

### API

```rust
impl KvCacheCompressor {
    /// Create from config. Returns error on invalid dimensions/bits.
    /// Validates: head_dim >= 3, key_bits in 2..=4, value_bits in 1..=4.
    pub fn new(config: KvCacheConfig) -> Result<Self, TurboQuantError>

    /// Compress and append a key-value pair. Returns token index.
    /// Returns DimensionMismatch if key.len() or value.len() != head_dim.
    pub fn compress_token(&mut self, key: &[f64], value: &[f64]) -> Result<usize, TurboQuantError>

    /// Compute attention scores: inner product of query against all stored keys.
    /// Returns one score per token (unnormalized logits).
    /// Returns EmptyCache if no tokens stored.
    /// Returns DimensionMismatch if query.len() != head_dim.
    pub fn attention_scores(&self, query: &[f64]) -> Result<Vec<f64>, TurboQuantError>

    /// Full attention: softmax(scores) weighted sum of decoded values.
    /// Returns the attended value vector (dimension = head_dim).
    /// Returns EmptyCache if no tokens stored.
    /// Returns DimensionMismatch if query.len() != head_dim.
    pub fn attend(&self, query: &[f64]) -> Result<Vec<f64>, TurboQuantError>

    /// Number of stored tokens.
    pub fn len(&self) -> usize

    /// Whether cache is empty.
    pub fn is_empty(&self) -> bool

    /// Decode a specific token's value. Returns DimensionMismatch-style error if index out of bounds.
    pub fn decode_value(&self, index: usize) -> Result<Vec<f64>, TurboQuantError>
}
```

### `attend` Algorithm

1. Compute `scores = attention_scores(query)?`
2. Numerically stable softmax:
   - `max_score = scores.max()`
   - `exp_scores[i] = exp(scores[i] - max_score)`
   - `sum = exp_scores.sum()`
   - `weights[i] = exp_scores[i] / sum`
3. For each token i:
   - `value_i = value_quantizer.dequantize(&tokens[i].value)`
   - Accumulate: `output += weights[i] * value_i`
4. Return `output`

### Error Conditions

| Method | Error | Condition |
|--------|-------|-----------|
| `new` | `InvalidBitWidth` | key_bits < 2 or > 4, value_bits < 1 or > 4 |
| `new` | `DimensionMismatch` | head_dim < 3 |
| `compress_token` | `DimensionMismatch` | key or value length != head_dim |
| `attention_scores` | `EmptyCache` | no tokens stored |
| `attention_scores` | `DimensionMismatch` | query length != head_dim |
| `attend` | `EmptyCache` | no tokens stored |
| `attend` | `DimensionMismatch` | query length != head_dim |
| `decode_value` | `DimensionMismatch` | index >= len() |

## Serde Additions

Add `#[derive(Serialize, Deserialize)]` to:
- `MseQuantized` (in `turbo_mse.rs`)
- `ProdQuantized` (in `turbo_prod.rs`)
- `Codebook` (in `codebook.rs`)
- `CompressedToken` (in `kv.rs`)

This enables persisting compressed tokens to disk/network via any serde-compatible format.

## Test Strategy

### `tests/test_kv.rs`

| Test | What it verifies |
|------|-----------------|
| `test_compress_and_count` | Token count increments after each compress_token |
| `test_attention_scores_length` | Returns one score per stored token |
| `test_attend_dimension` | Output vector has dimension = head_dim |
| `test_attend_favors_similar_key` | Query attends more to similar keys (higher weight) than dissimilar |
| `test_empty_cache_errors` | attention_scores and attend return EmptyCache |
| `test_dimension_mismatch_errors` | Wrong-sized key/value/query returns DimensionMismatch |
| `test_softmax_weights_sum_to_one` | Verify softmax normalization (indirectly via attend output magnitude) |
| `test_serialization_roundtrip` | CompressedToken survives serde_json serialize/deserialize |
| `test_independent_key_value_seeds` | Key and value quantizers produce different results for same input |

### Serde tests (in `tests/test_serde.rs`)

| Test | What it verifies |
|------|-----------------|
| `test_mse_quantized_roundtrip` | MseQuantized serializes/deserializes correctly |
| `test_prod_quantized_roundtrip` | ProdQuantized serializes/deserializes correctly |
