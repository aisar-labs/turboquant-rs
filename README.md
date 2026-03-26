# TurboQuant-RS

> **This is a research implementation.** It is intended for learning, experimentation, and reproducing the paper's results. It is **not intended for production use.** There are no stability guarantees, no optimized kernels, and no battle-tested error recovery. If you need production quantization, look at established libraries or wait for official implementations.

Rust implementation of [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://arxiv.org/abs/2504.19874) (ICLR 2026, Google Research).

TurboQuant is a data-oblivious vector quantization algorithm that achieves near-optimal compression without training data. This implementation prioritizes correctness and clarity over performance — f64 throughout, no SIMD, no unsafe code.

## Why TurboQuant Matters

### The Problem

Modern AI has a memory bottleneck. Two critical pain points:

**KV Cache in LLMs:** When an LLM processes a long conversation, it stores key-value vectors for every token seen. A 128K context window on a 70B model can consume 80-160 GB of GPU memory just for the KV cache — leaving almost nothing for model weights or batch processing.

**Vector Search (RAG, embeddings):** 1 billion embedding vectors at 768 dimensions (float32) is 3 TB of storage. Compression is essential for search at scale.

### Why Not Existing Methods?

| Method | Limitation |
|--------|-----------|
| Product Quantization (PQ) | Requires training on your data. Slow to set up. Data-dependent. |
| Naive uniform quantization | Wastes bits storing min/max per block. Adds 1-2 extra bits of overhead. |
| GPTQ, AWQ, etc. | Require calibration data and fine-tuning. Can't compress on-the-fly. |

TurboQuant is the first method that is simultaneously **near-optimal AND data-oblivious** — no training, no calibration, works immediately on any vector.

### Impact for GPU Cloud Providers

The KV cache is the #1 bottleneck for serving LLMs at scale. TurboQuant at 3.5 bits gives 6x compression with zero quality loss:

- **6x more concurrent users per GPU** — direct 6x reduction in cost per query
- **6x longer context windows** in the same memory budget
- **No calibration step** — compress on-the-fly as tokens stream in
- **8x speedup on attention** at 4-bit on H100 GPUs (less data to load from HBM)

At H100 prices (~$2-3/hr), serving 6x more users per GPU translates to millions in savings at scale.

### Impact for Local/Edge AI

Running a 7B model with 8K context on a phone requires ~4 GB for KV cache (float16). Most phones have 6-8 GB total RAM. With TurboQuant at 3 bits, KV cache drops to ~700 MB:

- **Llama-3-8B fits on a phone** with room for long conversations
- **On-device RAG** — store and search thousands of embeddings in 100s of MB
- **Fully offline AI** — private, no internet needed, real context windows
- **IoT/Raspberry Pi** — edge devices with 2-4 GB RAM can run meaningful models
- **Zero setup** — no calibration dataset needed (you can't fine-tune on a phone)

### Impact for Vector Databases

For a RAG system with 100M documents:

| | Float32 | TurboQuant 4-bit |
|---|---------|-----------------|
| Storage | 300 GB | ~37 GB |
| RAM needed | 300 GB | ~37 GB |
| Index build time | Hours (PQ training) | **Instant** (data-oblivious) |
| Search quality | Baseline | Within 2.4x of information-theoretic optimal |

### The Key Insight

When you randomly rotate a vector, each coordinate becomes approximately Gaussian with a known variance. Since you **know the distribution in advance**, you can precompute the optimal quantization grid — no data needed. The rotation "homogenizes" the vector so no coordinate is special, and every coordinate uses the same codebook.

This turns a hard problem (data-dependent codebook learning) into a simple one (look up a precomputed table).

## What is TurboQuant?

Traditional vector quantization (e.g., product quantization) requires training on data to learn codebooks. TurboQuant skips this entirely:

1. **Random rotation** makes vector coordinates statistically well-behaved (approximately Gaussian)
2. **Optimal scalar quantization** per coordinate using precomputed Lloyd-Max codebooks for the known distribution
3. **QJL residual correction** (optional) adds 1-bit sign encoding of the quantization residual for unbiased inner product estimation

### Two Variants

- **TurboQuant_mse** — Minimizes reconstruction MSE. Uses b bits per coordinate for scalar quantization.
- **TurboQuant_prod** — Optimizes inner product estimation. Allocates (b-1) bits for MSE quantization + 1 bit for QJL residual correction. The inner product estimate is unbiased.

## Distortion Results

Reproduces the paper's distortion numbers at d=512 with 10,000 random unit vectors:

| Bits | Measured MSE | Shannon Lower Bound | Ratio |
|------|-------------|-------------------|-------|
| 1    | 0.3625      | 0.2500            | 1.45  |
| 2    | 0.1170      | 0.0625            | 1.87  |
| 3    | 0.0344      | 0.0156            | 2.20  |
| 4    | 0.0095      | 0.0039            | 2.42  |

TurboQuant beats naive uniform quantization at every bit-width, with distortion decreasing ~4x per additional bit.

## Project Structure

```
src/
  rotation.rs       Random orthogonal matrix (QR of Gaussian, Haar measure)
  lloyd_max.rs      Lloyd-Max optimal scalar quantizer (Beta PDF + Simpson's rule)
  codebook.rs       Precomputed codebooks for b=1..4, dimension-aware selection
  turbo_mse.rs      TurboQuant_mse: normalize, rotate, quantize, dequantize
  qjl.rs            QJL: sign-bit projection with unbiased dequantizer
  turbo_prod.rs     TurboQuant_prod: MSE + QJL residual correction
  baseline.rs       Naive uniform quantization (control group)
  distortion.rs     MSE/inner-product metrics and theoretical bounds

tests/              45 tests (unit + integration)
benches/            Criterion benchmarks reproducing the paper's table
```

## Usage

```rust
use turboquant::turbo_mse::TurboMse;
use turboquant::turbo_prod::TurboProd;

// MSE quantization at 2 bits per coordinate
let d = 128;
let tq = TurboMse::new(d, 2, Some(42)); // dimension, bits, seed
let x = vec![1.0; d];

let quantized = tq.quantize(&x);
let reconstructed = tq.dequantize(&quantized);

// Inner product estimation with TurboProd (3 bits total: 2 MSE + 1 QJL)
let tp = TurboProd::new(d, 3, Some(42));
let y = vec![0.5; d];

let q = tp.quantize(&x);
let estimate = tp.estimate_inner_product(&y, &q);
// estimate ≈ true <x, y> (unbiased)
```

## Building and Testing

```bash
cargo build             # Build
cargo test              # Run all 45 tests (debug, ~3 min)
cargo test --release    # Run all tests (release, ~1 min)
cargo bench             # Run benchmarks + print distortion table
cargo clippy            # Lint
```

## Algorithm Details

### Random Rotation

A d x d orthogonal matrix is sampled uniformly (Haar measure) via QR decomposition of a Gaussian random matrix, with sign correction on Q's columns.

### Lloyd-Max Codebook

Coordinates of a randomly rotated unit vector follow a Beta distribution:

```
f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
```

The Lloyd-Max algorithm finds optimal scalar quantization centroids for this distribution by iterating: update boundaries (midpoints) then update centroids (conditional means via numerical integration).

### QJL (Quantized Johnson-Lindenstrauss)

Projects a vector through a random Gaussian matrix and takes sign bits. The dequantizer `(sqrt(pi/2) / d) * S^T * z` provides an unbiased estimate of the input's direction.

### TurboQuant_prod Estimator

```
<y, x> ≈ <y, x_mse> + ||r|| * <y, Q_qjl^{-1}(sign(S * r_hat))>
```

where `r = x - x_mse` is the quantization residual and `r_hat = r / ||r||`.

## Roadmap

- [ ] **KV Cache Module** — Compress key-value pairs on-the-fly, compute attention scores directly on compressed data, full softmax + weighted value decoding. TurboProd for keys (unbiased inner products), TurboMse for values (good reconstruction).
- [ ] **Error Handling** — `Result<T>` returns instead of panics for new modules
- [ ] **Serde Support** — Serialize/deserialize quantized vectors for persistence and network transport
- [ ] **Rotation Trait** — Pluggable rotation interface for future SRHT (Structured Random Hadamard Transform, O(d log d) vs current O(d^2))
- [ ] **SIMD/GPU Kernels** — Optimized quantize/dequantize for production throughput

## References

- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://arxiv.org/abs/2504.19874) — Guo, Kang, Li, Xiao, Yu (Google Research, ICLR 2026)
- Lloyd-Max quantization: Lloyd (1982), Max (1960)
- Johnson-Lindenstrauss lemma: Johnson & Lindenstrauss (1984)

## License

MIT
