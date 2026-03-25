# TurboQuant-RS Design Spec

**Date:** 2026-03-26
**Paper:** [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://arxiv.org/abs/2504.19874) (ICLR 2026)
**Goal:** Research/learning Rust implementation of TurboQuant — faithful to the paper, prioritizing correctness and clarity over production polish.

## Overview

TurboQuant is a data-oblivious vector quantization algorithm with two variants:

- **TurboQuant_mse** — Random rotation + optimal scalar quantization per coordinate. Minimizes MSE distortion.
- **TurboQuant_prod** — Adds QJL (Quantized Johnson-Lindenstrauss) 1-bit residual correction for unbiased inner product estimation.

Key properties: no training data needed, near-optimal distortion bounds, useful for KV cache compression and vector search.

## Scope

- Bit-widths: b=1..4 with precomputed codebooks + Lloyd-Max solver for verification/extension
- Dimensions: small (d=8,16,32) for unit tests, realistic (d=128,256,512) for benchmarks
- Randomness: seeded deterministic via optional seed parameter
- Precision: f64 throughout for clarity
- Includes naive uniform quantization baseline for comparison
- Benchmarks reproduce the paper's distortion table

## Project Structure

```
turboquant-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── rotation.rs       # Π via QR decomposition of Gaussian matrix
│   ├── lloyd_max.rs      # Lloyd-Max solver for optimal scalar codebooks
│   ├── codebook.rs       # Precomputed codebooks for b=1..4 + Beta PDF
│   ├── turbo_mse.rs      # TurboQuant_mse: quantize/dequantize
│   ├── qjl.rs            # QJL: sign(S·x) projection + dequantize
│   ├── turbo_prod.rs     # TurboQuant_prod: mse + QJL residual correction
│   ├── baseline.rs       # Naive uniform quantization for comparison
│   └── distortion.rs     # MSE and inner-product distortion metrics
├── tests/
│   ├── test_rotation.rs
│   ├── test_codebook.rs
│   ├── test_turbo_mse.rs
│   ├── test_turbo_prod.rs
│   ├── test_distortion.rs
│   └── test_baseline.rs
└── benches/
    └── distortion_bench.rs  # Reproduce paper's distortion numbers
```

## Dependencies

- `nalgebra` — Matrix operations (rotation, QR decomposition, matrix-vector multiply)
- `rand` / `rand_distr` — Gaussian sampling for rotation matrix and QJL matrix
- `statrs` — Gamma function for Beta PDF evaluation in Lloyd-Max solver
- `criterion` — Benchmarking (dev-dependency)

## Core Data Types

**Note:** Indices are stored one-per-byte (`Vec<u8>` / `Vec<i8>`) for simplicity. Bit-packing is out of scope for this research implementation.

```rust
/// Precomputed codebook for a given bit-width b.
/// Contains 2^b centroids optimized for the Beta distribution.
pub struct Codebook {
    pub bit_width: u8,           // b = 1..4
    pub centroids: Vec<f64>,     // 2^b sorted centroids in [-1, 1]
    pub boundaries: Vec<f64>,    // 2^b - 1 decision boundaries
}

/// Result of TurboQuant_mse quantization.
/// Tied to the TurboMse instance that produced it — dequantizing with
/// a different instance (different rotation matrix) produces wrong results.
pub struct MseQuantized {
    pub indices: Vec<u8>,        // one index per coordinate (unpacked, one per byte)
    pub bit_width: u8,
    pub norm: f64,               // ||x||_2, stored separately
}

/// Result of TurboQuant_prod quantization.
/// Tied to the TurboProd instance that produced it.
pub struct ProdQuantized {
    pub mse_part: MseQuantized,  // (b-1)-bit MSE quantization
    pub qjl_signs: Vec<i8>,     // sign(S·r̂), +1 or -1, one per dimension (unpacked)
    pub residual_norm: f64,      // ||r||_2
}
```

## Module Designs

### rotation.rs — Random Orthogonal Matrix

Generates Π ∈ ℝ^(d×d) via QR decomposition of a d×d matrix with i.i.d. N(0,1) entries. Sign correction applied to ensure uniform Haar measure (multiply each column of Q by sign of corresponding R diagonal entry).

```rust
pub fn random_orthogonal(d: usize, rng: &mut impl Rng) -> DMatrix<f64>
pub fn rotate(pi: &DMatrix<f64>, x: &[f64]) -> Vec<f64>
pub fn inverse_rotate(pi: &DMatrix<f64>, y: &[f64]) -> Vec<f64>
```

### lloyd_max.rs — Lloyd-Max Solver

Solves the continuous 1D k-means problem for the Beta distribution of coordinates on the unit hypersphere.

**Beta PDF:** `f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)` for x ∈ [-1, 1]

**Precondition:** `d >= 3`. At d=2 the Beta PDF has a singularity at ±1 (arcsine distribution) causing numerical instability in Simpson's rule. At d=1 the distribution degenerates to point masses. Both `beta_pdf` and `solve` assert `d >= 3`.

**Algorithm:**
1. Initialize 2^b centroids uniformly in [-1, 1]
2. Compute decision boundaries as midpoints between adjacent centroids
3. Update centroids to conditional means: `cᵢ = ∫ x·f(x)dx / ∫ f(x)dx` over region
4. Convergence criterion: Δ = max_k |c_k^(new) - c_k^(old)| < 1e-12
5. Integrals via Simpson's rule

```rust
/// Requires d >= 3. Panics otherwise.
pub fn solve(d: usize, bit_width: u8, max_iter: usize) -> Codebook

/// Requires d >= 3. Panics otherwise.
pub fn beta_pdf(x: f64, d: usize) -> f64
```

### codebook.rs — Precomputed Codebooks

Hardcoded tables for b=1..4 at high-d (Gaussian approximation):

Empirical MSE from Lloyd-Max at d→∞ (Gaussian approximation), verified against paper:

| b | Lloyd-Max MSE | Shannon Lower Bound |
|---|--------------|-------------------|
| 1 | 0.3634 | 0.25 |
| 2 | 0.1175 | 0.0625 |
| 3 | ~0.03 | 0.0156 |
| 4 | ~0.009 | 0.0039 |

```rust
pub fn precomputed(bit_width: u8) -> Codebook
pub fn for_dimension(d: usize, bit_width: u8) -> Codebook
```

### turbo_mse.rs — TurboQuant_mse

**Quantization:**
1. Store `norm = ||x||₂`. If norm is zero, return zero-filled `MseQuantized` (all indices 0, norm 0.0).
2. Normalize: `x̂ = x / norm`
3. Rotate: `y = Π · x̂`
4. Per coordinate: find nearest centroid via linear scan over centroids (`argmin_k |y_j - c_k|`). On ties, first (lowest index) wins.

**Dequantization:**
1. Reconstruct: `ỹ_j = centroids[idx_j]`
2. Inverse rotate: `x̃ = Πᵀ · ỹ`
3. Rescale: `x̃ = norm · x̃`

**MSE distortion bound:** D_mse ≤ (√3·π/2) · (1/4^b)

```rust
pub struct TurboMse { rotation: DMatrix<f64>, codebook: Codebook }
impl TurboMse {
    pub fn new(d: usize, bit_width: u8, seed: Option<u64>) -> Self
    pub fn quantize(&self, x: &[f64]) -> MseQuantized
    pub fn dequantize(&self, q: &MseQuantized) -> Vec<f64>
}
```

### qjl.rs — Quantized Johnson-Lindenstrauss

**Quantization:** `signs = sign(S · r)` where S ∈ ℝ^(d×d), entries i.i.d. N(0,1)

**Dequantization:** `Q_qjl⁻¹(z) = (√(π/2) / d) · Sᵀ · z`

Unbiased estimator with variance bound: Var ≤ (π/(2d)) · ||y||₂²

```rust
pub struct Qjl { projection: DMatrix<f64>, d: usize }
impl Qjl {
    pub fn new(d: usize, seed: Option<u64>) -> Self
    pub fn quantize(&self, r: &[f64]) -> Vec<i8>
    pub fn dequantize(&self, signs: &[i8]) -> Vec<f64>
}
```

### turbo_prod.rs — TurboQuant_prod

**Important:** `TurboProd::new(d, bit_width, seed)` constructs its internal `TurboMse` with `bit_width - 1`, not `bit_width`. The assert `bit_width >= 2` is enforced inside `new`.

**Quantization:**
1. `assert!(bit_width >= 2)` — need at least 1 bit for MSE + 1 bit for QJL
2. Apply `self.turbo_mse.quantize(x)` — this uses (b-1) bits
3. Dequantize MSE part: `x̃_mse = self.turbo_mse.dequantize(&mse_part)`
4. Compute residual: `r = x - x̃_mse`
5. Store `residual_norm = ||r||₂`. If residual_norm is zero, store zero signs.
6. Normalize residual: `r̂ = r / residual_norm`
7. Apply QJL to **normalized** residual: `signs = self.qjl.quantize(&r_hat)` — caller must pass `r̂`, not `r`

**Inner product estimation:**
```
⟨y, x⟩ ≈ ⟨y, x̃_mse⟩ + residual_norm · ⟨y, Q_qjl⁻¹(signs)⟩
```

Unbiased: E[estimate] = ⟨y, x⟩

**Distortion bound:** D_prod ≤ (√3·π²·||y||₂²/d) · (1/4^b)

Requires bit_width ≥ 2 (1 bit for MSE + 1 bit for QJL). Panics on b < 2.

```rust
pub struct TurboProd { turbo_mse: TurboMse, qjl: Qjl, bit_width: u8 }
impl TurboProd {
    pub fn new(d: usize, bit_width: u8, seed: Option<u64>) -> Self
    pub fn quantize(&self, x: &[f64]) -> ProdQuantized
    pub fn estimate_inner_product(&self, y: &[f64], q: &ProdQuantized) -> f64
}
```

### baseline.rs — Naive Uniform Quantization

Control group: divide [min, max] into 2^b uniform bins. Requires storing min/max per block — the overhead TurboQuant eliminates.

```rust
pub struct UniformQuantized {
    pub indices: Vec<u8>,  // one index per coordinate (unpacked)
    pub min: f64,
    pub max: f64,
    pub bit_width: u8,
}
pub fn quantize(x: &[f64], bit_width: u8) -> UniformQuantized
pub fn dequantize(q: &UniformQuantized) -> Vec<f64>
```

### distortion.rs — Metrics

**Precondition:** All functions assert matching slice lengths.

```rust
pub fn mse_distortion(original: &[Vec<f64>], reconstructed: &[Vec<f64>]) -> f64
pub fn inner_product_distortion(xs: &[Vec<f64>], ys: &[Vec<f64>], estimates: &[f64]) -> f64
pub fn shannon_lower_bound(bit_width: u8) -> f64
pub fn turboquant_mse_upper_bound(bit_width: u8) -> f64
/// D_prod ≤ (√3·π²·||y||₂²/d) · (1/4^b)
pub fn turboquant_prod_upper_bound(bit_width: u8, d: usize, y_norm_sq: f64) -> f64
```

## Test Strategy

### Unit Tests (d=8, 16, 32)

| Test file | Verifications |
|-----------|--------------|
| `test_rotation.rs` | Πᵀ·Π = I, norm preservation, rotate/inverse_rotate roundtrip |
| `test_codebook.rs` | Lloyd-Max convergence, centroid symmetry, Beta PDF integrates to 1, solver matches precomputed at high d |
| `test_turbo_mse.rs` | Norm preservation, deterministic with seed, zero vector handling |
| `test_turbo_prod.rs` | Unbiased inner product estimate, panics on b<2 |
| `test_distortion.rs` | Perfect reconstruction = 0 distortion, metrics non-negative |
| `test_baseline.rs` | Uniform quantization correctness, min/max handling |

### Integration Tests (d=128, 256, 512)

- TurboQuant MSE distortion within paper's upper bound
- TurboQuant outperforms naive uniform baseline at every bit-width
- Distortion decreases ~4× per additional bit
- Inner product estimator is unbiased: mean absolute error < 0.01 · ||y||₂ over 10k random unit vector pairs

### Benchmarks

Reproduce paper's distortion table (b=1..4, d=512, 10k random unit vectors):

| b | Lloyd-Max MSE | Shannon LB | Ratio |
|---|--------------|-----------|-------|
| 1 | ~0.3634 | 0.25 | ~1.45 |
| 2 | ~0.1175 | 0.0625 | ~1.88 |
| 3 | ~0.03 | 0.0156 | ~1.92 |
| 4 | ~0.009 | 0.0039 | ~2.31 |

All tests use seeded RNG for determinism. Integration tests assert within statistical tolerance (measured MSE within 10% of expected).
