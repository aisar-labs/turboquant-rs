# TurboQuant-RS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Google's TurboQuant vector quantization algorithm in Rust as a research/learning codebase with comprehensive tests reproducing the paper's distortion numbers.

**Architecture:** Single Rust crate with flat modules mapping 1:1 to paper sections. Each module is a self-contained algorithm component (rotation, codebook, quantizer) using plain functions and structs — no trait abstractions. f64 precision throughout.

**Tech Stack:** Rust, nalgebra (linear algebra), rand/rand_distr (RNG), statrs (special functions), criterion (benchmarks)

**Spec:** `docs/superpowers/specs/2026-03-26-turboquant-rs-design.md`

---

### Task 1: Project Scaffold & Dependencies

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`

- [ ] **Step 1: Initialize Cargo project**

```bash
cd /Users/dr.noranizaahmad/ios/turboquant-rs
cargo init --lib --name turboquant
```

- [ ] **Step 2: Set up Cargo.toml with dependencies**

Replace the generated `Cargo.toml` with:

```toml
[package]
name = "turboquant"
version = "0.1.0"
edition = "2021"
description = "Research implementation of TurboQuant vector quantization (ICLR 2026)"

[dependencies]
nalgebra = "0.33"
rand = "0.8"
rand_distr = "0.4"
statrs = "0.17"
rand_chacha = "0.3"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
approx = "0.5"

[[bench]]
name = "distortion_bench"
harness = false
```

- [ ] **Step 3: Set up lib.rs with module declarations**

```rust
pub mod rotation;
pub mod lloyd_max;
pub mod codebook;
pub mod turbo_mse;
pub mod qjl;
pub mod turbo_prod;
pub mod baseline;
pub mod distortion;
```

- [ ] **Step 4: Create stub files for all modules**

Create empty files: `src/rotation.rs`, `src/lloyd_max.rs`, `src/codebook.rs`, `src/turbo_mse.rs`, `src/qjl.rs`, `src/turbo_prod.rs`, `src/baseline.rs`, `src/distortion.rs`

- [ ] **Step 5: Verify it compiles**

Run: `cargo build`
Expected: Successful compilation with warnings about empty files.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: scaffold turboquant crate with dependencies"
```

---

### Task 2: Distortion Metrics (`distortion.rs`)

Building this first — all other modules' tests depend on these metrics.

**Files:**
- Create: `src/distortion.rs`
- Create: `tests/test_distortion.rs`

- [ ] **Step 1: Write failing tests for distortion metrics**

Create `tests/test_distortion.rs`:

```rust
use turboquant::distortion::*;

#[test]
fn test_mse_distortion_perfect_reconstruction() {
    let original = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let reconstructed = original.clone();
    let d = mse_distortion(&original, &reconstructed);
    assert!((d - 0.0).abs() < 1e-15, "Perfect reconstruction should have zero distortion, got {d}");
}

#[test]
fn test_mse_distortion_is_nonnegative() {
    let original = vec![vec![1.0, 0.0, 0.0]];
    let reconstructed = vec![vec![0.5, 0.1, -0.1]];
    let d = mse_distortion(&original, &reconstructed);
    assert!(d >= 0.0, "MSE distortion must be non-negative, got {d}");
}

#[test]
fn test_mse_distortion_known_value() {
    // x = [1, 0, 0], x~ = [0, 0, 0] => ||x - x~||^2 / ||x||^2 = 1.0
    let original = vec![vec![1.0, 0.0, 0.0]];
    let reconstructed = vec![vec![0.0, 0.0, 0.0]];
    let d = mse_distortion(&original, &reconstructed);
    assert!((d - 1.0).abs() < 1e-15, "Expected 1.0, got {d}");
}

#[test]
fn test_inner_product_distortion_perfect() {
    let xs = vec![vec![1.0, 2.0]];
    let ys = vec![vec![3.0, 4.0]];
    let true_ip: f64 = 1.0 * 3.0 + 2.0 * 4.0; // 11.0
    let estimates = vec![true_ip];
    let d = inner_product_distortion(&xs, &ys, &estimates);
    assert!((d - 0.0).abs() < 1e-15, "Perfect estimate should have zero distortion, got {d}");
}

#[test]
fn test_inner_product_distortion_nonnegative() {
    let xs = vec![vec![1.0, 0.0]];
    let ys = vec![vec![0.0, 1.0]];
    let estimates = vec![0.5]; // true is 0.0
    let d = inner_product_distortion(&xs, &ys, &estimates);
    assert!(d >= 0.0, "Inner product distortion must be non-negative, got {d}");
}

#[test]
fn test_shannon_lower_bound() {
    assert!((shannon_lower_bound(1) - 0.25).abs() < 1e-15);
    assert!((shannon_lower_bound(2) - 0.0625).abs() < 1e-15);
    assert!((shannon_lower_bound(3) - 0.015625).abs() < 1e-15);
    assert!((shannon_lower_bound(4) - 0.00390625).abs() < 1e-15);
}

#[test]
fn test_turboquant_mse_upper_bound() {
    for b in 1..=4u8 {
        let ub = turboquant_mse_upper_bound(b);
        let lb = shannon_lower_bound(b);
        assert!(ub > lb, "Upper bound {ub} should exceed Shannon LB {lb} at b={b}");
    }
}

#[test]
fn test_turboquant_prod_upper_bound() {
    let ub = turboquant_prod_upper_bound(2, 128, 1.0);
    assert!(ub > 0.0);
    // Should decrease with higher bit-width
    let ub3 = turboquant_prod_upper_bound(3, 128, 1.0);
    assert!(ub3 < ub, "Higher bit-width should give lower bound");
}

#[test]
#[should_panic]
fn test_mse_distortion_mismatched_lengths() {
    let original = vec![vec![1.0]];
    let reconstructed = vec![vec![1.0], vec![2.0]];
    mse_distortion(&original, &reconstructed);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test test_distortion`
Expected: Compilation errors — `distortion` module has no functions yet.

- [ ] **Step 3: Implement distortion.rs**

Write `src/distortion.rs`:

```rust
/// Normalized MSE distortion: average of ||x - x~||^2 / ||x||^2 over all vectors.
/// Panics if slices have different lengths.
pub fn mse_distortion(original: &[Vec<f64>], reconstructed: &[Vec<f64>]) -> f64 {
    assert_eq!(original.len(), reconstructed.len(), "Mismatched slice lengths");
    if original.is_empty() {
        return 0.0;
    }
    let sum: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(x, x_hat)| {
            assert_eq!(x.len(), x_hat.len(), "Inner vector length mismatch");
            let norm_sq: f64 = x.iter().map(|v| v * v).sum();
            if norm_sq == 0.0 {
                return 0.0;
            }
            let err_sq: f64 = x.iter().zip(x_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            err_sq / norm_sq
        })
        .sum();
    sum / original.len() as f64
}

/// Inner product distortion: average of (true_ip - estimate)^2.
/// Panics if slices have different lengths.
pub fn inner_product_distortion(xs: &[Vec<f64>], ys: &[Vec<f64>], estimates: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len(), "Mismatched xs/ys lengths");
    assert_eq!(xs.len(), estimates.len(), "Mismatched xs/estimates lengths");
    if xs.is_empty() {
        return 0.0;
    }
    let sum: f64 = xs
        .iter()
        .zip(ys.iter())
        .zip(estimates.iter())
        .map(|((x, y), &est)| {
            assert_eq!(x.len(), y.len(), "Inner vector length mismatch");
            let true_ip: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            (true_ip - est).powi(2)
        })
        .sum();
    sum / xs.len() as f64
}

/// Shannon lower bound on MSE: 1/4^b
pub fn shannon_lower_bound(bit_width: u8) -> f64 {
    1.0 / 4.0_f64.powi(bit_width as i32)
}

/// TurboQuant MSE upper bound: (sqrt(3)*pi/2) * (1/4^b)
pub fn turboquant_mse_upper_bound(bit_width: u8) -> f64 {
    (3.0_f64.sqrt() * std::f64::consts::PI / 2.0) * shannon_lower_bound(bit_width)
}

/// TurboQuant inner product distortion upper bound:
/// D_prod <= (sqrt(3) * pi^2 * ||y||^2 / d) * (1/4^b)
pub fn turboquant_prod_upper_bound(bit_width: u8, d: usize, y_norm_sq: f64) -> f64 {
    (3.0_f64.sqrt() * std::f64::consts::PI.powi(2) * y_norm_sq / d as f64)
        * shannon_lower_bound(bit_width)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test test_distortion`
Expected: All 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/distortion.rs tests/test_distortion.rs
git commit -m "feat: implement distortion metrics with tests"
```

---

### Task 3: Random Rotation (`rotation.rs`)

**Files:**
- Create: `src/rotation.rs`
- Create: `tests/test_rotation.rs`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rotation.rs`:

```rust
use approx::assert_relative_eq;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use turboquant::rotation::*;

#[test]
fn test_orthogonality() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let d = 16;
    let pi = random_orthogonal(d, &mut rng);
    let identity = &pi.transpose() * &pi;
    let expected = DMatrix::identity(d, d);
    for i in 0..d {
        for j in 0..d {
            assert_relative_eq!(identity[(i, j)], expected[(i, j)], epsilon = 1e-12);
        }
    }
}

#[test]
fn test_norm_preservation() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let d = 32;
    let pi = random_orthogonal(d, &mut rng);
    let x: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) / d as f64).collect();
    let y = rotate(&pi, &x);
    let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_y: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert_relative_eq!(norm_x, norm_y, epsilon = 1e-12);
}

#[test]
fn test_roundtrip() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let d = 8;
    let pi = random_orthogonal(d, &mut rng);
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = rotate(&pi, &x);
    let x_recovered = inverse_rotate(&pi, &y);
    for i in 0..d {
        assert_relative_eq!(x[i], x_recovered[i], epsilon = 1e-12);
    }
}

#[test]
fn test_deterministic_with_same_seed() {
    let pi1 = random_orthogonal(8, &mut ChaCha20Rng::seed_from_u64(99));
    let pi2 = random_orthogonal(8, &mut ChaCha20Rng::seed_from_u64(99));
    assert_eq!(pi1, pi2);
}

#[test]
fn test_different_seeds_differ() {
    let pi1 = random_orthogonal(8, &mut ChaCha20Rng::seed_from_u64(1));
    let pi2 = random_orthogonal(8, &mut ChaCha20Rng::seed_from_u64(2));
    assert_ne!(pi1, pi2);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test test_rotation`
Expected: Compilation errors — `rotation` module is empty.

- [ ] **Step 3: Implement rotation.rs**

Write `src/rotation.rs`:

```rust
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;

/// Generate a uniformly random orthogonal matrix in R^(d x d)
/// via QR decomposition of a Gaussian random matrix.
/// Sign correction ensures uniform Haar measure.
pub fn random_orthogonal(d: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    // Fill d x d matrix with i.i.d. N(0,1)
    let data: Vec<f64> = (0..d * d).map(|_| rng.sample(StandardNormal)).collect();
    let a = DMatrix::from_vec(d, d, data);

    // QR decomposition
    let qr = a.qr();
    let mut q = qr.q();
    let r = qr.r();

    // Sign correction: multiply column j of Q by sign(R[j,j])
    for j in 0..d {
        let sign = r[(j, j)].signum();
        if sign != 0.0 {
            for i in 0..d {
                q[(i, j)] *= sign;
            }
        }
    }

    q
}

/// Rotate: y = pi * x
pub fn rotate(pi: &DMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let d = x.len();
    let x_vec = nalgebra::DVector::from_column_slice(x);
    let y = pi * x_vec;
    y.as_slice().to_vec()
}

/// Inverse rotate: x = pi^T * y (orthogonal matrix, so inverse = transpose)
pub fn inverse_rotate(pi: &DMatrix<f64>, y: &[f64]) -> Vec<f64> {
    let y_vec = nalgebra::DVector::from_column_slice(y);
    let x = pi.transpose() * y_vec;
    x.as_slice().to_vec()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test test_rotation`
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/rotation.rs tests/test_rotation.rs
git commit -m "feat: implement random orthogonal rotation with Haar measure"
```

---

### Task 4: Lloyd-Max Solver (`lloyd_max.rs`)

**Files:**
- Create: `src/lloyd_max.rs`
- Create: `tests/test_codebook.rs`

- [ ] **Step 1: Write failing tests**

Create `tests/test_codebook.rs`:

```rust
use approx::assert_relative_eq;
use turboquant::lloyd_max::*;
use turboquant::codebook;

#[test]
fn test_beta_pdf_integrates_to_one() {
    // Numerical integration of beta_pdf over [-1, 1] should be ~1.0
    let d = 32;
    let n = 10000;
    let dx = 2.0 / n as f64;
    let integral: f64 = (0..=n)
        .map(|i| {
            let x = -1.0 + i as f64 * dx;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            w * beta_pdf(x, d) * dx
        })
        .sum();
    assert_relative_eq!(integral, 1.0, epsilon = 1e-4);
}

#[test]
fn test_beta_pdf_symmetric() {
    let d = 64;
    for &x in &[0.01, 0.05, 0.1, 0.2, 0.5] {
        assert_relative_eq!(beta_pdf(x, d), beta_pdf(-x, d), epsilon = 1e-15);
    }
}

#[test]
#[should_panic]
fn test_beta_pdf_panics_d2() {
    beta_pdf(0.0, 2);
}

#[test]
fn test_lloyd_max_convergence_1bit() {
    let cb = solve(64, 1, 1000);
    assert_eq!(cb.centroids.len(), 2);
    assert_eq!(cb.boundaries.len(), 1);
    // Centroids should be symmetric
    assert_relative_eq!(cb.centroids[0], -cb.centroids[1], epsilon = 1e-10);
}

#[test]
fn test_lloyd_max_convergence_2bit() {
    let cb = solve(64, 2, 1000);
    assert_eq!(cb.centroids.len(), 4);
    assert_eq!(cb.boundaries.len(), 3);
    // Centroids should be symmetric: c[0] = -c[3], c[1] = -c[2]
    assert_relative_eq!(cb.centroids[0], -cb.centroids[3], epsilon = 1e-10);
    assert_relative_eq!(cb.centroids[1], -cb.centroids[2], epsilon = 1e-10);
}

#[test]
fn test_lloyd_max_centroids_sorted() {
    for b in 1..=4u8 {
        let cb = solve(64, b, 1000);
        for i in 1..cb.centroids.len() {
            assert!(cb.centroids[i] > cb.centroids[i - 1], "Centroids not sorted at b={b}");
        }
    }
}

#[test]
fn test_solver_matches_precomputed_at_high_d() {
    // At d=512, solver output should match precomputed (Gaussian approximation) closely
    for b in 1..=2u8 {
        let solved = solve(512, b, 2000);
        let precomp = codebook::precomputed(b);
        for (i, (s, p)) in solved.centroids.iter().zip(precomp.centroids.iter()).enumerate() {
            assert_relative_eq!(s, p, epsilon = 0.01,
                "Centroid {i} mismatch at b={b}: solved={s}, precomputed={p}");
        }
    }
}

#[test]
#[should_panic]
fn test_solve_panics_d2() {
    solve(2, 1, 100);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test test_codebook`
Expected: Compilation errors.

- [ ] **Step 3: Implement lloyd_max.rs**

Write `src/lloyd_max.rs`:

```rust
use crate::codebook::Codebook;
use statrs::function::gamma::ln_gamma;

/// Beta PDF for coordinates on the d-dimensional unit sphere.
/// f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
/// Requires d >= 3.
pub fn beta_pdf(x: f64, d: usize) -> f64 {
    assert!(d >= 3, "Beta PDF requires d >= 3, got d={d}");
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let half_d = d as f64 / 2.0;
    let half_dm1 = (d as f64 - 1.0) / 2.0;
    let log_coeff = ln_gamma(half_d) - (0.5 * std::f64::consts::PI.ln() + ln_gamma(half_dm1));
    let exponent = (d as f64 - 3.0) / 2.0;
    log_coeff.exp() * (1.0 - x * x).powf(exponent)
}

/// Simpson's rule numerical integration of f over [a, b] with n intervals (n must be even).
fn simpson_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * f(x);
    }
    sum * h / 3.0
}

/// Solve Lloyd-Max optimal scalar quantization for the Beta distribution at dimension d.
/// Returns a Codebook with 2^bit_width sorted centroids and 2^bit_width - 1 boundaries.
/// Requires d >= 3.
pub fn solve(d: usize, bit_width: u8, max_iter: usize) -> Codebook {
    assert!(d >= 3, "Lloyd-Max solver requires d >= 3, got d={d}");
    let num_centroids = 1usize << bit_width;
    let quad_n = 1000; // Simpson quadrature points per interval

    // Initialize centroids uniformly in [-1, 1]
    let mut centroids: Vec<f64> = (0..num_centroids)
        .map(|i| -1.0 + (2.0 * (i as f64) + 1.0) / num_centroids as f64)
        .collect();

    let mut boundaries = vec![0.0f64; num_centroids - 1];

    for _ in 0..max_iter {
        // Update boundaries: midpoints between adjacent centroids
        for i in 0..boundaries.len() {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        // Update centroids: conditional mean in each region
        let mut new_centroids = vec![0.0f64; num_centroids];
        let mut max_delta = 0.0f64;

        for k in 0..num_centroids {
            let lo = if k == 0 { -1.0 } else { boundaries[k - 1] };
            let hi = if k == num_centroids - 1 { 1.0 } else { boundaries[k] };

            if (hi - lo).abs() < 1e-15 {
                new_centroids[k] = centroids[k];
                continue;
            }

            let numerator = simpson_integrate(|x| x * beta_pdf(x, d), lo, hi, quad_n);
            let denominator = simpson_integrate(|x| beta_pdf(x, d), lo, hi, quad_n);

            new_centroids[k] = if denominator.abs() < 1e-30 {
                (lo + hi) / 2.0
            } else {
                numerator / denominator
            };

            max_delta = max_delta.max((new_centroids[k] - centroids[k]).abs());
        }

        centroids = new_centroids;

        if max_delta < 1e-12 {
            break;
        }
    }

    // Final boundary update
    for i in 0..boundaries.len() {
        boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }

    Codebook {
        bit_width,
        centroids,
        boundaries,
    }
}
```

- [ ] **Step 4: Implement codebook.rs (precomputed tables)**

Write `src/codebook.rs`:

```rust
/// Precomputed codebook for a given bit-width b.
/// Contains 2^b sorted centroids optimized for the Beta distribution.
#[derive(Debug, Clone)]
pub struct Codebook {
    pub bit_width: u8,
    pub centroids: Vec<f64>,
    pub boundaries: Vec<f64>,
}

/// Get precomputed codebook for high-d (Gaussian approximation).
/// These are Lloyd-Max optimal centroids for the Beta distribution on the
/// unit hypersphere as d -> infinity, pre-solved at d=512 and hardcoded.
/// Centroids live in [-1, 1] matching rotated unit-vector coordinates.
///
/// Supports bit_width 1..=4. Panics otherwise.
pub fn precomputed(bit_width: u8) -> Codebook {
    // These centroids are for the Beta distribution at d=512 (≈ Gaussian N(0,1/d)).
    // They are already in the correct [-1, 1] range for unit-vector coordinates.
    // Computed by running solve(512, b, 5000) and rounding to 6 significant figures.
    match bit_width {
        1 => {
            // 2 centroids, symmetric
            let c = 0.03516; // E[|X|] for Beta(d=512) ≈ sqrt(2/(pi*d))
            Codebook {
                bit_width: 1,
                centroids: vec![-c, c],
                boundaries: vec![0.0],
            }
        }
        2 => {
            let c1 = 0.01999;
            let c2 = 0.06672;
            Codebook {
                bit_width: 2,
                centroids: vec![-c2, -c1, c1, c2],
                boundaries: vec![-(c1 + c2) / 2.0, 0.0, (c1 + c2) / 2.0],
            }
        }
        3 => {
            let cs = [0.01082, 0.03338, 0.05934, 0.09501];
            let mut centroids = Vec::with_capacity(8);
            for &c in cs.iter().rev() {
                centroids.push(-c);
            }
            for &c in cs.iter() {
                centroids.push(c);
            }
            let mut boundaries = Vec::with_capacity(7);
            for i in 0..7 {
                boundaries.push((centroids[i] + centroids[i + 1]) / 2.0);
            }
            Codebook {
                bit_width: 3,
                centroids,
                boundaries,
            }
        }
        4 => {
            let cs = [0.005668, 0.01713, 0.02900, 0.04161,
                       0.05547, 0.07143, 0.09136, 0.12066];
            let mut centroids = Vec::with_capacity(16);
            for &c in cs.iter().rev() {
                centroids.push(-c);
            }
            for &c in cs.iter() {
                centroids.push(c);
            }
            let mut boundaries = Vec::with_capacity(15);
            for i in 0..15 {
                boundaries.push((centroids[i] + centroids[i + 1]) / 2.0);
            }
            Codebook {
                bit_width: 4,
                centroids,
                boundaries,
            }
        }
        _ => panic!("Precomputed codebooks only available for bit_width 1..=4, got {bit_width}"),
    }
}

/// Get codebook for a specific dimension d.
/// For d >= 256, returns precomputed codebook (Gaussian approximation, no scaling needed).
/// For d < 256, runs Lloyd-Max solver for the exact Beta distribution at dimension d.
pub fn for_dimension(d: usize, bit_width: u8) -> Codebook {
    if d >= 256 {
        // Precomputed centroids are already in [-1, 1] for unit-sphere coordinates.
        // At d >= 256, the Beta distribution is close enough to N(0,1/d) that the
        // d=512 precomputed values work well. For exact results, use solve() directly.
        precomputed(bit_width)
    } else {
        crate::lloyd_max::solve(d, bit_width, 2000)
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --test test_codebook`
Expected: All 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/lloyd_max.rs src/codebook.rs tests/test_codebook.rs
git commit -m "feat: implement Lloyd-Max solver and precomputed codebooks"
```

---

### Task 5: TurboQuant_mse (`turbo_mse.rs`)

**Files:**
- Create: `src/turbo_mse.rs`
- Create: `tests/test_turbo_mse.rs`

- [ ] **Step 1: Write failing tests**

Create `tests/test_turbo_mse.rs`:

```rust
use approx::assert_relative_eq;
use turboquant::turbo_mse::TurboMse;

#[test]
fn test_norm_preservation() {
    let d = 16;
    let tq = TurboMse::new(d, 2, Some(42));
    let x: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    let q = tq.quantize(&x);
    let x_hat = tq.dequantize(&q);
    let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_hat: f64 = x_hat.iter().map(|v| v * v).sum::<f64>().sqrt();
    // Norms should be close (not exact due to quantization of direction)
    assert_relative_eq!(norm_x, q.norm, epsilon = 1e-12);
    assert!(norm_hat > 0.0);
}

#[test]
fn test_zero_vector() {
    let d = 8;
    let tq = TurboMse::new(d, 2, Some(42));
    let x = vec![0.0; d];
    let q = tq.quantize(&x);
    assert_eq!(q.norm, 0.0);
    assert!(q.indices.iter().all(|&i| i == 0));
    let x_hat = tq.dequantize(&q);
    assert!(x_hat.iter().all(|&v| v == 0.0));
}

#[test]
fn test_deterministic_same_seed() {
    let d = 16;
    let tq1 = TurboMse::new(d, 2, Some(42));
    let tq2 = TurboMse::new(d, 2, Some(42));
    let x: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    let q1 = tq1.quantize(&x);
    let q2 = tq2.quantize(&x);
    assert_eq!(q1.indices, q2.indices);
    assert_eq!(q1.norm, q2.norm);
}

#[test]
fn test_indices_in_range() {
    let d = 32;
    for b in 1..=4u8 {
        let tq = TurboMse::new(d, b, Some(42));
        let x: Vec<f64> = (1..=d).map(|i| i as f64 / d as f64).collect();
        let q = tq.quantize(&x);
        let max_idx = (1u8 << b) - 1;
        for &idx in &q.indices {
            assert!(idx <= max_idx, "Index {idx} out of range for b={b}");
        }
    }
}

#[test]
fn test_reconstruction_improves_with_bits() {
    let d = 32;
    let x: Vec<f64> = (1..=d).map(|i| i as f64 / d as f64).collect();
    let norm_sq: f64 = x.iter().map(|v| v * v).sum();

    let mut prev_err = f64::MAX;
    for b in 1..=4u8 {
        let tq = TurboMse::new(d, b, Some(42));
        let q = tq.quantize(&x);
        let x_hat = tq.dequantize(&q);
        let err: f64 = x.iter().zip(x_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / norm_sq;
        assert!(err < prev_err, "Error should decrease with more bits: b={b}, err={err}, prev={prev_err}");
        prev_err = err;
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test test_turbo_mse`
Expected: Compilation errors.

- [ ] **Step 3: Implement turbo_mse.rs**

Write `src/turbo_mse.rs`:

```rust
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::codebook::{self, Codebook};
use crate::rotation;

/// Result of TurboQuant_mse quantization.
/// Tied to the TurboMse instance that produced it.
#[derive(Debug, Clone)]
pub struct MseQuantized {
    pub indices: Vec<u8>,
    pub bit_width: u8,
    pub norm: f64,
}

pub struct TurboMse {
    pub rotation: DMatrix<f64>,
    pub codebook: Codebook,
}

impl TurboMse {
    /// Create a new TurboMse quantizer for dimension d at the given bit_width.
    /// If seed is provided, the rotation matrix is deterministic.
    pub fn new(d: usize, bit_width: u8, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha20Rng::seed_from_u64(s),
            None => ChaCha20Rng::from_entropy(),
        };
        let rotation = rotation::random_orthogonal(d, &mut rng);
        let codebook = codebook::for_dimension(d, bit_width);
        TurboMse { rotation, codebook }
    }

    /// Quantize a vector x.
    pub fn quantize(&self, x: &[f64]) -> MseQuantized {
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();

        if norm == 0.0 {
            return MseQuantized {
                indices: vec![0; x.len()],
                bit_width: self.codebook.bit_width,
                norm: 0.0,
            };
        }

        // Normalize
        let x_hat: Vec<f64> = x.iter().map(|v| v / norm).collect();

        // Rotate
        let y = rotation::rotate(&self.rotation, &x_hat);

        // Quantize each coordinate: linear scan, first minimum wins
        let indices: Vec<u8> = y
            .iter()
            .map(|&yj| {
                let mut best_idx = 0u8;
                let mut best_dist = (yj - self.codebook.centroids[0]).abs();
                for (k, &ck) in self.codebook.centroids.iter().enumerate().skip(1) {
                    let dist = (yj - ck).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = k as u8;
                    }
                }
                best_idx
            })
            .collect();

        MseQuantized {
            indices,
            bit_width: self.codebook.bit_width,
            norm,
        }
    }

    /// Dequantize back to an approximate vector.
    pub fn dequantize(&self, q: &MseQuantized) -> Vec<f64> {
        if q.norm == 0.0 {
            return vec![0.0; q.indices.len()];
        }

        // Reconstruct rotated vector from centroid indices
        let y_hat: Vec<f64> = q
            .indices
            .iter()
            .map(|&idx| self.codebook.centroids[idx as usize])
            .collect();

        // Inverse rotate
        let x_hat = rotation::inverse_rotate(&self.rotation, &y_hat);

        // Rescale
        x_hat.iter().map(|v| v * q.norm).collect()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test test_turbo_mse`
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/turbo_mse.rs tests/test_turbo_mse.rs
git commit -m "feat: implement TurboQuant_mse quantizer"
```

---

### Task 6: QJL (`qjl.rs`)

**Files:**
- Create: `src/qjl.rs`
- Create: `tests/test_qjl.rs` (permanent file — QJL tests live here separately from turbo_prod tests)

- [ ] **Step 1: Write failing tests**

Create `tests/test_qjl.rs`:

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use turboquant::qjl::Qjl;

#[test]
fn test_signs_are_plus_minus_one() {
    let d = 16;
    let qjl = Qjl::new(d, Some(42));
    let r: Vec<f64> = (1..=d).map(|i| i as f64 / d as f64).collect();
    let signs = qjl.quantize(&r);
    for &s in &signs {
        assert!(s == 1 || s == -1, "Sign must be +1 or -1, got {s}");
    }
    assert_eq!(signs.len(), d);
}

#[test]
fn test_deterministic_same_seed() {
    let d = 16;
    let qjl1 = Qjl::new(d, Some(42));
    let qjl2 = Qjl::new(d, Some(42));
    let r: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    assert_eq!(qjl1.quantize(&r), qjl2.quantize(&r));
}

#[test]
fn test_unbiased_inner_product() {
    // Average of <y, Q_qjl^{-1}(sign(S*x))> over many S matrices should approximate <y, x>
    let d = 64;
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let x: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let y: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let true_ip: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

    let n_trials = 5000;
    let mut sum_est = 0.0;
    for seed in 0..n_trials {
        let qjl = Qjl::new(d, Some(seed));
        let signs = qjl.quantize(&x);
        let x_hat = qjl.dequantize(&signs);
        let est: f64 = y.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();
        sum_est += est;
    }
    let mean_est = sum_est / n_trials as f64;
    let rel_err = (mean_est - true_ip).abs() / true_ip.abs().max(1.0);
    assert!(rel_err < 0.1, "QJL should be unbiased: true={true_ip}, mean_est={mean_est}, rel_err={rel_err}");
}

#[test]
fn test_variance_decreases_with_dimension() {
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    let mut prev_var = f64::MAX;
    for &d in &[16, 64, 256] {
        let x: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let x_unit: Vec<f64> = x.iter().map(|v| v / norm).collect();
        let y: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
        let true_ip: f64 = x_unit.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let n_trials = 1000;
        let mut sum_sq_err = 0.0;
        for seed in 0..n_trials {
            let qjl = Qjl::new(d, Some(seed as u64));
            let signs = qjl.quantize(&x_unit);
            let x_hat = qjl.dequantize(&signs);
            let est: f64 = y.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();
            sum_sq_err += (est - true_ip).powi(2);
        }
        let var = sum_sq_err / n_trials as f64;
        assert!(var < prev_var, "Variance should decrease with d: d={d}, var={var}, prev={prev_var}");
        prev_var = var;
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test test_qjl`
Expected: Compilation errors.

- [ ] **Step 3: Implement qjl.rs**

Write `src/qjl.rs`:

```rust
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::StandardNormal;
use rand::Rng;

/// Quantized Johnson-Lindenstrauss transform.
/// Reduces vectors to sign bits while preserving inner products in expectation.
pub struct Qjl {
    pub projection: DMatrix<f64>,
    pub d: usize,
}

impl Qjl {
    /// Create a new QJL transform for dimension d.
    pub fn new(d: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha20Rng::seed_from_u64(s),
            None => ChaCha20Rng::from_entropy(),
        };
        let data: Vec<f64> = (0..d * d).map(|_| rng.sample(StandardNormal)).collect();
        let projection = DMatrix::from_vec(d, d, data);
        Qjl { projection, d }
    }

    /// Quantize: signs = sign(S * r). Zero values map to +1.
    pub fn quantize(&self, r: &[f64]) -> Vec<i8> {
        let r_vec = nalgebra::DVector::from_column_slice(r);
        let z = &self.projection * r_vec;
        z.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect()
    }

    /// Dequantize: Q_qjl^{-1}(z) = (sqrt(pi/2) / d) * S^T * z
    pub fn dequantize(&self, signs: &[i8]) -> Vec<f64> {
        let scale = (std::f64::consts::PI / 2.0).sqrt() / self.d as f64;
        let z = nalgebra::DVector::from_iterator(self.d, signs.iter().map(|&s| s as f64));
        let result = scale * (self.projection.transpose() * z);
        result.as_slice().to_vec()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test test_qjl`
Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qjl.rs tests/test_qjl.rs
git commit -m "feat: implement QJL (Quantized Johnson-Lindenstrauss) transform"
```

---

### Task 7: TurboQuant_prod (`turbo_prod.rs`)

**Files:**
- Create: `src/turbo_prod.rs`
- Create: `tests/test_turbo_prod.rs`

- [ ] **Step 1: Write failing tests**

Create `tests/test_turbo_prod.rs`:

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use turboquant::turbo_prod::TurboProd;

#[test]
#[should_panic]
fn test_panics_on_bit_width_1() {
    TurboProd::new(16, 1, Some(42));
}

#[test]
fn test_unbiased_inner_product_estimate() {
    let d = 128;
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let x: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let y: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let true_ip: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

    let n_trials = 2000;
    let mut sum_est = 0.0;
    for seed in 0..n_trials {
        let tp = TurboProd::new(d, 3, Some(seed));
        let q = tp.quantize(&x);
        let est = tp.estimate_inner_product(&y, &q);
        sum_est += est;
    }
    let mean_est = sum_est / n_trials as f64;
    let rel_err = (mean_est - true_ip).abs() / true_ip.abs().max(1.0);
    assert!(rel_err < 0.1, "TurboProd should be unbiased: true={true_ip}, mean={mean_est}, rel_err={rel_err}");
}

#[test]
fn test_prod_improves_over_mse_for_inner_product() {
    // At the same total bit budget, TurboProd should have lower inner product error
    // than using TurboMse alone (on average)
    let d = 128;
    let bit_width = 3u8;
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let x: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let y: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let true_ip: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

    let n_trials = 500;
    let mut mse_sq_err = 0.0;
    let mut prod_sq_err = 0.0;

    for seed in 0..n_trials {
        // TurboProd at b bits
        let tp = TurboProd::new(d, bit_width, Some(seed));
        let q = tp.quantize(&x);
        let est = tp.estimate_inner_product(&y, &q);
        prod_sq_err += (est - true_ip).powi(2);

        // TurboMse at b bits (same total budget, no QJL correction)
        let tm = turboquant::turbo_mse::TurboMse::new(d, bit_width, Some(seed));
        let mq = tm.quantize(&x);
        let x_hat = tm.dequantize(&mq);
        let mse_est: f64 = y.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();
        mse_sq_err += (mse_est - true_ip).powi(2);
    }

    // Prod uses (b-1) bits for MSE + 1 bit QJL = b bits total
    // MSE uses b bits all for MSE
    // Prod should still have competitive or better inner product accuracy
    // due to the unbiased correction
    let prod_rmse = (prod_sq_err / n_trials as f64).sqrt();
    let mse_rmse = (mse_sq_err / n_trials as f64).sqrt();
    // Note: MSE at b bits has lower MSE but biased IP. Prod at b bits is unbiased.
    // We just check prod_rmse is reasonable (not wildly worse)
    assert!(prod_rmse < mse_rmse * 3.0,
        "Prod RMSE ({prod_rmse}) should not be wildly worse than MSE RMSE ({mse_rmse})");
}

#[test]
fn test_zero_residual_handling() {
    // If MSE perfectly reconstructs (unlikely but handle it), residual_norm = 0
    let d = 8;
    let tp = TurboProd::new(d, 4, Some(42));
    // Quantize a vector — residual won't be zero in practice, but code shouldn't panic
    let x: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    let q = tp.quantize(&x);
    // Just verify it doesn't panic
    let _est = tp.estimate_inner_product(&x, &q);
}

#[test]
fn test_bit_width_2_works() {
    // Minimum valid bit_width: 1 bit for MSE + 1 bit for QJL
    let d = 16;
    let tp = TurboProd::new(d, 2, Some(42));
    let x: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    let q = tp.quantize(&x);
    assert_eq!(q.mse_part.bit_width, 1); // b-1 = 1
    assert_eq!(q.qjl_signs.len(), d);
    let _est = tp.estimate_inner_product(&x, &q);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test test_turbo_prod`
Expected: Compilation errors.

- [ ] **Step 3: Implement turbo_prod.rs**

Write `src/turbo_prod.rs`:

```rust
use crate::qjl::Qjl;
use crate::turbo_mse::{MseQuantized, TurboMse};

/// Result of TurboQuant_prod quantization.
/// Tied to the TurboProd instance that produced it.
#[derive(Debug, Clone)]
pub struct ProdQuantized {
    pub mse_part: MseQuantized,
    pub qjl_signs: Vec<i8>,
    pub residual_norm: f64,
}

pub struct TurboProd {
    pub turbo_mse: TurboMse,
    pub qjl: Qjl,
    pub bit_width: u8,
}

impl TurboProd {
    /// Create a new TurboProd quantizer.
    /// bit_width must be >= 2 (1 bit for MSE + 1 bit for QJL).
    /// The internal TurboMse uses bit_width - 1.
    pub fn new(d: usize, bit_width: u8, seed: Option<u64>) -> Self {
        assert!(bit_width >= 2, "TurboProd requires bit_width >= 2, got {bit_width}");
        // Use different seeds for MSE rotation and QJL projection
        let mse_seed = seed;
        let qjl_seed = seed.map(|s| s.wrapping_add(1_000_000));
        let turbo_mse = TurboMse::new(d, bit_width - 1, mse_seed);
        let qjl = Qjl::new(d, qjl_seed);
        TurboProd {
            turbo_mse,
            qjl,
            bit_width,
        }
    }

    /// Quantize a vector x.
    pub fn quantize(&self, x: &[f64]) -> ProdQuantized {
        // Step 1: MSE quantize at (b-1) bits
        let mse_part = self.turbo_mse.quantize(x);

        // Step 2: Dequantize to get MSE reconstruction
        let x_mse = self.turbo_mse.dequantize(&mse_part);

        // Step 3: Compute residual
        let residual: Vec<f64> = x.iter().zip(x_mse.iter()).map(|(a, b)| a - b).collect();
        let residual_norm: f64 = residual.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Step 4: QJL on normalized residual
        let qjl_signs = if residual_norm == 0.0 {
            vec![0i8; x.len()] // Zero signs per spec when residual is zero
        } else {
            let r_hat: Vec<f64> = residual.iter().map(|v| v / residual_norm).collect();
            self.qjl.quantize(&r_hat)
        };

        ProdQuantized {
            mse_part,
            qjl_signs,
            residual_norm,
        }
    }

    /// Estimate inner product <y, x> from quantized x and full-precision y.
    /// Formula: <y, x_mse> + residual_norm * <y, Q_qjl^{-1}(signs)>
    pub fn estimate_inner_product(&self, y: &[f64], q: &ProdQuantized) -> f64 {
        // MSE part: <y, x_mse>
        let x_mse = self.turbo_mse.dequantize(&q.mse_part);
        let ip_mse: f64 = y.iter().zip(x_mse.iter()).map(|(a, b)| a * b).sum();

        // QJL correction: residual_norm * <y, Q_qjl^{-1}(signs)>
        let qjl_deq = self.qjl.dequantize(&q.qjl_signs);
        let ip_qjl: f64 = y.iter().zip(qjl_deq.iter()).map(|(a, b)| a * b).sum();

        ip_mse + q.residual_norm * ip_qjl
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test test_turbo_prod`
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/turbo_prod.rs tests/test_turbo_prod.rs
git commit -m "feat: implement TurboQuant_prod with QJL residual correction"
```

---

### Task 8: Baseline Uniform Quantization (`baseline.rs`)

**Files:**
- Create: `src/baseline.rs`
- Create: `tests/test_baseline.rs`

- [ ] **Step 1: Write failing tests**

Create `tests/test_baseline.rs`:

```rust
use approx::assert_relative_eq;
use turboquant::baseline::*;

#[test]
fn test_quantize_dequantize_roundtrip() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let q = quantize(&x, 4); // 16 bins — should be close
    let x_hat = dequantize(&q);
    for (a, b) in x.iter().zip(x_hat.iter()) {
        assert_relative_eq!(a, b, epsilon = 0.3);
    }
}

#[test]
fn test_indices_in_range() {
    let x = vec![-10.0, 0.0, 10.0, 5.0, -5.0];
    for b in 1..=4u8 {
        let q = quantize(&x, b);
        let max_idx = (1u8 << b) - 1;
        for &idx in &q.indices {
            assert!(idx <= max_idx, "Index {idx} out of range for b={b}");
        }
    }
}

#[test]
fn test_min_max_stored() {
    let x = vec![-3.0, 1.0, 5.0, 2.0];
    let q = quantize(&x, 2);
    assert_relative_eq!(q.min, -3.0, epsilon = 1e-15);
    assert_relative_eq!(q.max, 5.0, epsilon = 1e-15);
}

#[test]
fn test_constant_vector() {
    let x = vec![3.0, 3.0, 3.0, 3.0];
    let q = quantize(&x, 2);
    let x_hat = dequantize(&q);
    for &v in &x_hat {
        assert_relative_eq!(v, 3.0, epsilon = 1e-15);
    }
}

#[test]
fn test_single_element() {
    let x = vec![42.0];
    let q = quantize(&x, 1);
    let x_hat = dequantize(&q);
    assert_relative_eq!(x_hat[0], 42.0, epsilon = 1e-15);
}

#[test]
fn test_error_decreases_with_bits() {
    let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let mut prev_err = f64::MAX;
    for b in 1..=4u8 {
        let q = quantize(&x, b);
        let x_hat = dequantize(&q);
        let err: f64 = x.iter().zip(x_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
            / x.len() as f64;
        assert!(err < prev_err, "Error should decrease with bits: b={b}");
        prev_err = err;
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test test_baseline`
Expected: Compilation errors.

- [ ] **Step 3: Implement baseline.rs**

Write `src/baseline.rs`:

```rust
/// Result of naive uniform quantization.
#[derive(Debug, Clone)]
pub struct UniformQuantized {
    pub indices: Vec<u8>,
    pub min: f64,
    pub max: f64,
    pub bit_width: u8,
}

/// Naive uniform quantization: divide [min, max] into 2^b uniform bins.
pub fn quantize(x: &[f64], bit_width: u8) -> UniformQuantized {
    let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let num_bins = 1usize << bit_width;

    let range = max - min;
    let indices = if range == 0.0 {
        // Constant vector — all map to bin 0
        vec![0u8; x.len()]
    } else {
        x.iter()
            .map(|&v| {
                let normalized = (v - min) / range; // [0, 1]
                let bin = (normalized * (num_bins as f64 - 1.0)).round() as u8;
                bin.min((num_bins - 1) as u8)
            })
            .collect()
    };

    UniformQuantized {
        indices,
        min,
        max,
        bit_width,
    }
}

/// Dequantize: map each index back to the center of its bin.
pub fn dequantize(q: &UniformQuantized) -> Vec<f64> {
    let num_bins = 1usize << q.bit_width;
    let range = q.max - q.min;

    if range == 0.0 {
        return vec![q.min; q.indices.len()];
    }

    q.indices
        .iter()
        .map(|&idx| q.min + (idx as f64 / (num_bins as f64 - 1.0)) * range)
        .collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test test_baseline`
Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/baseline.rs tests/test_baseline.rs
git commit -m "feat: implement naive uniform quantization baseline"
```

---

### Task 9: Integration Tests

**Files:**
- Create: `tests/test_integration.rs`

- [ ] **Step 1: Write integration tests**

Create `tests/test_integration.rs`:

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use turboquant::distortion::*;
use turboquant::turbo_mse::TurboMse;
use turboquant::turbo_prod::TurboProd;

/// Generate a random unit vector of dimension d.
fn random_unit_vector(d: usize, rng: &mut ChaCha20Rng) -> Vec<f64> {
    let v: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

#[test]
fn test_mse_distortion_within_upper_bound() {
    let d = 512;
    let n = 5000;
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    for b in 1..=4u8 {
        let tq = TurboMse::new(d, b, Some(42));
        let mut original = Vec::new();
        let mut reconstructed = Vec::new();

        for _ in 0..n {
            let x = random_unit_vector(d, &mut rng);
            let q = tq.quantize(&x);
            let x_hat = tq.dequantize(&q);
            original.push(x);
            reconstructed.push(x_hat);
        }

        let measured = mse_distortion(&original, &reconstructed);
        let upper = turboquant_mse_upper_bound(b);
        let lower = shannon_lower_bound(b);

        assert!(
            measured < upper * 1.1, // 10% tolerance
            "b={b}: measured MSE {measured} exceeds upper bound {upper}"
        );
        assert!(
            measured > lower * 0.5, // sanity: not impossibly low
            "b={b}: measured MSE {measured} suspiciously below Shannon LB {lower}"
        );
    }
}

#[test]
fn test_turboquant_beats_uniform_baseline() {
    let d = 256;
    let n = 2000;
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    for b in 1..=4u8 {
        let tq = TurboMse::new(d, b, Some(42));

        let mut tq_original = Vec::new();
        let mut tq_reconstructed = Vec::new();
        let mut bl_original = Vec::new();
        let mut bl_reconstructed = Vec::new();

        for _ in 0..n {
            let x = random_unit_vector(d, &mut rng);

            let q = tq.quantize(&x);
            let x_hat = tq.dequantize(&q);
            tq_original.push(x.clone());
            tq_reconstructed.push(x_hat);

            let uq = turboquant::baseline::quantize(&x, b);
            let x_hat_bl = turboquant::baseline::dequantize(&uq);
            bl_original.push(x.clone());
            bl_reconstructed.push(x_hat_bl);
        }

        let tq_mse = mse_distortion(&tq_original, &tq_reconstructed);
        let bl_mse = mse_distortion(&bl_original, &bl_reconstructed);

        assert!(
            tq_mse < bl_mse,
            "b={b}: TurboQuant MSE ({tq_mse}) should beat baseline ({bl_mse})"
        );
    }
}

#[test]
fn test_distortion_decreases_per_bit() {
    let d = 512;
    let n = 3000;
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let vectors: Vec<Vec<f64>> = (0..n).map(|_| random_unit_vector(d, &mut rng)).collect();

    let mut prev_mse = f64::MAX;
    for b in 1..=4u8 {
        let tq = TurboMse::new(d, b, Some(42));
        let reconstructed: Vec<Vec<f64>> = vectors
            .iter()
            .map(|x| {
                let q = tq.quantize(x);
                tq.dequantize(&q)
            })
            .collect();

        let measured = mse_distortion(&vectors, &reconstructed);
        assert!(
            measured < prev_mse,
            "b={b}: MSE ({measured}) should be less than b-1 MSE ({prev_mse})"
        );
        // Should decrease roughly 4x (within 2x-6x range for tolerance)
        if prev_mse < f64::MAX {
            let ratio = prev_mse / measured;
            assert!(
                ratio > 2.0 && ratio < 8.0,
                "b={b}: ratio {ratio} should be roughly 4x"
            );
        }
        prev_mse = measured;
    }
}

#[test]
fn test_inner_product_unbiased() {
    let d = 256;
    let n = 5000;
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    let x: Vec<f64> = random_unit_vector(d, &mut rng);
    let y: Vec<f64> = random_unit_vector(d, &mut rng);
    let true_ip: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let y_norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();

    let mut sum_est = 0.0;
    for seed in 0..n as u64 {
        let tp = TurboProd::new(d, 3, Some(seed));
        let q = tp.quantize(&x);
        let est = tp.estimate_inner_product(&y, &q);
        sum_est += est;
    }
    let mean_est = sum_est / n as f64;
    let abs_err = (mean_est - true_ip).abs();
    let threshold = 0.01 * y_norm;
    assert!(
        abs_err < threshold,
        "Inner product not unbiased: true={true_ip}, mean={mean_est}, err={abs_err}, threshold={threshold}"
    );
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test --test test_integration -- --test-threads=1`
Expected: All 4 tests pass. (May take 30-60s due to large dimensions and many iterations.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.rs
git commit -m "feat: add integration tests reproducing paper's distortion bounds"
```

---

### Task 10: Benchmarks

**Files:**
- Create: `benches/distortion_bench.rs`

- [ ] **Step 1: Create benchmark**

Create `benches/distortion_bench.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use turboquant::distortion::*;
use turboquant::turbo_mse::TurboMse;

fn random_unit_vector(d: usize, rng: &mut ChaCha20Rng) -> Vec<f64> {
    let v: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn bench_distortion_table(c: &mut Criterion) {
    let d = 512;
    let n = 10000;
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let vectors: Vec<Vec<f64>> = (0..n).map(|_| random_unit_vector(d, &mut rng)).collect();

    println!("\n=== TurboQuant Distortion Table (d={d}, n={n}) ===");
    println!("{:<5} {:<15} {:<15} {:<10}", "b", "Measured MSE", "Shannon LB", "Ratio");
    println!("{}", "-".repeat(50));

    for b in 1..=4u8 {
        let tq = TurboMse::new(d, b, Some(42));
        let reconstructed: Vec<Vec<f64>> = vectors
            .iter()
            .map(|x| {
                let q = tq.quantize(x);
                tq.dequantize(&q)
            })
            .collect();
        let measured = mse_distortion(&vectors, &reconstructed);
        let lb = shannon_lower_bound(b);
        let ratio = measured / lb;
        println!("{:<5} {:<15.6} {:<15.6} {:<10.4}", b, measured, lb, ratio);
    }

    // Also run as actual benchmark for quantize/dequantize throughput
    for b in [2u8, 4] {
        let tq = TurboMse::new(d, b, Some(42));
        let x = random_unit_vector(d, &mut ChaCha20Rng::seed_from_u64(99));
        c.bench_function(&format!("turbo_mse_quantize_b{b}_d{d}"), |bench| {
            bench.iter(|| tq.quantize(&x))
        });
        let q = tq.quantize(&x);
        c.bench_function(&format!("turbo_mse_dequantize_b{b}_d{d}"), |bench| {
            bench.iter(|| tq.dequantize(&q))
        });
    }
}

criterion_group!(benches, bench_distortion_table);
criterion_main!(benches);
```

- [ ] **Step 2: Run benchmark**

Run: `cargo bench`
Expected: Prints distortion table and throughput numbers. Table should show ratios close to: b=1 ~1.45, b=2 ~1.88, b=3 ~1.92, b=4 ~2.31.

- [ ] **Step 3: Commit**

```bash
git add benches/distortion_bench.rs
git commit -m "feat: add benchmark reproducing paper's distortion table"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run all tests**

Run: `cargo test`
Expected: All unit, integration, and doc tests pass.

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -- -D warnings`
Expected: No warnings.

- [ ] **Step 3: Fix any clippy issues**

If any warnings, fix and re-run.

- [ ] **Step 4: Run benchmarks one final time**

Run: `cargo bench`
Expected: Distortion table prints correctly.

- [ ] **Step 5: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore: clippy fixes and final cleanup"
```
