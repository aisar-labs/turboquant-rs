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

    // Throughput benchmarks
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
