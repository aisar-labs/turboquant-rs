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
            measured < upper * 1.1,
            "b={b}: measured MSE {measured} exceeds upper bound {upper}"
        );
        assert!(
            measured > lower * 0.5,
            "b={b}: measured MSE {measured} suspiciously below Shannon LB {lower}"
        );
    }
}

#[test]
fn test_turboquant_beats_uniform_baseline() {
    // Use d=512 to match the precomputed codebook calibration dimension.
    let d = 512;
    let n = 2000;

    for b in 1..=4u8 {
        // Reset RNG per bit-width so all b values are evaluated on the same vectors.
        let mut rng = ChaCha20Rng::seed_from_u64(0);
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
