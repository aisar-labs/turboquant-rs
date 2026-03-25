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
    // QJL dequantize estimates x/||x|| (the unit direction).
    // So <y, dequantize(sign(S*x_unit))> should approximate <y, x_unit>.
    let d = 64;
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let x_raw: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let x_norm: f64 = x_raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    let x_unit: Vec<f64> = x_raw.iter().map(|v| v / x_norm).collect();
    let y: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let true_ip: f64 = x_unit.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

    let n_trials = 5000;
    let mut sum_est = 0.0;
    for seed in 0..n_trials {
        let qjl = Qjl::new(d, Some(seed));
        let signs = qjl.quantize(&x_unit);
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
    // Variance of <y, Q_qjl^{-1}(sign(S*x))> should decrease with dimension
    // when both x and y are unit vectors (so ||y||^2/d decreases)
    let mut prev_var = f64::MAX;
    for &d in &[16, 64, 256] {
        // Use a fresh RNG per dimension to avoid correlation
        let mut rng = ChaCha20Rng::seed_from_u64(d as u64);
        let x_raw: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
        let x_norm: f64 = x_raw.iter().map(|v| v * v).sum::<f64>().sqrt();
        let x_unit: Vec<f64> = x_raw.iter().map(|v| v / x_norm).collect();

        let y_raw: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
        let y_norm: f64 = y_raw.iter().map(|v| v * v).sum::<f64>().sqrt();
        let y_unit: Vec<f64> = y_raw.iter().map(|v| v / y_norm).collect();

        let true_ip: f64 = x_unit.iter().zip(y_unit.iter()).map(|(a, b)| a * b).sum();

        let n_trials = 1000;
        let mut sum_sq_err = 0.0;
        for seed in 0..n_trials {
            let qjl = Qjl::new(d, Some(seed as u64 + 10000));
            let signs = qjl.quantize(&x_unit);
            let x_hat = qjl.dequantize(&signs);
            let est: f64 = y_unit.iter().zip(x_hat.iter()).map(|(a, b)| a * b).sum();
            sum_sq_err += (est - true_ip).powi(2);
        }
        let var = sum_sq_err / n_trials as f64;
        assert!(var < prev_var, "Variance should decrease with d: d={d}, var={var}, prev={prev_var}");
        prev_var = var;
    }
}
