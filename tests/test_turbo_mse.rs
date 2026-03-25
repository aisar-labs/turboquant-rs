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
