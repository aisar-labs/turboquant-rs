use approx::assert_relative_eq;
use turboquant::baseline::*;

#[test]
fn test_quantize_dequantize_roundtrip() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let q = quantize(&x, 4);
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
