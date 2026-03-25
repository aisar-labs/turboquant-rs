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
    let original = vec![vec![1.0, 0.0, 0.0]];
    let reconstructed = vec![vec![0.0, 0.0, 0.0]];
    let d = mse_distortion(&original, &reconstructed);
    assert!((d - 1.0).abs() < 1e-15, "Expected 1.0, got {d}");
}

#[test]
fn test_inner_product_distortion_perfect() {
    let xs = vec![vec![1.0, 2.0]];
    let ys = vec![vec![3.0, 4.0]];
    let true_ip: f64 = 1.0 * 3.0 + 2.0 * 4.0;
    let estimates = vec![true_ip];
    let d = inner_product_distortion(&xs, &ys, &estimates);
    assert!((d - 0.0).abs() < 1e-15, "Perfect estimate should have zero distortion, got {d}");
}

#[test]
fn test_inner_product_distortion_nonnegative() {
    let xs = vec![vec![1.0, 0.0]];
    let ys = vec![vec![0.0, 1.0]];
    let estimates = vec![0.5];
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
