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
