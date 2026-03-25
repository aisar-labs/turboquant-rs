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
fn test_zero_residual_handling() {
    let d = 8;
    let tp = TurboProd::new(d, 4, Some(42));
    let x: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    let q = tp.quantize(&x);
    let _est = tp.estimate_inner_product(&x, &q);
    // Just verify it doesn't panic
}

#[test]
fn test_bit_width_2_works() {
    let d = 16;
    let tp = TurboProd::new(d, 2, Some(42));
    let x: Vec<f64> = (1..=d).map(|i| i as f64).collect();
    let q = tp.quantize(&x);
    assert_eq!(q.mse_part.bit_width, 1); // b-1 = 1
    assert_eq!(q.qjl_signs.len(), d);
    let _est = tp.estimate_inner_product(&x, &q);
}
