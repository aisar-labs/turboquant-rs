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
