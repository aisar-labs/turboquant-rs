use approx::assert_relative_eq;
use turboquant::lloyd_max::*;
use turboquant::codebook;

#[test]
fn test_beta_pdf_integrates_to_one() {
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
    assert_relative_eq!(cb.centroids[0], -cb.centroids[1], epsilon = 1e-10);
}

#[test]
fn test_lloyd_max_convergence_2bit() {
    let cb = solve(64, 2, 1000);
    assert_eq!(cb.centroids.len(), 4);
    assert_eq!(cb.boundaries.len(), 3);
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
    for b in 1..=2u8 {
        let solved = solve(512, b, 2000);
        let precomp = codebook::precomputed(b);
        for (i, (s, p)) in solved.centroids.iter().zip(precomp.centroids.iter()).enumerate() {
            assert!(
                (s - p).abs() <= 0.01,
                "Centroid {i} mismatch at b={b}: solved={s}, precomputed={p}"
            );
        }
    }
}

#[test]
#[should_panic]
fn test_solve_panics_d2() {
    solve(2, 1, 100);
}
