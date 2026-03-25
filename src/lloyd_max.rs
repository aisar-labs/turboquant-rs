use crate::codebook::Codebook;
use statrs::function::gamma::ln_gamma;

/// Beta PDF for coordinates on the d-dimensional unit sphere.
/// f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
/// Requires d >= 3.
pub fn beta_pdf(x: f64, d: usize) -> f64 {
    assert!(d >= 3, "Beta PDF requires d >= 3, got d={d}");
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let half_d = d as f64 / 2.0;
    let half_dm1 = (d as f64 - 1.0) / 2.0;
    let log_coeff = ln_gamma(half_d) - (0.5 * std::f64::consts::PI.ln() + ln_gamma(half_dm1));
    let exponent = (d as f64 - 3.0) / 2.0;
    log_coeff.exp() * (1.0 - x * x).powf(exponent)
}

fn simpson_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * f(x);
    }
    sum * h / 3.0
}

/// Solve Lloyd-Max optimal scalar quantization for the Beta distribution at dimension d.
/// Requires d >= 3.
pub fn solve(d: usize, bit_width: u8, max_iter: usize) -> Codebook {
    assert!(d >= 3, "Lloyd-Max solver requires d >= 3, got d={d}");
    let num_centroids = 1usize << bit_width;
    let quad_n = 1000;

    let mut centroids: Vec<f64> = (0..num_centroids)
        .map(|i| -1.0 + (2.0 * (i as f64) + 1.0) / num_centroids as f64)
        .collect();

    let mut boundaries = vec![0.0f64; num_centroids - 1];

    for _ in 0..max_iter {
        for i in 0..boundaries.len() {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        let mut new_centroids = vec![0.0f64; num_centroids];
        let mut max_delta = 0.0f64;

        for k in 0..num_centroids {
            let lo = if k == 0 { -1.0 } else { boundaries[k - 1] };
            let hi = if k == num_centroids - 1 { 1.0 } else { boundaries[k] };

            if (hi - lo).abs() < 1e-15 {
                new_centroids[k] = centroids[k];
                continue;
            }

            let numerator = simpson_integrate(|x| x * beta_pdf(x, d), lo, hi, quad_n);
            let denominator = simpson_integrate(|x| beta_pdf(x, d), lo, hi, quad_n);

            new_centroids[k] = if denominator.abs() < 1e-30 {
                (lo + hi) / 2.0
            } else {
                numerator / denominator
            };

            max_delta = max_delta.max((new_centroids[k] - centroids[k]).abs());
        }

        centroids = new_centroids;

        if max_delta < 1e-12 {
            break;
        }
    }

    for i in 0..boundaries.len() {
        boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
    }

    Codebook { bit_width, centroids, boundaries }
}
