/// Precomputed codebook for a given bit-width b.
/// Contains 2^b sorted centroids optimized for the Beta distribution.
#[derive(Debug, Clone)]
pub struct Codebook {
    pub bit_width: u8,
    pub centroids: Vec<f64>,
    pub boundaries: Vec<f64>,
}

/// Get precomputed codebook for high-d (Gaussian approximation).
/// These are Lloyd-Max optimal centroids for the Beta distribution on the
/// unit hypersphere as d -> infinity, pre-solved at d=512 and hardcoded.
/// Centroids live in [-1, 1] matching rotated unit-vector coordinates.
pub fn precomputed(bit_width: u8) -> Codebook {
    // NOTE: These centroids are ALREADY in the [-1, 1] range for unit-sphere coordinates.
    // Do NOT scale them by 1/sqrt(d).
    match bit_width {
        1 => {
            let c = 0.03516; // E[|X|] for Beta(d=512) ≈ sqrt(2/(pi*d))
            Codebook {
                bit_width: 1,
                centroids: vec![-c, c],
                boundaries: vec![0.0],
            }
        }
        2 => {
            let c1 = 0.01999;
            let c2 = 0.06672;
            Codebook {
                bit_width: 2,
                centroids: vec![-c2, -c1, c1, c2],
                boundaries: vec![-(c1 + c2) / 2.0, 0.0, (c1 + c2) / 2.0],
            }
        }
        3 => {
            let cs = [0.01082, 0.03338, 0.05934, 0.09501];
            let mut centroids = Vec::with_capacity(8);
            for &c in cs.iter().rev() { centroids.push(-c); }
            for &c in cs.iter() { centroids.push(c); }
            let mut boundaries = Vec::with_capacity(7);
            for i in 0..7 { boundaries.push((centroids[i] + centroids[i + 1]) / 2.0); }
            Codebook { bit_width: 3, centroids, boundaries }
        }
        4 => {
            let cs = [0.005668, 0.01713, 0.02900, 0.04161, 0.05547, 0.07143, 0.09136, 0.12066];
            let mut centroids = Vec::with_capacity(16);
            for &c in cs.iter().rev() { centroids.push(-c); }
            for &c in cs.iter() { centroids.push(c); }
            let mut boundaries = Vec::with_capacity(15);
            for i in 0..15 { boundaries.push((centroids[i] + centroids[i + 1]) / 2.0); }
            Codebook { bit_width: 4, centroids, boundaries }
        }
        _ => panic!("Precomputed codebooks only available for bit_width 1..=4, got {bit_width}"),
    }
}

/// Get codebook for a specific dimension d.
/// For d >= 256, returns precomputed codebook (no scaling needed).
/// For d < 256, runs Lloyd-Max solver for the exact Beta distribution.
pub fn for_dimension(d: usize, bit_width: u8) -> Codebook {
    if d >= 256 {
        precomputed(bit_width)
    } else {
        crate::lloyd_max::solve(d, bit_width, 2000)
    }
}
