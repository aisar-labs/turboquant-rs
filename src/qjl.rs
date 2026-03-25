use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::StandardNormal;
use rand::Rng;

/// Quantized Johnson-Lindenstrauss transform.
/// Reduces vectors to sign bits while preserving inner products in expectation.
pub struct Qjl {
    pub projection: DMatrix<f64>,
    pub d: usize,
}

impl Qjl {
    /// Create a new QJL transform for dimension d.
    pub fn new(d: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha20Rng::seed_from_u64(s),
            None => ChaCha20Rng::from_entropy(),
        };
        let data: Vec<f64> = (0..d * d).map(|_| rng.sample(StandardNormal)).collect();
        let projection = DMatrix::from_vec(d, d, data);
        Qjl { projection, d }
    }

    /// Quantize: signs = sign(S * r). Zero values map to +1.
    pub fn quantize(&self, r: &[f64]) -> Vec<i8> {
        let r_vec = nalgebra::DVector::from_column_slice(r);
        let z = &self.projection * r_vec;
        z.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect()
    }

    /// Dequantize: Q_qjl^{-1}(z) = (sqrt(pi/2) / d) * S^T * z
    pub fn dequantize(&self, signs: &[i8]) -> Vec<f64> {
        let scale = (std::f64::consts::PI / 2.0).sqrt() / self.d as f64;
        let z = nalgebra::DVector::from_iterator(self.d, signs.iter().map(|&s| s as f64));
        let result = scale * (self.projection.transpose() * z);
        result.as_slice().to_vec()
    }
}
