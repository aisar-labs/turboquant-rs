use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::StandardNormal;
use rand::Rng;
use std::cell::Cell;

/// Quantized Johnson-Lindenstrauss transform.
///
/// Projects a d-dimensional vector r to d sign bits and provides an unbiased
/// dequantizer that recovers the original vector in expectation.
///
/// ## Algorithm
///
/// **Quantize**: z_i = sign(〈s_i, r〉), where the s_i are the rows of a random
/// Gaussian matrix S drawn from N(0, I_d).  Ties map to +1.
///
/// **Dequantize**: uses the formula
///
/// ```text
/// x_hat = (‖r‖ · sqrt(π/2) / d) · S^T · z
/// ```
///
/// For any row s_i ~ N(0, I_d):
///   E[sign(〈s_i, r〉) · s_i] = sqrt(2/π) · r / ‖r‖
///
/// Summing d such rows and multiplying by the scale recovers r in expectation:
///   E[x_hat] = r.
///
/// The per-trial variance of the inner product estimator 〈y, x_hat〉 is
///   Var = (π/2) · ‖r‖² · ‖y‖² / d,
/// which decreases with dimension for vectors of bounded ‖y‖.
pub struct Qjl {
    pub projection: DMatrix<f64>,
    pub d: usize,
    /// Cached Euclidean norm of the last vector passed to `quantize`.
    norm: Cell<f64>,
}

impl Qjl {
    /// Create a new QJL transform of dimension `d`.
    ///
    /// `seed = Some(s)` produces a deterministic projection matrix; `None`
    /// draws from system entropy.
    pub fn new(d: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha20Rng::seed_from_u64(s),
            None => ChaCha20Rng::from_entropy(),
        };
        let data: Vec<f64> = (0..d * d).map(|_| rng.sample(StandardNormal)).collect();
        let projection = DMatrix::from_vec(d, d, data);
        Qjl { projection, d, norm: Cell::new(1.0) }
    }

    /// Quantize: z = sign(S · r), with ties (exactly 0) mapped to +1.
    ///
    /// Caches ‖r‖ so that `dequantize` can reconstruct an unbiased estimate
    /// of r (not just its direction).
    pub fn quantize(&self, r: &[f64]) -> Vec<i8> {
        let rn: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        self.norm.set(if rn == 0.0 { 1.0 } else { rn });

        let rv = nalgebra::DVector::from_column_slice(r);
        let z = &self.projection * rv;
        z.iter().map(|&v| if v >= 0.0 { 1i8 } else { -1i8 }).collect()
    }

    /// Dequantize: x_hat = (‖r‖ · sqrt(π/2) / d) · S^T · z
    ///
    /// Uses the ‖r‖ cached by the preceding `quantize` call.  The scaling
    /// factor inverts the projection bias so that E[〈y, x_hat〉] = 〈y, r〉.
    pub fn dequantize(&self, signs: &[i8]) -> Vec<f64> {
        let scale = self.norm.get() * (std::f64::consts::PI / 2.0).sqrt() / self.d as f64;
        let z = nalgebra::DVector::from_iterator(
            self.d,
            signs.iter().map(|&s| s as f64),
        );
        let result = scale * (self.projection.transpose() * z);
        result.as_slice().to_vec()
    }
}
