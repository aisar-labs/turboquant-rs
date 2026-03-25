use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::codebook::{self, Codebook};
use crate::rotation;

#[derive(Debug, Clone)]
pub struct MseQuantized {
    pub indices: Vec<u8>,
    pub bit_width: u8,
    pub norm: f64,
}

pub struct TurboMse {
    pub rotation: DMatrix<f64>,
    pub codebook: Codebook,
}

impl TurboMse {
    pub fn new(d: usize, bit_width: u8, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha20Rng::seed_from_u64(s),
            None => ChaCha20Rng::from_entropy(),
        };
        let rotation = rotation::random_orthogonal(d, &mut rng);
        let codebook = codebook::for_dimension(d, bit_width);
        TurboMse { rotation, codebook }
    }

    pub fn quantize(&self, x: &[f64]) -> MseQuantized {
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            return MseQuantized {
                indices: vec![0; x.len()],
                bit_width: self.codebook.bit_width,
                norm: 0.0,
            };
        }
        let x_hat: Vec<f64> = x.iter().map(|v| v / norm).collect();
        let y = rotation::rotate(&self.rotation, &x_hat);
        let indices: Vec<u8> = y
            .iter()
            .map(|&yj| {
                let mut best_idx = 0u8;
                let mut best_dist = (yj - self.codebook.centroids[0]).abs();
                for (k, &ck) in self.codebook.centroids.iter().enumerate().skip(1) {
                    let dist = (yj - ck).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = k as u8;
                    }
                }
                best_idx
            })
            .collect();
        MseQuantized { indices, bit_width: self.codebook.bit_width, norm }
    }

    pub fn dequantize(&self, q: &MseQuantized) -> Vec<f64> {
        if q.norm == 0.0 {
            return vec![0.0; q.indices.len()];
        }
        let y_hat: Vec<f64> = q
            .indices
            .iter()
            .map(|&idx| self.codebook.centroids[idx as usize])
            .collect();
        let x_hat = rotation::inverse_rotate(&self.rotation, &y_hat);
        x_hat.iter().map(|v| v * q.norm).collect()
    }
}
