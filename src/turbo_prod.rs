use crate::qjl::Qjl;
use crate::turbo_mse::{MseQuantized, TurboMse};

/// Result of TurboQuant_prod quantization.
/// Tied to the TurboProd instance that produced it.
#[derive(Debug, Clone)]
pub struct ProdQuantized {
    pub mse_part: MseQuantized,
    pub qjl_signs: Vec<i8>,
    pub residual_norm: f64,
}

pub struct TurboProd {
    pub turbo_mse: TurboMse,
    pub qjl: Qjl,
    pub bit_width: u8,
}

impl TurboProd {
    /// Create a new TurboProd quantizer.
    /// bit_width must be >= 2 (1 bit for MSE + 1 bit for QJL).
    /// The internal TurboMse uses bit_width - 1.
    pub fn new(d: usize, bit_width: u8, seed: Option<u64>) -> Self {
        assert!(bit_width >= 2, "TurboProd requires bit_width >= 2, got {bit_width}");
        let mse_seed = seed;
        let qjl_seed = seed.map(|s| s.wrapping_add(1_000_000));
        let turbo_mse = TurboMse::new(d, bit_width - 1, mse_seed);
        let qjl = Qjl::new(d, qjl_seed);
        TurboProd { turbo_mse, qjl, bit_width }
    }

    /// Quantize a vector x.
    pub fn quantize(&self, x: &[f64]) -> ProdQuantized {
        // Step 1: MSE quantize at (b-1) bits
        let mse_part = self.turbo_mse.quantize(x);

        // Step 2: Dequantize to get MSE reconstruction
        let x_mse = self.turbo_mse.dequantize(&mse_part);

        // Step 3: Compute residual
        let residual: Vec<f64> = x.iter().zip(x_mse.iter()).map(|(a, b)| a - b).collect();
        let residual_norm: f64 = residual.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Step 4: QJL on NORMALIZED residual
        let qjl_signs = if residual_norm == 0.0 {
            vec![1i8; x.len()] // Arbitrary valid signs; residual_norm == 0 gates these to zero in estimate
        } else {
            let r_hat: Vec<f64> = residual.iter().map(|v| v / residual_norm).collect();
            self.qjl.quantize(&r_hat)
        };

        ProdQuantized { mse_part, qjl_signs, residual_norm }
    }

    /// Estimate inner product <y, x> from quantized x and full-precision y.
    /// Formula: <y, x_mse> + residual_norm * <y, Q_qjl^{-1}(signs)>
    pub fn estimate_inner_product(&self, y: &[f64], q: &ProdQuantized) -> f64 {
        assert_eq!(
            y.len(),
            q.mse_part.indices.len(),
            "Dimension mismatch: y has {} elements but quantized vector has {}",
            y.len(),
            q.mse_part.indices.len()
        );

        let x_mse = self.turbo_mse.dequantize(&q.mse_part);
        let ip_mse: f64 = y.iter().zip(x_mse.iter()).map(|(a, b)| a * b).sum();

        let qjl_deq = self.qjl.dequantize(&q.qjl_signs);
        let ip_qjl: f64 = y.iter().zip(qjl_deq.iter()).map(|(a, b)| a * b).sum();

        ip_mse + q.residual_norm * ip_qjl
    }
}
