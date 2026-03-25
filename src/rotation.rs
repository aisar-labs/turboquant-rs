use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;

/// Generate a uniformly random orthogonal matrix in R^(d x d)
/// via QR decomposition of a Gaussian random matrix.
/// Sign correction ensures uniform Haar measure.
pub fn random_orthogonal(d: usize, rng: &mut impl Rng) -> DMatrix<f64> {
    let data: Vec<f64> = (0..d * d).map(|_| rng.sample(StandardNormal)).collect();
    let a = DMatrix::from_vec(d, d, data);
    let qr = a.qr();
    let mut q = qr.q();
    let r = qr.r();
    for j in 0..d {
        let sign = r[(j, j)].signum();
        if sign != 0.0 {
            for i in 0..d {
                q[(i, j)] *= sign;
            }
        }
    }
    q
}

/// Rotate: y = pi * x
pub fn rotate(pi: &DMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let x_vec = nalgebra::DVector::from_column_slice(x);
    let y = pi * x_vec;
    y.as_slice().to_vec()
}

/// Inverse rotate: x = pi^T * y (orthogonal matrix, so inverse = transpose)
pub fn inverse_rotate(pi: &DMatrix<f64>, y: &[f64]) -> Vec<f64> {
    let y_vec = nalgebra::DVector::from_column_slice(y);
    let x = pi.transpose() * y_vec;
    x.as_slice().to_vec()
}
