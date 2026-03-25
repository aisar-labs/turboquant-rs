#[derive(Debug, Clone)]
pub struct UniformQuantized {
    pub indices: Vec<u8>,
    pub min: f64,
    pub max: f64,
    pub bit_width: u8,
}

pub fn quantize(x: &[f64], bit_width: u8) -> UniformQuantized {
    let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let num_bins = 1usize << bit_width;
    let range = max - min;
    let indices = if range == 0.0 {
        vec![0u8; x.len()]
    } else {
        x.iter()
            .map(|&v| {
                let normalized = (v - min) / range;
                let bin = (normalized * (num_bins as f64 - 1.0)).round() as u8;
                bin.min((num_bins - 1) as u8)
            })
            .collect()
    };
    UniformQuantized { indices, min, max, bit_width }
}

pub fn dequantize(q: &UniformQuantized) -> Vec<f64> {
    let num_bins = 1usize << q.bit_width;
    let range = q.max - q.min;
    if range == 0.0 {
        return vec![q.min; q.indices.len()];
    }
    q.indices.iter()
        .map(|&idx| q.min + (idx as f64 / (num_bins as f64 - 1.0)) * range)
        .collect()
}
