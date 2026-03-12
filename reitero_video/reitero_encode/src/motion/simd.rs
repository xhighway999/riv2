//! SIMD motion estimation backend placeholder.
//!
//! Until optimized kernels land we simply forward to the scalar implementation
//! so benchmarks/tests can still exercise a SIMD entry point.

use super::{LumaPlane, MotionVector};

pub fn hex_search_sad_with_scores_luma(
    width: usize,
    height: usize,
    search_range: u8,
    prev_mvs: Option<&[MotionVector]>,
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    zero_mv_threshold: i64,
    predictor_threshold: i64,
    lambda: f64,
) -> (Vec<MotionVector>, Vec<i64>) {
    super::scalar::hex_search_sad_with_scores_luma(
        width,
        height,
        search_range,
        prev_mvs,
        prev_luma,
        curr_luma,
        zero_mv_threshold,
        predictor_threshold,
        lambda,
    )
}
