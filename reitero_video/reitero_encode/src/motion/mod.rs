/// Motion estimation for block-based inter prediction.
///
/// `scalar` hosts the reference implementation while `simd` can override it when
/// the `simd` feature flag is enabled.

pub const BLOCK_SIZE: usize = 16;

// MotionVector is now defined in reitero_video_common
// Re-export for convenience
pub use reitero_video_common::MotionVector;

mod backend;
pub(crate) mod luma;
use luma::LumaPlane;
pub(crate) use luma::MotionLuma;
use reitero_video_common::Yuv420Frame;

mod satd_scalar;
#[cfg(feature = "simd")]
mod satd_simd;
mod scalar;
#[cfg(feature = "simd")]
mod simd;
#[cfg(test)]
mod tests;


pub fn hex_search_yuv_sad_with_scores(
    prev: &Yuv420Frame,
    curr: &Yuv420Frame,
    width: usize,
    height: usize,
    search_range: u8,
    prev_mvs: Option<&[MotionVector]>,
    zero_mv_threshold: i64,
    predictor_threshold: i64,
    lambda: f64,
) -> (Vec<MotionVector>, Vec<i64>) {
    debug_assert_eq!(prev.width(), width);
    debug_assert_eq!(prev.height(), height);
    debug_assert_eq!(curr.width(), width);
    debug_assert_eq!(curr.height(), height);
    let prev_luma = LumaPlane::from_y_plane(prev.y_plane(), width, height);
    let curr_luma = LumaPlane::from_y_plane(curr.y_plane(), width, height);
    #[cfg(feature = "simd")]
    {
        simd::hex_search_sad_with_scores_luma(
            width,
            height,
            search_range,
            prev_mvs,
            &prev_luma,
            &curr_luma,
            zero_mv_threshold,
            predictor_threshold,
            lambda,
        )
    }
    #[cfg(not(feature = "simd"))]
    {
        scalar::hex_search_sad_with_scores_luma(
            width,
            height,
            search_range,
            prev_mvs,
            &prev_luma,
            &curr_luma,
            zero_mv_threshold,
            predictor_threshold,
            lambda,
        )
    }
}
