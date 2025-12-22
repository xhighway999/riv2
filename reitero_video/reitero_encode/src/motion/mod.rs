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

pub use reitero_video_common::build_predicted;

#[cfg(feature = "bench-luma")]
pub mod bench_api {
    use std::hint::black_box;
    use std::sync::Arc;

    use super::LumaPlane;

    const SAD_POSITIONS: &[(i32, i32)] = &[
        (0, 0),
        (
            (super::BLOCK_SIZE / 2) as i32,
            (super::BLOCK_SIZE / 3) as i32,
        ),
    ];

    const SAD_OFFSETS: &[(i32, i32)] = &[(0, 0), (1, 0), (2, 2), (3, 1), (4, 3)];
    const SATD_POSITIONS: &[(i32, i32)] = &[
        (0, 0),
        (
            (super::BLOCK_SIZE / 4) as i32,
            (super::BLOCK_SIZE / 5) as i32,
        ),
    ];
    const SATD_OFFSETS: &[(i32, i32)] = &[(-2, -1), (0, 0), (1, 2), (3, -2)];

    #[derive(Clone)]
    pub struct PreparedPlanes {
        prev: Arc<LumaPlane>,
        curr: Arc<LumaPlane>,
        width: usize,
        height: usize,
    }

    impl PreparedPlanes {
        pub fn from_rgb(prev: &[u8], curr: &[u8], width: usize, height: usize) -> Self {
            Self {
                prev: Arc::new(LumaPlane::from_rgb(prev, width, height)),
                curr: Arc::new(LumaPlane::from_rgb(curr, width, height)),
                width,
                height,
            }
        }

        fn prev(&self) -> &LumaPlane {
            self.prev.as_ref()
        }

        fn curr(&self) -> &LumaPlane {
            self.curr.as_ref()
        }
    }

    /// Run the scalar RGB→luma conversion and feed the result into `black_box`
    /// so benchmarks can measure the work without leaking internal types.
    pub fn run_scalar_rgb_to_luma_plane(rgb: &[u8], width: usize, height: usize) {
        let plane = super::satd_scalar::rgb_to_luma_plane(rgb, width, height);
        black_box(plane);
    }

    /// Same helper for the SIMD feature path.
    #[cfg(feature = "simd")]
    pub fn run_simd_rgb_to_luma_plane(rgb: &[u8], width: usize, height: usize) {
        let plane = super::satd_simd::rgb_to_luma_plane(rgb, width, height);
        black_box(plane);
    }

    pub fn run_scalar_sample_luma_halfpel(rgb: &[u8], width: usize, height: usize) {
        let plane = super::satd_simd::rgb_to_luma_plane(rgb, width, height);
        super::satd_scalar::bench_sample_luma_halfpel(&plane, width, height);
    }

    #[cfg(feature = "simd")]
    pub fn run_simd_sample_luma_halfpel(rgb: &[u8], width: usize, height: usize) {
        let plane = super::satd_simd::rgb_to_luma_plane(rgb, width, height);
        let plane = super::LumaPlane::from(plane);
        super::satd_simd::bench_sample_luma_halfpel(&plane, width, height);
    }

    pub fn run_scalar_sad_halfpel(planes: &PreparedPlanes) {
        let acc = super::satd_scalar::bench_sad_block_halfpel(
            planes.prev(),
            planes.curr(),
            planes.width,
            planes.height,
            SAD_POSITIONS,
            SAD_OFFSETS,
        );
        black_box(acc);
    }

    #[cfg(feature = "simd")]
    pub fn run_simd_sad_halfpel(planes: &PreparedPlanes) {
        let acc = super::satd_simd::bench_sad_block_halfpel(
            planes.prev(),
            planes.curr(),
            planes.width,
            planes.height,
            SAD_POSITIONS,
            SAD_OFFSETS,
        );
        black_box(acc);
    }

    pub fn run_scalar_satd_block_int(planes: &PreparedPlanes) {
        let mut acc = 0i64;
        for &(x0, y0) in SATD_POSITIONS {
            for &(dx, dy) in SATD_OFFSETS {
                acc ^= super::satd_scalar::satd_block_int(
                    planes.prev(),
                    planes.curr(),
                    planes.width,
                    planes.height,
                    x0,
                    y0,
                    dx,
                    dy,
                );
            }
        }
        black_box(acc);
    }

    pub fn run_scalar_sad_block_int(planes: &PreparedPlanes) {
        let mut acc = 0i64;
        for &(x0, y0) in SATD_POSITIONS {
            for &(dx, dy) in SATD_OFFSETS {
                acc ^= super::satd_scalar::sad_block_int(
                    planes.prev(),
                    planes.curr(),
                    planes.width,
                    planes.height,
                    x0,
                    y0,
                    dx,
                    dy,
                );
            }
        }
        black_box(acc);
    }

    #[cfg(feature = "simd")]
    pub fn run_simd_satd_block_int(planes: &PreparedPlanes) {
        let mut acc = 0i64;
        for &(x0, y0) in SATD_POSITIONS {
            for &(dx, dy) in SATD_OFFSETS {
                acc ^= super::satd_simd::satd_block_int(
                    planes.prev(),
                    planes.curr(),
                    planes.width,
                    planes.height,
                    x0,
                    y0,
                    dx,
                    dy,
                );
            }
        }
        black_box(acc);
    }

    #[cfg(feature = "simd")]
    pub fn run_simd_sad_block_int(planes: &PreparedPlanes) {
        let mut acc = 0i64;
        for &(x0, y0) in SATD_POSITIONS {
            for &(dx, dy) in SATD_OFFSETS {
                acc ^= super::satd_simd::sad_block_int(
                    planes.prev(),
                    planes.curr(),
                    planes.width,
                    planes.height,
                    x0,
                    y0,
                    dx,
                    dy,
                );
            }
        }
        black_box(acc);
    }

    pub fn run_scalar_hex_search(planes: &PreparedPlanes, search_range: u8) {
        let result = super::scalar::hex_search_sad_with_scores_luma(
            planes.width,
            planes.height,
            search_range,
            None,
            planes.prev(),
            planes.curr(),
            0,
            0,
            0.0,
        );
        black_box(result);
    }

    #[cfg(feature = "simd")]
    pub fn run_simd_hex_search(planes: &PreparedPlanes, search_range: u8) {
        let result = super::simd::hex_search_sad_with_scores_luma(
            planes.width,
            planes.height,
            search_range,
            None,
            planes.prev(),
            planes.curr(),
            0,
            0,
            0.0,
        );
        black_box(result);
    }
}

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
