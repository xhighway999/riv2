//! Backend selector for SATD computations.
//!
//! Today we always fall back to the scalar implementation, but the module layout
//! allows us to plug in SIMD-accelerated code paths once they are ready.

#[cfg(feature = "simd")]
pub(super) use super::satd_simd::{
    sad_block_halfpel_limit_luma, sad_block_int_limit,
};

#[cfg(not(feature = "simd"))]
pub(super) use super::satd_scalar::{
    sad_block_halfpel_limit_luma, sad_block_int_limit,
};

