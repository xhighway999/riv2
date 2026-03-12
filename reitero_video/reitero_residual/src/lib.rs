//! Residual coding for the Reitero video codec.
//!
//! Handles the low-level encode/decode pipeline between motion-compensated prediction
//! and the final bitstream: DCT transform, quantization, and RANS entropy coding.
//!
//! # Main types
//!
//! - [`ResidualEncoder`] / [`ResidualDecoder`] — encode/decode intra and inter residuals
//! - [`MvRansEncoder`] / [`MvRansDecoder`] — RANS entropy coding for motion vectors
//! - [`MvPredictors`] / [`MvNeighborSet`] — motion vector prediction helpers
//! - [`quant_step_from_quality`] — map a 1–100 quality value to a DCT quantization step
//!
//! This crate is primarily used by `reitero_encode` and `reitero_decode`; most
//! applications should use those higher-level crates instead.

mod mv_layout;
mod mv_predictor;
mod mv_rans;
mod rans;
mod residual;

pub use mv_layout::{mv_interleaved_to_planar, mv_planar_to_interleaved};
pub use mv_predictor::{
    MvMode, MvNeighborSet, MvPredictors, derive_mv_predictors, derive_mv_predictors_with_stats,
    gather_mv_neighbor_set, mv_mode_context,
};
pub use mv_rans::{MvCodedBlock, MvRansDecoder, MvRansEncoder, Subpel, mv_class_from_magnitude};
pub use residual::{
    InterResidualDecodeParams, InterResidualEncodeParams, InterResidualEncodeResult,
    IntraResidualEncodeResult, ResidualDecoder, ResidualEncoder, ResidualError,
    RleCompressionStats, drain_residual_phase_counters,
    quant_step_from_quality,
};
