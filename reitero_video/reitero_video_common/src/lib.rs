//! Common types and utilities shared across the Reitero video codec crates.
//!
//! This crate is an implementation detail of `reitero_encode` and `reitero_decode`.
//! Most applications should use those higher-level crates directly.
//!
//! # Contents
//!
//! - [`VideoHeader`] / [`PackedFrame`] — binary format structures
//! - [`FrameType`] — I-frame vs P-frame discriminant
//! - [`Yuv420Frame`] — planar YUV420 frame with RGB24 conversion
//! - [`MotionVector`] — per-block motion vector (integer + half-pixel)
//! - [`build_predicted`] — motion-compensated prediction helper
//! - [`RIV_MAGIC`] / [`RIV_VERSION`] — format version constants

mod deblock;
mod format;
mod motion;
mod motion_vector;
mod fast_motion;
mod yuv;
mod instrumentation; // <- new instrumentation module
pub mod rans;

pub use deblock::{deblock_level_from_quant_step, deblock_yuv420};
pub use format::PackedFrameData;
pub use format::{FrameType, PackedFrame, RIV_MAGIC, RIV_VERSION, VideoHeader};
#[doc(hidden)]
pub use format::{bytes_to_i16_vec, i16_vec_to_bytes};
pub use motion_vector::MotionVector;
#[doc(hidden)]
pub use motion::{
	reference_build_predicted,
	decode_halfpel_i8,
	sample_rgb_halfpel,
};
pub use fast_motion::build_predicted;
pub use yuv::{Yuv420Frame, YuvConvertError};
#[doc(hidden)]
pub use instrumentation::Instrument;
