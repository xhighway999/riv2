//! Decoder for the Reitero video format (`.riv` files).
//!
//! # Quick start
//!
//! ```no_run
//! use reitero_decode::{Decoder, VideoReader};
//! use std::fs::File;
//! use std::io::Read;
//!
//! // Implement VideoReader for your source (e.g. a file, network stream, or in-memory buffer).
//! // Then:
//! // let mut decoder = Decoder::new(my_reader).unwrap();
//! // while decoder.has_more_frames() {
//! //     let frame = decoder.decode_frame().unwrap();
//! //     // frame.data is RGB24, frame.width × frame.height
//! // }
//! ```
//!
//! The main types are [`Decoder`] (stateful frame-by-frame decoder) and [`DecodedFrame`]
//! (RGB24 output). Implement [`VideoReader`] to provide your own input source.
//!
//! [`DecodeError`] covers all failure modes; use [`Result`] as the return type in your call sites.

mod decoder;
mod error;
mod reader;

pub use decoder::{DecodedFrame, Decoder, DecodePhaseTimings};
pub use error::{DecodeError, Result};
pub use reader::VideoReader;

pub use reitero_video_common::{FrameType, VideoHeader};
