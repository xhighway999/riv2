//! Encoder for the Reitero video format (`.riv` files).
//!
//! # Quick start
//!
//! ```no_run
//! use reitero_encode::{Encoder, EncoderConfig, Frame, VecWriter};
//!
//! let config = EncoderConfig::new(1920, 1080, 30)
//!     .with_intra_quality(90)
//!     .with_inter_quality(35);
//!
//! let writer = VecWriter::new();
//! let mut encoder = Encoder::new(config, writer).unwrap();
//!
//! // Feed RGB24 frames:
//! // let frame = Frame::new(rgb_bytes, 1920, 1080, timestamp_ms);
//! // let stats = encoder.encode_frame(frame).unwrap();
//! // let output_bytes = encoder.finish().unwrap();
//! ```
//!
//! The main types are [`Encoder`] (stateful frame-by-frame encoder), [`EncoderConfig`]
//! (builder-style configuration), and [`Frame`] (raw RGB24 input).
//! Use [`VecWriter`] to encode to an in-memory buffer or implement [`VideoWriter`] for
//! custom sinks (files, network, etc.).

mod config;
mod encoder;
mod error;
mod motion;
mod rdo;
mod writer;

#[cfg(test)]
mod mv_format_test;

pub use config::{EncoderConfig, KeyframeInterval};
pub use encoder::{EncodeFrameStats, Encoder, Frame};
pub use error::{EncodeError, Result};
pub use writer::{VecWriter, VideoWriter};

pub use reitero_video_common::{FrameType, VideoHeader};
