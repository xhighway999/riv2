// ReItero video encoding library

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
#[cfg(feature = "bench-luma")]
pub use motion::bench_api;
pub use motion::{BLOCK_SIZE, MotionVector};
pub use writer::{VecWriter, VideoWriter};

// Re-export common types
pub use reitero_video_common::{FrameType, PackedFrame, RIV_MAGIC, RIV_VERSION, VideoHeader};
