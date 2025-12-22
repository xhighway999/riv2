// ReItero video decoding library

mod decoder;
mod error;
mod reader;

pub use decoder::{DecodedFrame, Decoder};
pub use error::{DecodeError, Result};
pub use reader::VideoReader;

// Re-export common types
pub use reitero_video_common::{FrameType, PackedFrame, RIV_MAGIC, RIV_VERSION, VideoHeader};
