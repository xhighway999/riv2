use thiserror::Error;

#[derive(Error, Debug)]
pub enum EncodeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Encoding failed: {0}")]
    EncodingFailed(String),

    #[error("Unsupported codec: {0}")]
    UnsupportedCodec(String),

    #[error("Frame encoding error: {0}")]
    FrameError(String),
}

pub type Result<T> = std::result::Result<T, EncodeError>;
