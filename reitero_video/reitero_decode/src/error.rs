use thiserror::Error;

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    #[error("Invalid frame data: {0}")]
    InvalidFrame(String),

    #[error("End of stream")]
    EndOfStream,
}

pub type Result<T> = std::result::Result<T, DecodeError>;
