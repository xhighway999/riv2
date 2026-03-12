use crate::error::Result;

/// Trait for reading encoded video data. Used by [`crate::Decoder`]; implement for streams that support read and seek.
pub trait VideoReader {
    /// Read up to `buf.len()` bytes. Return 0 at end of stream.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;

    /// Get the current position in the input
    fn position(&mut self) -> u64;

    /// Seek to a specific position in the input
    fn seek(&mut self, pos: u64) -> Result<()>;
}
