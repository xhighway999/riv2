use crate::error::Result;

/// Trait for reading encoded video data with seeking support
pub trait VideoReader {
    /// Read data from the input
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;

    /// Get the current position in the input
    fn position(&mut self) -> u64;

    /// Seek to a specific position in the input
    fn seek(&mut self, pos: u64) -> Result<()>;
}
