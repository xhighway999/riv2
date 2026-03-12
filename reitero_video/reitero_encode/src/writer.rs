use crate::error::Result;

/// Trait for writing encoded video data with seeking support
/// This allows updating the header after encoding is complete
pub trait VideoWriter {
    /// Write data to the output
    fn write(&mut self, data: &[u8]) -> Result<()>;

    /// Get the current position in the output
    fn position(&self) -> u64;

    /// Seek to a specific position in the output
    fn seek(&mut self, pos: u64) -> Result<()>;

    /// Flush any buffered data
    fn flush(&mut self) -> Result<()>;
}

/// Simple in-memory writer implementation
pub struct VecWriter {
    buffer: Vec<u8>,
    position: u64,
}

impl VecWriter {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            position: 0,
        }
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }
}

impl Default for VecWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoWriter for VecWriter {
    fn write(&mut self, data: &[u8]) -> Result<()> {
        let pos = self.position as usize;

        // Extend buffer if needed
        if pos + data.len() > self.buffer.len() {
            self.buffer.resize(pos + data.len(), 0);
        }

        // Write data
        self.buffer[pos..pos + data.len()].copy_from_slice(data);
        self.position += data.len() as u64;

        Ok(())
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn seek(&mut self, pos: u64) -> Result<()> {
        // Ensure buffer is large enough
        if pos as usize > self.buffer.len() {
            self.buffer.resize(pos as usize, 0);
        }
        self.position = pos;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // No-op for in-memory buffer
        Ok(())
    }
}
