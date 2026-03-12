use anyhow::{Context, Result};
use reitero_encode::{EncodeError, VideoWriter};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

pub struct FileWriter {
    writer: BufWriter<File>,
    position: u64,
}

impl FileWriter {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::create(path.as_ref()).context("Failed to create output file")?;
        Ok(Self {
            writer: BufWriter::new(file),
            position: 0,
        })
    }
}

impl VideoWriter for FileWriter {
    fn write(&mut self, data: &[u8]) -> reitero_encode::Result<()> {
        self.writer
            .write_all(data)
            .map_err(|e| EncodeError::Io(e))?;
        self.position += data.len() as u64;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn seek(&mut self, pos: u64) -> reitero_encode::Result<()> {
        self.writer
            .seek(std::io::SeekFrom::Start(pos))
            .map_err(|e| EncodeError::Io(e))?;
        self.position = pos;
        Ok(())
    }

    fn flush(&mut self) -> reitero_encode::Result<()> {
        self.writer.flush().map_err(|e| EncodeError::Io(e))
    }
}
