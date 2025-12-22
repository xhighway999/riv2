use reitero_decode::Result;
use reitero_decode::VideoReader;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub struct FileVideoReader {
    file: File,
}

impl FileVideoReader {
    pub fn new(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        Ok(Self { file })
    }
}

impl VideoReader for FileVideoReader {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        Ok(self.file.read(buf)?)
    }

    fn position(&mut self) -> u64 {
        self.file.stream_position().unwrap_or(0)
    }

    fn seek(&mut self, pos: u64) -> Result<()> {
        self.file.seek(SeekFrom::Start(pos))?;
        Ok(())
    }
}
