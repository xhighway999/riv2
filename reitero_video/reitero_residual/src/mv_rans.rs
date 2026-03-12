//! Range Asymmetric Numeral Systems (RANS) encoding for motion vectors
//!
//! This module provides RANS encoding/decoding for motion vectors as a replacement for DEFLATE.
//! The **contexts** live for the entire video so probabilities carry over between inter frames.
//! The actual `RansWriter32`/`RansReader32` instances are **per frame**:
//! - Encoder: new writer + buffer per frame → compressed `Vec<u8>` for that frame
//! - Decoder: new reader per frame from that frame's compressed bytes
//! This avoids unbounded internal RANS stacks while still getting cross‑frame adaptation.
use std::{
    cell::RefCell,
    io::Write,
    rc::Rc,
};

use crate::mv_predictor::MvMode;

const MODE_CTXS: usize = 4;
const MV_MAX_CLASS: u8 = 10;
const MV_CLASS_TREE_DEPTH: usize = 4;
const MV_MAX_MAG_BITS: usize = 10;
const CANDIDATE_SIZE: usize = 16;
const CANDIDATE_IDX_DEPTH: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Subpel {
    Zero,
    PlusHalf,
    MinusHalf,
}

impl Default for Subpel {
    fn default() -> Self { Subpel::Zero }
}

impl Subpel {
    #[inline]
    pub fn to_i8(self) -> i8 {
        match self {
            Subpel::Zero => 0,
            Subpel::PlusHalf => 1,
            Subpel::MinusHalf => -1,
        }
    }

    #[inline]
    pub fn from_flag_bits(bits: u8) -> Self {
        match bits & 0x03 {
            1 => Subpel::PlusHalf,
            2 => Subpel::MinusHalf,
            _ => Subpel::Zero,
        }
    }

    #[inline]
    pub fn to_flag_bits(self) -> u8 {
        match self {
            Subpel::Zero => 0,
            Subpel::PlusHalf => 1,
            Subpel::MinusHalf => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvCodedBlock {
    pub mode: MvMode,
    /// For `MvMode::New`, selects which integer predictor base to subtract.
    /// Encoding: 0=nearest, 1=near, 2=top-right, 3=top-left, 4=temporal.
    pub new_base: u8,
    pub delta_x: i8,
    pub delta_y: i8,
    /// Fractional half-pixel offset for X axis (-0.5, 0.0, +0.5)
    pub subpel_x: Subpel,
    /// Fractional half-pixel offset for Y axis (-0.5, 0.0, +0.5)
    pub subpel_y: Subpel,
    /// Whether this block is skipped (authoritative post-residual decision)
    pub skip: bool,
}

pub fn mv_class_from_magnitude(mag: u16) -> u8 {
    if mag == 0 {
        return 0;
    }
    let msb = 15 - mag.leading_zeros() as u8;
    (msb + 1).min(MV_MAX_CLASS)
}

fn mv_class_base(class: u8) -> u16 {
    if class == 0 { 0 } else { 1u16 << (class - 1) }
}

fn mv_class_bits(class: u8) -> u8 {
    if class <= 1 { 0 } else { class - 1 }
}

struct SharedBuffer(Rc<RefCell<Vec<u8>>>);

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.borrow_mut().flush()
    }
}

/// RANS encoder for motion vectors
///
/// This encoder owns **only the probability contexts**.
/// For each frame, `encode_frame_and_get_data()` creates a fresh `RansWriter32`
/// and buffer, encodes the frame with the shared contexts, then returns the
/// compressed bytes. Contexts are updated and reused across frames.
///
/// In addition to the bitwise contexts, we keep a shortcut context for blocks
/// whose vector is predicted perfectly (`ddx == ddy == 0`). That flag lets us
/// skip emitting both zigzag payloads entirely when both axes match the
/// predictor + bias exactly.
pub mod encoder;
pub mod decoder;

pub use encoder::MvRansEncoder;
pub use decoder::MvRansDecoder;

/// RANS decoder for motion vectors
///
/// This decoder consumes compressed motion vector bytes and decodes them frame by frame.
/// Probabilities are maintained across frames for better compression.
/// Call `consume()` to add compressed bytes, then `decode_frame()` for each frame.
// Decoder implementation moved to submodule

#[cfg(test)]
mod tests;
