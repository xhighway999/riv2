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
    io::{Cursor, Write},
    rc::Rc,
};

use cabac::{
    CabacReader, CabacWriter,
    rans32::{RansReader32, RansWriter32},
    vp8::VP8Context,
};

use crate::mv_predictor::{MvMode, mv_mode_context};

const MODE_CTXS: usize = 4;
const MV_MAX_CLASS: u8 = 10;
const MV_CLASS_TREE_DEPTH: usize = 4;
const MV_MAX_MAG_BITS: usize = 10;
const CANDIDATE_SIZE: usize = 16;
const CANDIDATE_IDX_DEPTH: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvCodedBlock {
    pub mode: MvMode,
    /// For `MvMode::New`, selects which integer predictor base to subtract.
    /// Encoding: 0=nearest, 1=near, 2=top-right, 3=top-left, 4=temporal.
    pub new_base: u8,
    pub delta_x: i8,
    pub delta_y: i8,
    pub flags: u8,
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

fn encode_class_symbol(
    writer: &mut RansWriter32<SharedBuffer>,
    ctx: &mut [VP8Context; MV_CLASS_TREE_DEPTH],
    class: u8,
) {
    let mut lo = 0i32;
    let mut hi = MV_MAX_CLASS as i32;
    let mut depth = 0usize;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let go_high = class as i32 > mid;
        writer
            .put(go_high, &mut ctx[depth])
            .expect("RANS writer failure on class tree");
        if go_high {
            lo = mid + 1;
        } else {
            hi = mid;
        }
        depth += 1;
    }
}

fn decode_class_symbol(
    reader: &mut RansReader32<Cursor<Vec<u8>>>,
    ctx: &mut [VP8Context; MV_CLASS_TREE_DEPTH],
) -> u8 {
    let mut lo = 0i32;
    let mut hi = MV_MAX_CLASS as i32;
    let mut depth = 0usize;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let go_high = reader
            .get(&mut ctx[depth])
            .expect("RANS reader failure on class tree");
        if go_high {
            lo = mid + 1;
        } else {
            hi = mid;
        }
        depth += 1;
    }
    lo as u8
}

fn encode_candidate_index(
    writer: &mut RansWriter32<SharedBuffer>,
    ctx: &mut [VP8Context; CANDIDATE_IDX_DEPTH],
    index: u8,
) {
    let mut lo = 0i32;
    let mut hi = CANDIDATE_SIZE as i32;
    let mut depth = 0usize;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        let go_high = (index as i32) >= mid;
        writer
            .put(go_high, &mut ctx[depth])
            .expect("RANS writer failure on candidate tree");
        if go_high {
            lo = mid;
        } else {
            hi = mid;
        }
        depth += 1;
    }
}

fn decode_candidate_index(
    reader: &mut RansReader32<Cursor<Vec<u8>>>,
    ctx: &mut [VP8Context; CANDIDATE_IDX_DEPTH],
) -> u8 {
    let mut lo = 0i32;
    let mut hi = CANDIDATE_SIZE as i32;
    let mut depth = 0usize;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        let go_high = reader
            .get(&mut ctx[depth])
            .expect("RANS reader failure on candidate tree");
        if go_high {
            lo = mid;
        } else {
            hi = mid;
        }
        depth += 1;
    }
    lo as u8
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

/// Zigzag encode a signed i8 value to an unsigned u8
///
/// This transformation maps small deltas (0, -1, 1) to small unsigned symbols,
/// centering the entropy around zero for better RANS compression.
///
/// Formula: `(n << 1) ^ (n >> 7)`
/// - For n >= 0: maps to even numbers (0→0, 1→2, 2→4, ...)
/// - For n < 0: maps to odd numbers (-1→1, -2→3, -3→5, ...)
///
/// # Arguments
/// * `n` - Signed i8 value (-128 to 127)
///
/// # Returns
/// Unsigned u8 value (0 to 255)
#[inline]
pub fn zigzag_encode_i8(n: i8) -> u8 {
    // For i8: n >> 7 is 0 for n >= 0, and -1 (0xFF) for n < 0
    ((n as u8) << 1) ^ ((n >> 7) as u8)
}

/// Zigzag decode an unsigned u8 value back to a signed i8
///
/// This is the inverse of `zigzag_encode_i8`.
///
/// Formula:
/// - If n is even: n >> 1 (positive value)
/// - If n is odd: -((n + 1) >> 1) (negative value)
///
/// # Arguments
/// * `n` - Unsigned u8 value (0 to 255)
///
/// # Returns
/// Signed i8 value (-128 to 127)
#[inline]
pub fn zigzag_decode_i8(n: u8) -> i8 {
    if n & 1 == 0 {
        // Even: positive value
        (n >> 1) as i8
    } else {
        // Odd: negative value
        // Use i16 for intermediate calculation to avoid overflow when negating
        let magnitude = ((n as u16 + 1) >> 1) as i16;
        (-magnitude) as i8
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
pub struct MvRansEncoder {
    skip_ctx: VP8Context,
    mode_is_new_ctx: [VP8Context; MODE_CTXS],
    mode_is_nonzero_ctx: [VP8Context; MODE_CTXS],
    mode_is_near_ctx: [VP8Context; MODE_CTXS],
    mode_is_temporal_ctx: [VP8Context; MODE_CTXS],
    mode_is_top_right_ctx: [VP8Context; MODE_CTXS],
    mode_is_top_left_ctx: [VP8Context; MODE_CTXS],
    // NEW base selector decision tree (5-way): nearest/near/top-right/top-left/temporal.
    new_base_is_nearest_ctx: [VP8Context; MODE_CTXS],
    new_base_is_near_ctx: [VP8Context; MODE_CTXS],
    new_base_is_top_right_ctx: [VP8Context; MODE_CTXS],
    new_base_is_top_left_ctx: [VP8Context; MODE_CTXS],
    class_ctx_x: [VP8Context; MV_CLASS_TREE_DEPTH],
    class_ctx_y: [VP8Context; MV_CLASS_TREE_DEPTH],
    sign_ctx_x: VP8Context,
    sign_ctx_y: VP8Context,
    mag_bit_ctx_x: [VP8Context; MV_MAX_MAG_BITS],
    mag_bit_ctx_y: [VP8Context; MV_MAX_MAG_BITS],
    sub_x_has_ctx: VP8Context,
    sub_x_sign_ctx: VP8Context,
    sub_y_has_ctx: VP8Context,
    sub_y_sign_ctx: VP8Context,
    candidate_hit_ctx: [VP8Context; 3],
    candidate_idx_ctx: [VP8Context; CANDIDATE_IDX_DEPTH],
    candidates: Vec<(i8, i8, u8)>,
    candidate_counts: std::collections::HashMap<(i8, i8, u8), u32>,
    total_new_history: u32,
}

impl MvRansEncoder {
    /// Create a new RANS encoder for motion vectors
    ///
    /// This encoder will be used for the entire video, maintaining probability
    /// contexts across all frames.
    ///
    /// # Returns
    /// A new `MvRansEncoder` ready for encoding operations
    pub fn new() -> Self {
        Self {
            skip_ctx: VP8Context::default(),
            mode_is_new_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_nonzero_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_near_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_temporal_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_top_right_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_top_left_ctx: [VP8Context::default(); MODE_CTXS],
            new_base_is_nearest_ctx: [VP8Context::default(); MODE_CTXS],
            new_base_is_near_ctx: [VP8Context::default(); MODE_CTXS],
            new_base_is_top_right_ctx: [VP8Context::default(); MODE_CTXS],
            new_base_is_top_left_ctx: [VP8Context::default(); MODE_CTXS],
            class_ctx_x: [VP8Context::default(); MV_CLASS_TREE_DEPTH],
            class_ctx_y: [VP8Context::default(); MV_CLASS_TREE_DEPTH],
            sign_ctx_x: VP8Context::default(),
            sign_ctx_y: VP8Context::default(),
            mag_bit_ctx_x: [VP8Context::default(); MV_MAX_MAG_BITS],
            mag_bit_ctx_y: [VP8Context::default(); MV_MAX_MAG_BITS],
            sub_x_has_ctx: VP8Context::default(),
            sub_x_sign_ctx: VP8Context::default(),
            sub_y_has_ctx: VP8Context::default(),
            sub_y_sign_ctx: VP8Context::default(),
            candidate_hit_ctx: [VP8Context::default(); 3],
            candidate_idx_ctx: [VP8Context::default(); CANDIDATE_IDX_DEPTH],
            candidates: Vec::new(),
            candidate_counts: std::collections::HashMap::new(),
            total_new_history: 0,
        }
    }

    /// Reset all probability contexts back to their initial state.
    ///
    /// Callers should invoke this whenever an intra (key) frame is emitted so
    /// that subsequent inter frames form a seekable restart point.
    pub fn reset_contexts(&mut self) {
        self.skip_ctx = VP8Context::default();
        self.mode_is_new_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_nonzero_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_near_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_temporal_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_top_right_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_top_left_ctx = [VP8Context::default(); MODE_CTXS];
        self.new_base_is_nearest_ctx = [VP8Context::default(); MODE_CTXS];
        self.new_base_is_near_ctx = [VP8Context::default(); MODE_CTXS];
        self.new_base_is_top_right_ctx = [VP8Context::default(); MODE_CTXS];
        self.new_base_is_top_left_ctx = [VP8Context::default(); MODE_CTXS];
        self.class_ctx_x = [VP8Context::default(); MV_CLASS_TREE_DEPTH];
        self.class_ctx_y = [VP8Context::default(); MV_CLASS_TREE_DEPTH];
        self.sign_ctx_x = VP8Context::default();
        self.sign_ctx_y = VP8Context::default();
        self.mag_bit_ctx_x = [VP8Context::default(); MV_MAX_MAG_BITS];
        self.mag_bit_ctx_y = [VP8Context::default(); MV_MAX_MAG_BITS];
        self.sub_x_has_ctx = VP8Context::default();
        self.sub_x_sign_ctx = VP8Context::default();
        self.sub_y_has_ctx = VP8Context::default();
        self.sub_y_sign_ctx = VP8Context::default();
        self.candidate_hit_ctx = [VP8Context::default(); 3];
        self.candidate_idx_ctx = [VP8Context::default(); CANDIDATE_IDX_DEPTH];
        self.candidates.clear();
        self.candidate_counts.clear();
        self.total_new_history = 0;
    }

    /// Core encoding routine: encode motion vectors for a single frame into the
    /// provided RANS writer, updating this encoder's contexts.
    ///
    /// Motion vectors are provided as interleaved `[ddx, ddy, flags]` per block.
    /// Encodes in forward raster order (top-left to bottom-right).
    fn encode_frame_to_writer(
        &mut self,
        writer: &mut RansWriter32<SharedBuffer>,
        blocks: &[MvCodedBlock],
        blocks_w: usize,
        blocks_h: usize,
    ) {
        assert_eq!(blocks.len(), blocks_w * blocks_h);
        let mut modes_so_far: Vec<MvMode> = Vec::with_capacity(blocks.len());
        let mut hits_so_far: Vec<bool> = Vec::with_capacity(blocks.len());

        for (i, block) in blocks.iter().enumerate() {
            let bx = i % blocks_w;
            let by = i / blocks_w;
            let skip = (block.flags & 0x40) != 0;
            writer
                .put(skip, &mut self.skip_ctx)
                .expect("RANS writer failure on skip flag");

            let ctx = mv_mode_context(&modes_so_far, blocks_w, bx, by);
            self.encode_mode(writer, block.mode, ctx);

            if block.mode == MvMode::New {
                // Encode NEW base selector using a simple entropy-coded decision tree.
                // 0=nearest, 1=near, 2=top-right, 3=top-left, 4=temporal.
                let b = block.new_base;
                let is_nearest = b == 0;
                writer
                    .put(is_nearest, &mut self.new_base_is_nearest_ctx[ctx])
                    .expect("RANS writer failure on NEW base split (nearest)");
                if !is_nearest {
                    let is_near = b == 1;
                    writer
                        .put(is_near, &mut self.new_base_is_near_ctx[ctx])
                        .expect("RANS writer failure on NEW base split (near)");
                    if !is_near {
                        let is_top_right = b == 2;
                        writer
                            .put(is_top_right, &mut self.new_base_is_top_right_ctx[ctx])
                            .expect("RANS writer failure on NEW base split (top-right)");
                        if !is_top_right {
                            let is_top_left = b == 3;
                            writer
                                .put(is_top_left, &mut self.new_base_is_top_left_ctx[ctx])
                                .expect("RANS writer failure on NEW base split (top-left)");
                        }
                    }
                }

                let mut encoded_fractional_via_candidate = false;
                if !self.candidates.is_empty() {
                    let frac_flags = block.flags & 0x0F;
                    let candidate_idx = self
                        .candidates
                        .iter()
                        .position(|&p| p == (block.delta_x, block.delta_y, frac_flags));
                    let hit = candidate_idx.is_some();

                    let left_hit = if bx > 0 { hits_so_far[i - 1] } else { false };
                    let top_hit = if by > 0 {
                        hits_so_far[i - blocks_w]
                    } else {
                        false
                    };
                    let hit_ctx_idx = (left_hit as usize) + (top_hit as usize);

                    writer
                        .put(hit, &mut self.candidate_hit_ctx[hit_ctx_idx])
                        .expect("RANS writer failure on candidate hit");

                    if let Some(idx) = candidate_idx {
                        encode_candidate_index(writer, &mut self.candidate_idx_ctx, idx as u8);
                        encoded_fractional_via_candidate = true;
                    } else {
                        Self::encode_component(
                            writer,
                            block.delta_x,
                            &mut self.class_ctx_x,
                            &mut self.sign_ctx_x,
                            &mut self.mag_bit_ctx_x,
                        );
                        Self::encode_component(
                            writer,
                            block.delta_y,
                            &mut self.class_ctx_y,
                            &mut self.sign_ctx_y,
                            &mut self.mag_bit_ctx_y,
                        );
                    }
                    hits_so_far.push(hit);
                } else {
                    Self::encode_component(
                        writer,
                        block.delta_x,
                        &mut self.class_ctx_x,
                        &mut self.sign_ctx_x,
                        &mut self.mag_bit_ctx_x,
                    );
                    Self::encode_component(
                        writer,
                        block.delta_y,
                        &mut self.class_ctx_y,
                        &mut self.sign_ctx_y,
                        &mut self.mag_bit_ctx_y,
                    );
                    hits_so_far.push(false);
                }

                if !encoded_fractional_via_candidate {
                    self.encode_fractional(writer, block.flags);
                }
            } else {
                hits_so_far.push(false);
            }

            modes_so_far.push(block.mode);
        }
    }

    fn encode_mode(&mut self, writer: &mut RansWriter32<SharedBuffer>, mode: MvMode, ctx: usize) {
        let is_new = matches!(mode, MvMode::New);
        writer
            .put(is_new, &mut self.mode_is_new_ctx[ctx])
            .expect("RANS writer failure on mode split 0");
        if is_new {
            return;
        }

        let non_zero = !matches!(mode, MvMode::Zero);
        writer
            .put(non_zero, &mut self.mode_is_nonzero_ctx[ctx])
            .expect("RANS writer failure on mode split 1");
        if !non_zero {
            return;
        }

        let is_near = matches!(mode, MvMode::Near);
        writer
            .put(is_near, &mut self.mode_is_near_ctx[ctx])
            .expect("RANS writer failure on mode split 2");
        if is_near {
            return;
        }

        let is_temporal = matches!(mode, MvMode::Temporal);
        writer
            .put(is_temporal, &mut self.mode_is_temporal_ctx[ctx])
            .expect("RANS writer failure on mode split 3");
        if is_temporal {
            return;
        }

        let is_top_right = matches!(mode, MvMode::TopRight);
        writer
            .put(is_top_right, &mut self.mode_is_top_right_ctx[ctx])
            .expect("RANS writer failure on mode split 4");
        if is_top_right {
            return;
        }

        let is_top_left = matches!(mode, MvMode::TopLeft);
        writer
            .put(is_top_left, &mut self.mode_is_top_left_ctx[ctx])
            .expect("RANS writer failure on mode split 5");
    }

    fn encode_component(
        writer: &mut RansWriter32<SharedBuffer>,
        delta: i8,
        class_ctx: &mut [VP8Context; MV_CLASS_TREE_DEPTH],
        sign_ctx: &mut VP8Context,
        mag_ctx: &mut [VP8Context; MV_MAX_MAG_BITS],
    ) {
        let mag = (i16::from(delta).abs()) as u16;
        let class = mv_class_from_magnitude(mag);
        encode_class_symbol(writer, class_ctx, class);
        if class == 0 {
            return;
        }

        let sign = delta < 0;
        writer
            .put(sign, sign_ctx)
            .expect("RANS writer failure on component sign");

        let base = mv_class_base(class);
        let bits = mv_class_bits(class);
        let offset = mag - base;
        for bit in 0..bits {
            let bit_val = (offset >> bit) & 1 != 0;
            writer
                .put(bit_val, &mut mag_ctx[bit as usize])
                .expect("RANS writer failure on component mag bit");
        }
    }

    fn encode_fractional(&mut self, writer: &mut RansWriter32<SharedBuffer>, flags: u8) {
        let sub_x = flags & 0x03;
        let has_frac_x = sub_x != 0;
        writer
            .put(has_frac_x, &mut self.sub_x_has_ctx)
            .expect("RANS writer failure on frac-x flag");
        if has_frac_x {
            let sign_x = sub_x == 1;
            writer
                .put(sign_x, &mut self.sub_x_sign_ctx)
                .expect("RANS writer failure on frac-x sign");
        }

        let sub_y = (flags >> 2) & 0x03;
        let has_frac_y = sub_y != 0;
        writer
            .put(has_frac_y, &mut self.sub_y_has_ctx)
            .expect("RANS writer failure on frac-y flag");
        if has_frac_y {
            let sign_y = sub_y == 1;
            writer
                .put(sign_y, &mut self.sub_y_sign_ctx)
                .expect("RANS writer failure on frac-y sign");
        }
    }

    /// Encode a frame and get its compressed data
    ///
    /// This creates a fresh RANS writer + buffer, encodes the frame with the
    /// current contexts, finalizes the writer, and returns the compressed bytes.
    /// The contexts stored in this encoder are updated and reused across frames.
    ///
    /// # Arguments
    /// * `mv_interleaved` - Motion vector data as interleaved bytes
    ///
    /// # Returns
    /// Vector containing the compressed bytes for this frame
    pub fn encode_frame_and_get_data(
        &mut self,
        blocks: &[MvCodedBlock],
        blocks_w: usize,
        blocks_h: usize,
    ) -> Vec<u8> {
        // Local buffer shared with the writer
        let buf = Rc::new(RefCell::new(Vec::new()));
        let writer_handle = SharedBuffer(Rc::clone(&buf));
        let mut writer = RansWriter32::new(writer_handle);

        // Encode this frame into the local writer, updating our contexts
        self.encode_frame_to_writer(&mut writer, blocks, blocks_w, blocks_h);

        // Finalize writer and return the buffer contents
        writer.finish().unwrap();

        // Update candidates for the next frame
        self.update_candidates(blocks);

        buf.borrow().clone()
    }

    fn update_candidates(&mut self, blocks: &[MvCodedBlock]) {
        // Decay existing counts
        for count in self.candidate_counts.values_mut() {
            *count = (*count * 7) / 8;
        }
        self.total_new_history = (self.total_new_history * 7) / 8;

        // Add new counts
        for block in blocks {
            if block.mode == MvMode::New {
                let frac_flags = block.flags & 0x0F;
                *self
                    .candidate_counts
                    .entry((block.delta_x, block.delta_y, frac_flags))
                    .or_insert(0) += 1;
                self.total_new_history += 1;
            }
        }

        // Prune small counts to keep map size manageable
        self.candidate_counts.retain(|_, &mut c| c > 0);

        if self.total_new_history < 100 {
            self.candidates.clear();
            return;
        }

        let mut sorted: Vec<_> = self
            .candidate_counts
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        // Sort by frequency descending, then by value ascending for determinism
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Threshold: > 1% of history
        let threshold = self.total_new_history / 100;

        self.candidates = sorted
            .into_iter()
            .filter(|&(_, count)| count > threshold)
            .take(CANDIDATE_SIZE)
            .map(|(k, _)| k)
            .collect();
    }
}

/// RANS decoder for motion vectors
///
/// This decoder consumes compressed motion vector bytes and decodes them frame by frame.
/// Probabilities are maintained across frames for better compression.
/// Call `consume()` to add compressed bytes, then `decode_frame()` for each frame.
pub struct MvRansDecoder {
    reader: Option<RansReader32<Cursor<Vec<u8>>>>,
    // Contexts for decoding components
    skip_ctx: VP8Context,
    mode_is_new_ctx: [VP8Context; MODE_CTXS],
    mode_is_nonzero_ctx: [VP8Context; MODE_CTXS],
    mode_is_near_ctx: [VP8Context; MODE_CTXS],
    mode_is_temporal_ctx: [VP8Context; MODE_CTXS],
    mode_is_top_right_ctx: [VP8Context; MODE_CTXS],
    mode_is_top_left_ctx: [VP8Context; MODE_CTXS],
    new_base_is_nearest_ctx: [VP8Context; MODE_CTXS],
    new_base_is_near_ctx: [VP8Context; MODE_CTXS],
    new_base_is_top_right_ctx: [VP8Context; MODE_CTXS],
    new_base_is_top_left_ctx: [VP8Context; MODE_CTXS],
    class_ctx_x: [VP8Context; MV_CLASS_TREE_DEPTH],
    class_ctx_y: [VP8Context; MV_CLASS_TREE_DEPTH],
    sign_ctx_x: VP8Context,
    sign_ctx_y: VP8Context,
    mag_bit_ctx_x: [VP8Context; MV_MAX_MAG_BITS],
    mag_bit_ctx_y: [VP8Context; MV_MAX_MAG_BITS],
    sub_x_has_ctx: VP8Context,
    sub_x_sign_ctx: VP8Context,
    sub_y_has_ctx: VP8Context,
    sub_y_sign_ctx: VP8Context,
    candidate_hit_ctx: [VP8Context; 3],
    candidate_idx_ctx: [VP8Context; CANDIDATE_IDX_DEPTH],
    candidates: Vec<(i8, i8, u8)>,
    candidate_counts: std::collections::HashMap<(i8, i8, u8), u32>,
    total_new_history: u32,
}

impl MvRansDecoder {
    /// Create a new RANS decoder for motion vectors
    ///
    /// This decoder will be used for the entire video, maintaining probability
    /// contexts across all frames.
    ///
    /// # Returns
    /// A new `MvRansDecoder` ready for decoding operations
    pub fn new() -> Self {
        Self {
            reader: None,
            skip_ctx: VP8Context::default(),
            mode_is_new_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_nonzero_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_near_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_temporal_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_top_right_ctx: [VP8Context::default(); MODE_CTXS],
            mode_is_top_left_ctx: [VP8Context::default(); MODE_CTXS],
            new_base_is_nearest_ctx: [VP8Context::default(); MODE_CTXS],
            new_base_is_near_ctx: [VP8Context::default(); MODE_CTXS],
            new_base_is_top_right_ctx: [VP8Context::default(); MODE_CTXS],
            new_base_is_top_left_ctx: [VP8Context::default(); MODE_CTXS],
            class_ctx_x: [VP8Context::default(); MV_CLASS_TREE_DEPTH],
            class_ctx_y: [VP8Context::default(); MV_CLASS_TREE_DEPTH],
            sign_ctx_x: VP8Context::default(),
            sign_ctx_y: VP8Context::default(),
            mag_bit_ctx_x: [VP8Context::default(); MV_MAX_MAG_BITS],
            mag_bit_ctx_y: [VP8Context::default(); MV_MAX_MAG_BITS],
            sub_x_has_ctx: VP8Context::default(),
            sub_x_sign_ctx: VP8Context::default(),
            sub_y_has_ctx: VP8Context::default(),
            sub_y_sign_ctx: VP8Context::default(),
            candidate_hit_ctx: [VP8Context::default(); 3],
            candidate_idx_ctx: [VP8Context::default(); CANDIDATE_IDX_DEPTH],
            candidates: Vec::new(),
            candidate_counts: std::collections::HashMap::new(),
            total_new_history: 0,
        }
    }

    /// Reset all probability contexts back to their initial state.
    ///
    /// Decoders call this whenever they encounter an intra frame so that
    /// starting from that key frame does not depend on earlier bit history.
    pub fn reset_contexts(&mut self) {
        self.skip_ctx = VP8Context::default();
        self.mode_is_new_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_nonzero_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_near_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_temporal_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_top_right_ctx = [VP8Context::default(); MODE_CTXS];
        self.mode_is_top_left_ctx = [VP8Context::default(); MODE_CTXS];
        self.new_base_is_nearest_ctx = [VP8Context::default(); MODE_CTXS];
        self.new_base_is_near_ctx = [VP8Context::default(); MODE_CTXS];
        self.new_base_is_top_right_ctx = [VP8Context::default(); MODE_CTXS];
        self.new_base_is_top_left_ctx = [VP8Context::default(); MODE_CTXS];
        self.class_ctx_x = [VP8Context::default(); MV_CLASS_TREE_DEPTH];
        self.class_ctx_y = [VP8Context::default(); MV_CLASS_TREE_DEPTH];
        self.sign_ctx_x = VP8Context::default();
        self.sign_ctx_y = VP8Context::default();
        self.mag_bit_ctx_x = [VP8Context::default(); MV_MAX_MAG_BITS];
        self.mag_bit_ctx_y = [VP8Context::default(); MV_MAX_MAG_BITS];
        self.sub_x_has_ctx = VP8Context::default();
        self.sub_x_sign_ctx = VP8Context::default();
        self.sub_y_has_ctx = VP8Context::default();
        self.sub_y_sign_ctx = VP8Context::default();
        self.candidate_hit_ctx = [VP8Context::default(); 3];
        self.candidate_idx_ctx = [VP8Context::default(); CANDIDATE_IDX_DEPTH];
        self.candidates.clear();
        self.candidate_counts.clear();
        self.total_new_history = 0;
    }

    /// Consume compressed motion vector bytes for a single frame
    ///
    /// This should be called at the beginning of each frame with the compressed bytes
    /// for that frame. The decoder contexts are maintained across frames for better compression.
    ///
    /// # Arguments
    /// * `bytes` - The compressed motion vector bytes for this frame
    pub fn consume_frame(&mut self, bytes: &[u8]) {
        // For per-frame decoding, we need to append to existing data or create new reader
        // Since RANS reader consumes from a cursor, we'll create a new reader for each frame
        // but the contexts persist across frames
        let cursor = Cursor::new(bytes.to_vec());
        self.reader = Some(RansReader32::new(cursor).expect("RANS Init Failed"));
    }

    /// Decode motion vectors for a single frame into structured blocks.
    pub fn decode_frame(&mut self, blocks_w: usize, blocks_h: usize) -> Vec<MvCodedBlock> {
        let num_blocks = blocks_w * blocks_h;
        let mut reader = self
            .reader
            .take()
            .expect("Decoder called before consume_frame()");

        let mut result = Vec::with_capacity(num_blocks);
        let mut modes_so_far: Vec<MvMode> = Vec::with_capacity(num_blocks);
        let mut hits_so_far: Vec<bool> = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let bx = block_idx % blocks_w;
            let by = block_idx / blocks_w;

            let skip = reader
                .get(&mut self.skip_ctx)
                .expect("Failed to read skip flag");
            let mut flags = if skip { 0x40 } else { 0 };

            let ctx = mv_mode_context(&modes_so_far, blocks_w, bx, by);
            let mode = self.decode_mode(&mut reader, ctx);

            let mut new_base = 0u8;
            let mut delta_x = 0i8;
            let mut delta_y = 0i8;
            let mut decoded_fractional_via_candidate = false;

            if mode == MvMode::New {
                // Decode NEW base selector using same decision tree as encoder.
                let is_nearest = reader
                    .get(&mut self.new_base_is_nearest_ctx[ctx])
                    .expect("Failed to read NEW base split (nearest)");
                if is_nearest {
                    new_base = 0;
                } else {
                    let is_near = reader
                        .get(&mut self.new_base_is_near_ctx[ctx])
                        .expect("Failed to read NEW base split (near)");
                    if is_near {
                        new_base = 1;
                    } else {
                        let is_top_right = reader
                            .get(&mut self.new_base_is_top_right_ctx[ctx])
                            .expect("Failed to read NEW base split (top-right)");
                        if is_top_right {
                            new_base = 2;
                        } else {
                            let is_top_left = reader
                                .get(&mut self.new_base_is_top_left_ctx[ctx])
                                .expect("Failed to read NEW base split (top-left)");
                            new_base = if is_top_left { 3 } else { 4 };
                        }
                    }
                }

                if !self.candidates.is_empty() {
                    let left_hit = if bx > 0 {
                        hits_so_far[block_idx - 1]
                    } else {
                        false
                    };
                    let top_hit = if by > 0 {
                        hits_so_far[block_idx - blocks_w]
                    } else {
                        false
                    };
                    let hit_ctx_idx = (left_hit as usize) + (top_hit as usize);

                    let hit = reader
                        .get(&mut self.candidate_hit_ctx[hit_ctx_idx])
                        .expect("Failed to read candidate hit");
                    if hit {
                        let idx = decode_candidate_index(&mut reader, &mut self.candidate_idx_ctx);
                        if (idx as usize) < self.candidates.len() {
                            let (dx, dy, f) = self.candidates[idx as usize];
                            delta_x = dx;
                            delta_y = dy;
                            flags |= f;
                            decoded_fractional_via_candidate = true;
                        } else {
                            delta_x = 0;
                            delta_y = 0;
                        }
                    } else {
                        delta_x = Self::decode_component(
                            &mut reader,
                            &mut self.class_ctx_x,
                            &mut self.sign_ctx_x,
                            &mut self.mag_bit_ctx_x,
                        );
                        delta_y = Self::decode_component(
                            &mut reader,
                            &mut self.class_ctx_y,
                            &mut self.sign_ctx_y,
                            &mut self.mag_bit_ctx_y,
                        );
                    }
                    hits_so_far.push(hit);
                } else {
                    delta_x = Self::decode_component(
                        &mut reader,
                        &mut self.class_ctx_x,
                        &mut self.sign_ctx_x,
                        &mut self.mag_bit_ctx_x,
                    );
                    delta_y = Self::decode_component(
                        &mut reader,
                        &mut self.class_ctx_y,
                        &mut self.sign_ctx_y,
                        &mut self.mag_bit_ctx_y,
                    );
                    hits_so_far.push(false);
                }

                if !decoded_fractional_via_candidate {
                    flags |= self.decode_fractional(&mut reader);
                }
            } else {
                hits_so_far.push(false);
            }

            result.push(MvCodedBlock {
                mode,
                new_base,
                delta_x,
                delta_y,
                flags,
            });
            modes_so_far.push(mode);
        }

        self.reader = Some(reader);

        // Update candidates for the next frame
        self.update_candidates(&result);

        result
    }

    fn update_candidates(&mut self, blocks: &[MvCodedBlock]) {
        // Decay old history
        for count in self.candidate_counts.values_mut() {
            *count = (*count * 7) / 8;
        }
        self.total_new_history = (self.total_new_history * 7) / 8;

        // Add new frame data
        for block in blocks {
            if block.mode == MvMode::New {
                let frac_flags = block.flags & 0x0F;
                *self
                    .candidate_counts
                    .entry((block.delta_x, block.delta_y, frac_flags))
                    .or_insert(0) += 1;
                self.total_new_history += 1;
            }
        }

        // Prune very small counts to keep map size reasonable
        self.candidate_counts.retain(|_, &mut c| c > 0);

        if self.total_new_history < 100 {
            self.candidates.clear();
            return;
        }

        let mut sorted: Vec<_> = self
            .candidate_counts
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let threshold = self.total_new_history / 100;

        self.candidates = sorted
            .into_iter()
            .filter(|&(_, count)| count > threshold)
            .take(CANDIDATE_SIZE)
            .map(|(k, _)| k)
            .collect();
    }

    fn decode_mode(&mut self, reader: &mut RansReader32<Cursor<Vec<u8>>>, ctx: usize) -> MvMode {
        let is_new = reader
            .get(&mut self.mode_is_new_ctx[ctx])
            .expect("Failed to read mode split 0");
        if is_new {
            return MvMode::New;
        }

        let non_zero = reader
            .get(&mut self.mode_is_nonzero_ctx[ctx])
            .expect("Failed to read mode split 1");
        if !non_zero {
            return MvMode::Zero;
        }

        let is_near = reader
            .get(&mut self.mode_is_near_ctx[ctx])
            .expect("Failed to read mode split 2");
        if is_near {
            return MvMode::Near;
        }

        let is_temporal = reader
            .get(&mut self.mode_is_temporal_ctx[ctx])
            .expect("Failed to read mode split 3");
        if is_temporal {
            return MvMode::Temporal;
        }

        let is_top_right = reader
            .get(&mut self.mode_is_top_right_ctx[ctx])
            .expect("Failed to read mode split 4");
        if is_top_right {
            return MvMode::TopRight;
        }

        let is_top_left = reader
            .get(&mut self.mode_is_top_left_ctx[ctx])
            .expect("Failed to read mode split 5");
        if is_top_left {
            return MvMode::TopLeft;
        }

        MvMode::Nearest
    }

    fn decode_component(
        reader: &mut RansReader32<Cursor<Vec<u8>>>,
        class_ctx: &mut [VP8Context; MV_CLASS_TREE_DEPTH],
        sign_ctx: &mut VP8Context,
        mag_ctx: &mut [VP8Context; MV_MAX_MAG_BITS],
    ) -> i8 {
        let class = decode_class_symbol(reader, class_ctx);
        if class == 0 {
            return 0;
        }

        let sign = reader.get(sign_ctx).expect("Failed to read component sign");

        let base = mv_class_base(class);
        let bits = mv_class_bits(class);
        let mut offset = 0u16;
        for bit in 0..bits {
            let bit_val = reader
                .get(&mut mag_ctx[bit as usize])
                .expect("Failed to read component mag bit");
            if bit_val {
                offset |= 1 << bit;
            }
        }
        let mag = base + offset;
        let value = mag as i16;
        let signed = if sign { -(value as i16) } else { value as i16 };
        signed.clamp(-128, 127) as i8
    }

    fn decode_fractional(&mut self, reader: &mut RansReader32<Cursor<Vec<u8>>>) -> u8 {
        let mut flags = 0u8;
        let has_frac_x = reader
            .get(&mut self.sub_x_has_ctx)
            .expect("Failed to read frac-x flag");
        if has_frac_x {
            let sign_x = reader
                .get(&mut self.sub_x_sign_ctx)
                .expect("Failed to read frac-x sign");
            flags |= if sign_x { 1 } else { 2 };
        }

        let has_frac_y = reader
            .get(&mut self.sub_y_has_ctx)
            .expect("Failed to read frac-y flag");
        if has_frac_y {
            let sign_y = reader
                .get(&mut self.sub_y_sign_ctx)
                .expect("Failed to read frac-y sign");
            flags |= (if sign_y { 1 } else { 2 }) << 2;
        }

        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reitero_video_common::MotionVector;

    fn blocks_from_interleaved(bytes: &[u8]) -> Vec<MvCodedBlock> {
        bytes
            .chunks_exact(3)
            .map(|chunk| {
                let ddx = chunk[0] as i8;
                let ddy = chunk[1] as i8;
                let flags = chunk[2];
                let frac_flags = flags & 0x0F;
                let mode = if ddx == 0 && ddy == 0 && frac_flags == 0 {
                    MvMode::Zero
                } else {
                    MvMode::New
                };
                let mv = MotionVector::from_raw(ddx, ddy, flags);
                MvCodedBlock {
                    mode,
                    new_base: 0,
                    delta_x: ddx,
                    delta_y: ddy,
                    flags: mv.raw_flags(),
                }
            })
            .collect()
    }

    #[test]
    fn test_zigzag_encode_decode_roundtrip() {
        // Test roundtrip for all i8 values
        for n in i8::MIN..=i8::MAX {
            let encoded = zigzag_encode_i8(n);
            let decoded = zigzag_decode_i8(encoded);
            assert_eq!(
                decoded, n,
                "Roundtrip failed for n={}: encoded={}, decoded={}",
                n, encoded, decoded
            );
        }
    }
    #[test]
    fn test_zigzag_encode_extremes() {
        // Test edge cases
        assert_eq!(zigzag_encode_i8(i8::MIN), 255); // -128 -> 255
        assert_eq!(zigzag_encode_i8(i8::MAX), 254); // 127 -> 254
    }

    #[test]
    fn test_zigzag_decode_small_values() {
        // Test that small unsigned values decode to small deltas
        assert_eq!(zigzag_decode_i8(0), 0);
        assert_eq!(zigzag_decode_i8(1), -1);
        assert_eq!(zigzag_decode_i8(2), 1);
        assert_eq!(zigzag_decode_i8(3), -2);
        assert_eq!(zigzag_decode_i8(4), 2);
        assert_eq!(zigzag_decode_i8(5), -3);
        assert_eq!(zigzag_decode_i8(6), 3);
    }

    #[test]
    fn test_zigzag_decode_extremes() {
        // Test edge cases
        assert_eq!(zigzag_decode_i8(255), i8::MIN); // 255 -> -128
        assert_eq!(zigzag_decode_i8(254), i8::MAX); // 254 -> 127
    }

    #[test]
    fn test_zigzag_encode_entropy_centering() {
        // Verify that the encoding centers entropy around zero
        // Small deltas should map to small unsigned values
        let test_values = vec![0, 1, -1, 2, -2, 3, -3, 10, -10, 50, -50];
        for &n in &test_values {
            let encoded = zigzag_encode_i8(n);
            let abs_n = n.abs() as u8;
            // Encoded value should be roughly 2 * abs(n) for small values
            if n >= 0 {
                assert_eq!(encoded, (n as u8) * 2);
            } else {
                assert_eq!(encoded, (abs_n * 2).saturating_sub(1));
            }
        }
    }

    #[test]
    fn test_zigzag_encode_full_table() {
        // Print the full mapping table
        println!("\n=== ZigZag Encoding Table (i8 -> u8) ===");
        println!("{:<6} {:<6} {:<6}", "i8", "u8", "Hex");
        println!("{}", "-".repeat(20));

        // Print a range around zero (most common values)
        for n in -10..=10 {
            let encoded = zigzag_encode_i8(n);
            println!(
                "{:<6} {:<6} {:<6}",
                n,
                encoded,
                format!("0x{:02X}", encoded)
            );
        }

        println!("\n=== Edge Cases ===");
        println!("{:<6} {:<6} {:<6}", "i8", "u8", "Hex");
        println!("{}", "-".repeat(20));
        for &n in &[i8::MIN, -127, -1, 0, 1, 126, i8::MAX] {
            let encoded = zigzag_encode_i8(n);
            println!(
                "{:<6} {:<6} {:<6}",
                n,
                encoded,
                format!("0x{:02X}", encoded)
            );
        }

        // Verify the table is correct
        for n in i8::MIN..=i8::MAX {
            let encoded = zigzag_encode_i8(n);
            let decoded = zigzag_decode_i8(encoded);
            assert_eq!(decoded, n);
        }

        println!("\n✓ All 256 values roundtrip correctly");
    }

    #[test]
    fn test_zigzag_encode_sequential_roundtrip() {
        // Test sequential encoding/decoding
        let values: Vec<i8> = vec![
            0, 1, -1, 2, -2, 5, -5, 10, -10, 50, -50, 100, -100, 127, -128,
        ];
        for &n in &values {
            let encoded = zigzag_encode_i8(n);
            let decoded = zigzag_decode_i8(encoded);
            assert_eq!(decoded, n, "Failed for n={}", n);
        }
    }

    #[test]
    fn test_mv_rans_roundtrip_single_frame() {
        // Test roundtrip for a single frame with some motion vectors
        let mut encoder = MvRansEncoder::new();

        // Create some test motion vectors: [ddx, ddy, flags] per block
        // Note: ddx and ddy are stored as u8 bytes representing i8 values
        // -3 as i8 = 0xFD as u8, -2 as i8 = 0xFE as u8
        let mv_interleaved = vec![
            5u8, 0xFDu8, 0x05, // Block 0: ddx=5, ddy=-3, flags=0x05
            0xFEu8, 1u8, 0x40, // Block 1: ddx=-2, ddy=1, flags=0x40 (skip)
            0u8, 0u8, 0x00, // Block 2: ddx=0, ddy=0, flags=0x00
        ];

        let blocks = blocks_from_interleaved(&mv_interleaved);

        // Encode frame and get compressed data
        let compressed = encoder.encode_frame_and_get_data(&blocks, 3, 1);

        let mut decoder = MvRansDecoder::new();
        decoder.consume_frame(&compressed);
        let decoded = decoder.decode_frame(3, 1); // 3 blocks

        assert_eq!(decoded, blocks, "Roundtrip failed");
    }

    #[test]
    fn test_mv_rans_roundtrip_multiple_frames() {
        // Test roundtrip for multiple frames (probabilities carry over)
        let mut encoder = MvRansEncoder::new();

        // Frame 1: 2 blocks
        let frame1 = vec![
            1u8,
            2u8,
            0x05, // Block 0: ddx=1, ddy=2, flags=0x05
            (-1i8) as u8,
            0u8,
            0x00, // Block 1: ddx=-1, ddy=0, flags=0x00
        ];

        // Frame 2: 3 blocks
        let frame2 = vec![
            0u8,
            0u8,
            0x40, // Block 0: ddx=0, ddy=0, flags=0x40 (skip)
            5u8,
            (-5i8) as u8,
            0x09, // Block 1: ddx=5, ddy=-5, flags=0x09
            (-10i8) as u8,
            10u8,
            0x00, // Block 2: ddx=-10, ddy=10, flags=0x00
        ];

        let frame1_blocks = blocks_from_interleaved(&frame1);
        let frame2_blocks = blocks_from_interleaved(&frame2);

        // Encode frame 1 and get data (encoder contexts are updated)
        let compressed_frame1 = encoder.encode_frame_and_get_data(&frame1_blocks, 2, 1);

        // Encode frame 2 and get data (encoder contexts persist and are updated)
        let compressed_frame2 = encoder.encode_frame_and_get_data(&frame2_blocks, 3, 1);

        let mut decoder = MvRansDecoder::new();
        decoder.consume_frame(&compressed_frame1);
        let decoded_frame1 = decoder.decode_frame(2, 1);
        decoder.consume_frame(&compressed_frame2);
        let decoded_frame2 = decoder.decode_frame(3, 1);

        assert_eq!(decoded_frame1, frame1_blocks, "Frame 1 roundtrip failed");
        assert_eq!(decoded_frame2, frame2_blocks, "Frame 2 roundtrip failed");
    }

    #[test]
    fn test_mv_rans_roundtrip_all_halfpel_codes() {
        // Test all possible half-pixel code combinations
        let mut encoder = MvRansEncoder::new();
        let mut mv_interleaved = Vec::new();

        // Test all 3x3 = 9 valid half-pixel code combinations (codes 0..2)
        for qx in 0..3 {
            for qy in 0..3 {
                let flags = qx | (qy << 2);
                mv_interleaved.push(0u8); // ddx = 0
                mv_interleaved.push(0u8); // ddy = 0
                mv_interleaved.push(flags);
            }
        }

        let blocks = blocks_from_interleaved(&mv_interleaved);
        let compressed = encoder.encode_frame_and_get_data(&blocks, blocks.len(), 1);

        let mut decoder = MvRansDecoder::new();
        decoder.consume_frame(&compressed);
        let decoded = decoder.decode_frame(blocks.len(), 1);

        assert_eq!(decoded, blocks, "Half-pixel roundtrip failed");
    }

    #[test]
    fn test_mv_rans_roundtrip_edge_cases() {
        // Test edge cases: min/max values, all skip flags, etc.
        let mut encoder = MvRansEncoder::new();

        let mv_interleaved = vec![
            i8::MIN as u8,
            i8::MAX as u8,
            0x00, // ddx=MIN, ddy=MAX, no skip
            i8::MAX as u8,
            i8::MIN as u8,
            0x40, // ddx=MAX, ddy=MIN, skip
            0u8,
            0u8,
            0x4A, // ddx=0, ddy=0, max valid half-pel flags (both axes -0.5) + skip
            127u8,
            (-128i8) as u8,
            0x00, // ddx=127, ddy=-128
        ];

        let blocks = blocks_from_interleaved(&mv_interleaved);
        let compressed = encoder.encode_frame_and_get_data(&blocks, blocks.len(), 1);

        let mut decoder = MvRansDecoder::new();
        decoder.consume_frame(&compressed);
        let decoded = decoder.decode_frame(blocks.len(), 1);

        assert_eq!(decoded, blocks, "Edge case roundtrip failed");
    }

    #[test]
    fn test_mv_rans_roundtrip_large_frame() {
        // Test with a larger frame (simulating a real video frame)
        let mut encoder = MvRansEncoder::new();
        let blocks_w = 40;
        let blocks_h = 30;
        let num_blocks = blocks_w * blocks_h;

        let mut mv_interleaved = Vec::with_capacity(num_blocks * 3);
        for i in 0..num_blocks {
            // Create some variation in motion vectors
            let ddx = ((i % 17) as i8).wrapping_sub(8); // Range roughly -8 to 8
            let ddy = ((i % 13) as i8).wrapping_sub(6); // Range roughly -6 to 6
            let qx = (i % 3) as u8;
            let qy = ((i / 3) % 3) as u8;
            let skip = (i % 10 == 0) as u8; // Every 10th block is skipped
            let flags = qx | (qy << 2) | (skip << 6);

            mv_interleaved.push(ddx as u8);
            mv_interleaved.push(ddy as u8);
            mv_interleaved.push(flags);
        }

        let blocks = blocks_from_interleaved(&mv_interleaved);
        let compressed = encoder.encode_frame_and_get_data(&blocks, blocks_w, blocks_h);

        println!(
            "Large frame: {} blocks, compressed size: {} bytes",
            num_blocks,
            compressed.len()
        );

        let mut decoder = MvRansDecoder::new();
        decoder.consume_frame(&compressed);
        let decoded = decoder.decode_frame(blocks_w, blocks_h);

        assert_eq!(decoded, blocks, "Large frame roundtrip failed");
    }

    #[test]
    fn test_mv_rans_roundtrip_500_frames() {
        // Test with 500 frames to verify probabilities carry over correctly
        let mut encoder = MvRansEncoder::new();
        let blocks_w = 40;
        let blocks_h = 30;
        let num_blocks = blocks_w * blocks_h;
        let num_frames = 500;

        // Generate 500 frames of motion vector data
        let mut frames: Vec<Vec<MvCodedBlock>> = Vec::with_capacity(num_frames);
        let mut total_uncompressed = 0usize;

        for frame_idx in 0..num_frames {
            let mut mv_interleaved = Vec::with_capacity(num_blocks * 3);

            for block_idx in 0..num_blocks {
                // Create motion vectors with some patterns that vary across frames
                // This simulates real video motion
                let frame_offset = (frame_idx % 30) as i8; // Cycle every 30 frames
                let block_pattern = (block_idx % 17) as i8;

                // ddx: varies with frame and block position
                let ddx = (frame_offset.wrapping_add(block_pattern).wrapping_sub(8)) as i8;

                // ddy: different pattern
                let ddy = ((frame_idx % 13) as i8).wrapping_sub(6);

                // Sub-pixel: vary with frame
                let qx = ((frame_idx + block_idx) % 3) as u8;
                let qy = ((frame_idx * 2 + block_idx) % 3) as u8;

                // Skip: every 10th block, or when motion is very small
                let skip = (block_idx % 10 == 0 || (ddx.abs() < 2 && ddy.abs() < 2)) as u8;

                let flags = qx | (qy << 2) | (skip << 6);

                mv_interleaved.push(ddx as u8);
                mv_interleaved.push(ddy as u8);
                mv_interleaved.push(flags);
            }

            let blocks = blocks_from_interleaved(&mv_interleaved);
            total_uncompressed += blocks.len() * 4; // mode + two deltas + flags
            frames.push(blocks);
        }

        // Encode all frames (encoder lives across frames, contexts persist)
        println!("\n=== Encoding 500 frames ===");
        let mut compressed_frames = Vec::new();
        let mut total_compressed = 0usize;

        for (frame_idx, frame_data) in frames.iter().enumerate() {
            // Encode frame and get compressed data (contexts are updated)
            let frame_compressed =
                encoder.encode_frame_and_get_data(frame_data, blocks_w, blocks_h);
            total_compressed += frame_compressed.len();
            compressed_frames.push(frame_compressed);

            if (frame_idx + 1) % 100 == 0 {
                println!("Encoded {} frames...", frame_idx + 1);
            }
        }

        println!("\n=== Compression Statistics ===");
        println!("Frames: {}", num_frames);
        println!("Blocks per frame: {}", num_blocks);
        println!("Total blocks: {}", num_frames * num_blocks);
        println!("Uncompressed size: {} bytes", total_uncompressed);
        println!("Compressed size: {} bytes", total_compressed);
        println!(
            "Compression ratio: {:.2}%",
            (total_compressed as f64 / total_uncompressed as f64) * 100.0
        );
        println!(
            "Average bytes per frame: {:.2}",
            total_compressed as f64 / num_frames as f64
        );
        println!(
            "Average bytes per block: {:.4}",
            total_compressed as f64 / (num_frames * num_blocks) as f64
        );

        // Decode all frames (decoder lives across frames, contexts persist)
        println!("\n=== Decoding 500 frames ===");
        let mut decoder = MvRansDecoder::new();

        for (frame_idx, (expected_frame, frame_compressed)) in
            frames.iter().zip(compressed_frames.iter()).enumerate()
        {
            decoder.consume_frame(frame_compressed);
            let decoded = decoder.decode_frame(blocks_w, blocks_h);

            assert_eq!(
                decoded, *expected_frame,
                "Frame {}: roundtrip failed",
                frame_idx
            );

            if (frame_idx + 1) % 100 == 0 {
                println!("Decoded {} frames...", frame_idx + 1);
            }
        }

        println!("\n✓ All 500 frames roundtrip correctly!");
    }
}
