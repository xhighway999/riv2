//! Range Asymmetric Numeral Systems (RANS) encoding for DCT coefficients
//!
//! This module provides RANS encoding/decoding as a drop-in replacement for RLE.
//! It uses a streaming design where the encoder accumulates state and the decoder
//! consumes bytes incrementally.
use std::{
    cell::RefCell,
    io::{Cursor, Write},
    rc::Rc,
};

use reitero_video_common::rans::{RansDecoder, RansEncoder, BinProb};

struct SharedBuffer(Rc<RefCell<Vec<u8>>>);

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.borrow_mut().flush()
    }
}

// ---------------------------------------------------------------------------
// EOB class scheme: value → class index
//   Class 0: value=0
//   Class 1: value=1
//   Class k (k≥2): values [2^(k-1), 2^k - 1]
// Max classes: luma 256 coeffs → max EOB 255 → class 8 (9 classes)
//              chroma 64 coeffs → max EOB 63  → class 6 (7 classes)
// ---------------------------------------------------------------------------

const EOB_Y_CLASSES: usize = 9;
const EOB_UV_CLASSES: usize = 7;

// Magnitude class count: values 0..=4095 → classes 0..=12 (13 classes)
// Needs headroom for DC prediction deltas which can exceed single-coeff range.
const MAG_CLASSES: usize = 13;

// Luma magnitude bands: DC, low AC, mid AC, high AC (4 bands × 9 classes)
const LUMA_MAG_BANDS: usize = 4;
// Chroma magnitude bands: DC, low AC, high AC (3 bands × 9 classes)
const CHROMA_MAG_BANDS: usize = 3;

#[inline]
fn eob_class(value: usize) -> usize {
    if value <= 1 {
        value
    } else {
        (usize::BITS - value.leading_zeros()) as usize
    }
}

#[inline]
fn luma_mag_band(pos: usize) -> usize {
    if pos == 0 { 0 }
    else if pos < 16 { 1 }
    else if pos < 64 { 2 }
    else { 3 }
}

#[inline]
fn chroma_mag_band(pos: usize) -> usize {
    if pos == 0 { 0 }
    else if pos < 8 { 1 }
    else { 2 }
}

/// Encode a magnitude value using class-based coding (same scheme as EOB).
/// Class is coded as truncated unary; offset within class as bypass bits.
fn encode_magnitude(
    writer: &mut RansEncoder<SharedBuffer>,
    value: usize,
    contexts: &mut [BinProb],  // exactly MAG_CLASSES elements
) {
    let max_class = contexts.len() - 1;
    let class = eob_class(value); // reuse same log2-based class mapping

    // Truncated unary for the class
    for c in 0..class {
        writer.put(true, &mut contexts[c]).expect("mag class continue");
    }
    if class < max_class {
        writer.put(false, &mut contexts[class]).expect("mag class stop");
    }

    // Offset bits within class (bypass)
    if class >= 2 {
        let base = 1usize << (class - 1);
        let offset = value - base;
        for bit in (0..(class - 1)).rev() {
            writer
                .put_bypass((offset >> bit) & 1 != 0)
                .expect("mag offset bit");
        }
    }
}

/// Decode a magnitude value using class-based coding (mirrors encode_magnitude).
fn decode_magnitude(
    reader: &mut RansDecoder<Cursor<Vec<u8>>>,
    contexts: &mut [BinProb],  // exactly MAG_CLASSES elements
) -> usize {
    let max_class = contexts.len() - 1;

    let mut class = 0;
    while class < max_class {
        if !reader.get(&mut contexts[class]).expect("mag class bit") {
            break;
        }
        class += 1;
    }

    match class {
        0 => 0,
        1 => 1,
        _ => {
            let base = 1usize << (class - 1);
            let mut offset = 0usize;
            for _ in 0..(class - 1) {
                offset = (offset << 1) | reader.get_bypass().expect("mag offset bit") as usize;
            }
            base + offset
        }
    }
}

/// Encode an EOB value using class-based coding.
///
/// Class is coded as truncated unary with per-position contexts.
/// Offset within class is coded as bypass bits (MSB first).
fn encode_eob(
    writer: &mut RansEncoder<SharedBuffer>,
    value: usize,
    contexts: &mut [BinProb],
) {
    let max_class = contexts.len() - 1;
    let class = eob_class(value);

    // Truncated unary for the class
    for c in 0..class {
        writer.put(true, &mut contexts[c]).expect("EOB class continue");
    }
    if class < max_class {
        writer.put(false, &mut contexts[class]).expect("EOB class stop");
    }

    // Offset bits within class (bypass)
    if class >= 2 {
        let base = 1usize << (class - 1);
        let offset = value - base;
        for bit in (0..(class - 1)).rev() {
            writer
                .put_bypass((offset >> bit) & 1 != 0)
                .expect("EOB offset bit");
        }
    }
}

/// Decode an EOB value using class-based coding (mirrors encode_eob).
fn decode_eob(
    reader: &mut RansDecoder<Cursor<Vec<u8>>>,
    contexts: &mut [BinProb],
) -> usize {
    let max_class = contexts.len() - 1;

    // Decode class via truncated unary
    let mut class = 0;
    while class < max_class {
        if !reader.get(&mut contexts[class]).expect("EOB class bit") {
            break;
        }
        class += 1;
    }

    match class {
        0 => 0,
        1 => 1,
        _ => {
            let base = 1usize << (class - 1);
            let mut offset = 0usize;
            for _ in 0..(class - 1) {
                offset = (offset << 1) | reader.get_bypass().expect("EOB offset bit") as usize;
            }
            base + offset
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

pub struct DctRansEncoder {
    output: Rc<RefCell<Vec<u8>>>,
    writer: RansEncoder<SharedBuffer>,
    // Per-position significance contexts
    luma_contexts: [BinProb; 256],
    chroma_contexts: [BinProb; 64],
    // Band × class magnitude contexts (class-based coding per frequency band)
    luma_mag: [[BinProb; MAG_CLASSES]; LUMA_MAG_BANDS],
    chroma_mag: [[BinProb; MAG_CLASSES]; CHROMA_MAG_BANDS],
    // Separate EOB contexts for Y and UV
    eob_y: [BinProb; EOB_Y_CLASSES],
    eob_uv: [BinProb; EOB_UV_CLASSES],
    // DC prediction: previous block's DC per plane (intra only)
    prev_dc: [i16; 3], // [Y, U, V]
    dc_prediction: bool,
}

impl DctRansEncoder {
    pub fn new() -> Self {
        Self::with_dc_prediction(false)
    }

    pub fn with_dc_prediction(dc_prediction: bool) -> Self {
        let buf = Rc::new(RefCell::new(Vec::new()));
        let writer_handle = SharedBuffer(Rc::clone(&buf));
        Self {
            output: buf,
            writer: RansEncoder::new(writer_handle),
            luma_contexts: [BinProb::default(); 256],
            chroma_contexts: [BinProb::default(); 64],
            luma_mag: [[BinProb::default(); MAG_CLASSES]; LUMA_MAG_BANDS],
            chroma_mag: [[BinProb::default(); MAG_CLASSES]; CHROMA_MAG_BANDS],
            eob_y: [BinProb::default(); EOB_Y_CLASSES],
            eob_uv: [BinProb::default(); EOB_UV_CLASSES],
            prev_dc: [0; 3],
            dc_prediction,
        }
    }

    pub fn encode_block(
        &mut self,
        y: &[i16],
        u: &[i16],
        v: &[i16],
        _y_size: usize,
        _u_size: usize,
        _v_size: usize,
    ) {
        self.encode_single_plane(y, true, 0);
        self.encode_single_plane(u, false, 1);
        self.encode_single_plane(v, false, 2);
    }

    fn encode_single_plane(&mut self, plane: &[i16], is_luma: bool, plane_idx: usize) {
        // Apply DC prediction when enabled (intra frames)
        let (dc_val, last_nz) = if self.dc_prediction {
            let dc_orig = plane[0];
            let dc_pred = self.prev_dc[plane_idx];
            let dc_delta = dc_orig - dc_pred;
            self.prev_dc[plane_idx] = dc_orig;

            let last_nz_ac = plane[1..].iter().rposition(|&x| x != 0).map(|p| p + 1);
            let last_nz = match (dc_delta != 0, last_nz_ac) {
                (_, Some(ac_pos)) => ac_pos,
                (true, None) => 0,
                (false, None) => 0,
            };
            (dc_delta, last_nz)
        } else {
            let last_nz = plane.iter().rposition(|&x| x != 0).unwrap_or(0);
            (plane[0], last_nz)
        };

        // Destructure self so the borrow checker sees disjoint field access
        let Self {
            writer,
            luma_contexts,
            chroma_contexts,
            luma_mag,
            chroma_mag,
            eob_y,
            eob_uv,
            output: _,
            prev_dc: _,
            dc_prediction: _,
        } = self;

        // 1. EOB: class-based coding with plane-specific contexts
        {
            let eob = if is_luma { &mut eob_y[..] } else { &mut eob_uv[..] };
            encode_eob(writer, last_nz, eob);
        }

        // 2. Coefficient coding
        let sig: &mut [BinProb] = if is_luma {
            &mut luma_contexts[..]
        } else {
            &mut chroma_contexts[..]
        };
        for i in 0..=last_nz {
            let val = if i == 0 { dc_val } else { plane[i] };

            // Significance bit
            writer
                .put(val != 0, &mut sig[i])
                .expect("significance bit");

            if val != 0 {
                // Sign (bypass — 50/50)
                writer.put_bypass(val < 0).expect("sign bit");

                // Magnitude: class-based coding with band-specific contexts
                let abs_val = (val.abs() as usize) - 1;
                let band = if is_luma {
                    luma_mag_band(i)
                } else {
                    chroma_mag_band(i)
                };
                let mag_ctx = if is_luma {
                    &mut luma_mag[band]
                } else {
                    &mut chroma_mag[band]
                };
                encode_magnitude(writer, abs_val, mag_ctx);
            }
        }
    }

    pub fn finish(&mut self) -> Vec<u8> {
        self.writer.finish().unwrap();
        self.output.borrow_mut().clone()
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

pub struct DctRansDecoder {
    reader: Option<RansDecoder<Cursor<Vec<u8>>>>,
    luma_contexts: [BinProb; 256],
    chroma_contexts: [BinProb; 64],
    luma_mag: [[BinProb; MAG_CLASSES]; LUMA_MAG_BANDS],
    chroma_mag: [[BinProb; MAG_CLASSES]; CHROMA_MAG_BANDS],
    eob_y: [BinProb; EOB_Y_CLASSES],
    eob_uv: [BinProb; EOB_UV_CLASSES],
    // DC prediction: previous block's DC per plane (intra only)
    prev_dc: [i16; 3], // [Y, U, V]
    dc_prediction: bool,
}

impl DctRansDecoder {
    pub fn new() -> Self {
        Self::with_dc_prediction(false)
    }

    pub fn with_dc_prediction(dc_prediction: bool) -> Self {
        Self {
            reader: None,
            luma_contexts: [BinProb::default(); 256],
            chroma_contexts: [BinProb::default(); 64],
            luma_mag: [[BinProb::default(); MAG_CLASSES]; LUMA_MAG_BANDS],
            chroma_mag: [[BinProb::default(); MAG_CLASSES]; CHROMA_MAG_BANDS],
            eob_y: [BinProb::default(); EOB_Y_CLASSES],
            eob_uv: [BinProb::default(); EOB_UV_CLASSES],
            prev_dc: [0; 3],
            dc_prediction,
        }
    }

    pub fn consume(&mut self, bytes: &[u8]) {
        reitero_video_common::Instrument::start_measure("20a_rans_consume_rans");
        let cursor = Cursor::new(bytes.to_vec());
        self.reader = Some(RansDecoder::new(cursor).expect("RANS Init Failed"));
        reitero_video_common::Instrument::stop_measure("20a_rans_consume_rans");
    }

    #[cfg(test)]
    pub fn decode_block(
        &mut self,
        y_size: usize,
        u_size: usize,
        v_size: usize,
    ) -> (Vec<i16>, Vec<i16>, Vec<i16>) {
        let mut y = vec![0i16; y_size];
        let mut u = vec![0i16; u_size];
        let mut v = vec![0i16; v_size];
        self.decode_block_into(&mut y, &mut u, &mut v);
        (y, u, v)
    }

    pub fn decode_block_into(&mut self, y: &mut [i16], u: &mut [i16], v: &mut [i16]) {
        self.decode_single_plane_into(y, true, 0);
        self.decode_single_plane_into(u, false, 1);
        self.decode_single_plane_into(v, false, 2);
    }

    fn decode_single_plane_into(&mut self, block: &mut [i16], is_luma: bool, plane_idx: usize) {
        let size = block.len();
        let dc_pred = if self.dc_prediction { self.prev_dc[plane_idx] } else { 0 };

        // Destructure for split borrows
        let Self {
            reader,
            luma_contexts,
            chroma_contexts,
            luma_mag,
            chroma_mag,
            eob_y,
            eob_uv,
            prev_dc: _,
            dc_prediction: _,
        } = self;

        let reader = reader.as_mut().expect("Decoder called before consume()");

        // 1. Decode EOB via class scheme
        let end_idx = {
            let eob = if is_luma { &mut eob_y[..] } else { &mut eob_uv[..] };
            decode_eob(reader, eob)
        };

        assert!(
            end_idx < size,
            "Decoded EOB index {} exceeds block size {}",
            end_idx,
            size
        );

        // 2. Decode coefficients
        let sig: &mut [BinProb] = if is_luma {
            &mut luma_contexts[..]
        } else {
            &mut chroma_contexts[..]
        };

        for i in 0..=end_idx {
            let is_nz = reader.get(&mut sig[i]).expect("significance bit");

            if is_nz {
                let is_negative = reader.get_bypass().expect("sign bit");

                // Class-based magnitude decode with band-specific contexts
                let band = if is_luma {
                    luma_mag_band(i)
                } else {
                    chroma_mag_band(i)
                };
                let mag_ctx = if is_luma {
                    &mut luma_mag[band]
                } else {
                    &mut chroma_mag[band]
                };
                let abs_val = (decode_magnitude(reader, mag_ctx) + 1) as i16;

                block[i] = if is_negative { -abs_val } else { abs_val };
            }
        }

        // 3. Reconstruct DC from delta + prediction (when DC prediction is enabled)
        if self.dc_prediction {
            block[0] = block[0] + dc_pred;
            self.prev_dc[plane_idx] = block[0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rans_roundtrip_single_block() {
        const Y_SIZE: usize = 256;
        const U_SIZE: usize = 64;
        const V_SIZE: usize = 64;

        let mut y_block = vec![0i16; Y_SIZE];
        y_block[0] = 42;
        y_block[1] = -15;
        y_block[10] = 7;

        let mut u_block = vec![0i16; U_SIZE];
        u_block[0] = 20;
        u_block[5] = -8;

        let mut v_block = vec![0i16; V_SIZE];
        v_block[0] = -12;
        v_block[3] = 5;

        let mut encoder = DctRansEncoder::new();
        encoder.encode_block(&y_block, &u_block, &v_block, Y_SIZE, U_SIZE, V_SIZE);
        let encoded = encoder.finish();

        let mut decoder = DctRansDecoder::new();
        decoder.consume(&encoded);
        let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

        assert_eq!(y_block, decoded_y, "Y block mismatch");
        assert_eq!(u_block, decoded_u, "U block mismatch");
        assert_eq!(v_block, decoded_v, "V block mismatch");
    }

    #[test]
    fn test_rans_roundtrip_640x480_residual() {
        const STORAGE_W: usize = 640;
        const STORAGE_H: usize = 480;
        const BLOCKS_W: usize = STORAGE_W / 16;
        const BLOCKS_H: usize = STORAGE_H / 16;
        const BLOCKS_TOTAL: usize = BLOCKS_W * BLOCKS_H;

        const Y_SIZE: usize = 256;
        const U_SIZE: usize = 64;
        const V_SIZE: usize = 64;

        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(12345);

        let mut all_y_blocks = Vec::new();
        let mut all_u_blocks = Vec::new();
        let mut all_v_blocks = Vec::new();

        let mut non_zero_blocks = 0;
        for _ in 0..BLOCKS_TOTAL {
            let mut y_block = vec![0i16; Y_SIZE];
            if rng.gen_bool(0.8) {
                y_block[0] = rng.gen_range(-200..=200);
            }
            for i in 1..Y_SIZE {
                if rng.gen_bool(0.1) {
                    y_block[i] = rng.gen_range(-100..=100);
                }
            }

            let mut u_block = vec![0i16; U_SIZE];
            if rng.gen_bool(0.7) {
                u_block[0] = rng.gen_range(-150..=150);
            }
            for i in 1..U_SIZE {
                if rng.gen_bool(0.15) {
                    u_block[i] = rng.gen_range(-80..=80);
                }
            }

            let mut v_block = vec![0i16; V_SIZE];
            if rng.gen_bool(0.7) {
                v_block[0] = rng.gen_range(-150..=150);
            }
            for i in 1..V_SIZE {
                if rng.gen_bool(0.15) {
                    v_block[i] = rng.gen_range(-80..=80);
                }
            }

            let has_non_zero = y_block.iter().any(|&x| x != 0)
                || u_block.iter().any(|&x| x != 0)
                || v_block.iter().any(|&x| x != 0);

            if has_non_zero {
                non_zero_blocks += 1;
            }

            all_y_blocks.push(y_block);
            all_u_blocks.push(u_block);
            all_v_blocks.push(v_block);
        }

        assert!(
            non_zero_blocks > 0,
            "Test requires at least one block with non-zero values"
        );

        let mut encoder = DctRansEncoder::new();
        for i in 0..BLOCKS_TOTAL {
            encoder.encode_block(
                &all_y_blocks[i],
                &all_u_blocks[i],
                &all_v_blocks[i],
                Y_SIZE,
                U_SIZE,
                V_SIZE,
            );
        }
        let encoded = encoder.finish();

        let mut decoder = DctRansDecoder::new();
        decoder.consume(&encoded);

        for i in 0..BLOCKS_TOTAL {
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            assert_eq!(all_y_blocks[i], decoded_y, "Y block {} mismatch", i);
            assert_eq!(all_u_blocks[i], decoded_u, "U block {} mismatch", i);
            assert_eq!(all_v_blocks[i], decoded_v, "V block {} mismatch", i);
        }
    }

    #[test]
    fn test_rans_roundtrip_partial_zero_planes() {
        const Y_SIZE: usize = 256;
        const U_SIZE: usize = 64;
        const V_SIZE: usize = 64;

        // Test case 1: Y has data, U and V are all zeros
        {
            let mut y_block = vec![0i16; Y_SIZE];
            y_block[0] = 100;
            y_block[5] = -50;
            y_block[20] = 25;

            let u_block = vec![0i16; U_SIZE];
            let v_block = vec![0i16; V_SIZE];

            let mut encoder = DctRansEncoder::new();
            encoder.encode_block(&y_block, &u_block, &v_block, Y_SIZE, U_SIZE, V_SIZE);
            let encoded = encoder.finish();

            let mut decoder = DctRansDecoder::new();
            decoder.consume(&encoded);
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            assert_eq!(y_block, decoded_y, "Y block mismatch (U and V zero)");
            assert_eq!(u_block, decoded_u, "U block mismatch (should be zero)");
            assert_eq!(v_block, decoded_v, "V block mismatch (should be zero)");
        }

        // Test case 2: U has data, Y and V are all zeros
        {
            let y_block = vec![0i16; Y_SIZE];

            let mut u_block = vec![0i16; U_SIZE];
            u_block[0] = 75;
            u_block[3] = -30;
            u_block[10] = 15;

            let v_block = vec![0i16; V_SIZE];

            let mut encoder = DctRansEncoder::new();
            encoder.encode_block(&y_block, &u_block, &v_block, Y_SIZE, U_SIZE, V_SIZE);
            let encoded = encoder.finish();

            let mut decoder = DctRansDecoder::new();
            decoder.consume(&encoded);
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            assert_eq!(y_block, decoded_y, "Y block mismatch (should be zero)");
            assert_eq!(u_block, decoded_u, "U block mismatch (Y and V zero)");
            assert_eq!(v_block, decoded_v, "V block mismatch (should be zero)");
        }

        // Test case 3: V has data, Y and U are all zeros
        {
            let y_block = vec![0i16; Y_SIZE];
            let u_block = vec![0i16; U_SIZE];

            let mut v_block = vec![0i16; V_SIZE];
            v_block[0] = -60;
            v_block[7] = 40;
            v_block[15] = -20;

            let mut encoder = DctRansEncoder::new();
            encoder.encode_block(&y_block, &u_block, &v_block, Y_SIZE, U_SIZE, V_SIZE);
            let encoded = encoder.finish();

            let mut decoder = DctRansDecoder::new();
            decoder.consume(&encoded);
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            assert_eq!(y_block, decoded_y, "Y block mismatch (should be zero)");
            assert_eq!(u_block, decoded_u, "U block mismatch (should be zero)");
            assert_eq!(v_block, decoded_v, "V block mismatch (Y and U zero)");
        }
    }

    #[test]
    fn test_rans_roundtrip_640x480_all_zero_except_last_block() {
        const STORAGE_W: usize = 640;
        const STORAGE_H: usize = 480;
        const BLOCKS_W: usize = STORAGE_W / 16;
        const BLOCKS_H: usize = STORAGE_H / 16;
        const BLOCKS_TOTAL: usize = BLOCKS_W * BLOCKS_H;

        const Y_SIZE: usize = 256;
        const U_SIZE: usize = 64;
        const V_SIZE: usize = 64;

        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(54321);

        let mut all_y_blocks = Vec::new();
        let mut all_u_blocks = Vec::new();
        let mut all_v_blocks = Vec::new();

        for i in 0..BLOCKS_TOTAL {
            if i == BLOCKS_TOTAL - 1 {
                let mut y_block = vec![0i16; Y_SIZE];
                y_block[0] = rng.gen_range(-200..=200);
                y_block[5] = rng.gen_range(-100..=100);
                y_block[15] = rng.gen_range(-50..=50);

                let mut u_block = vec![0i16; U_SIZE];
                u_block[0] = rng.gen_range(-150..=150);
                u_block[3] = rng.gen_range(-80..=80);

                let mut v_block = vec![0i16; V_SIZE];
                v_block[0] = rng.gen_range(-150..=150);
                v_block[7] = rng.gen_range(-80..=80);

                all_y_blocks.push(y_block);
                all_u_blocks.push(u_block);
                all_v_blocks.push(v_block);
            } else {
                all_y_blocks.push(vec![0i16; Y_SIZE]);
                all_u_blocks.push(vec![0i16; U_SIZE]);
                all_v_blocks.push(vec![0i16; V_SIZE]);
            }
        }

        let mut encoder = DctRansEncoder::new();
        for i in 0..BLOCKS_TOTAL {
            encoder.encode_block(
                &all_y_blocks[i],
                &all_u_blocks[i],
                &all_v_blocks[i],
                Y_SIZE,
                U_SIZE,
                V_SIZE,
            );
        }
        let encoded = encoder.finish();

        let mut decoder = DctRansDecoder::new();
        decoder.consume(&encoded);

        for i in 0..BLOCKS_TOTAL {
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            assert_eq!(all_y_blocks[i], decoded_y, "Y block {} mismatch", i);
            assert_eq!(all_u_blocks[i], decoded_u, "U block {} mismatch", i);
            assert_eq!(all_v_blocks[i], decoded_v, "V block {} mismatch", i);
        }
    }

    #[test]
    fn test_rans_roundtrip_640x480_plausible_residual() {
        const STORAGE_W: usize = 640;
        const STORAGE_H: usize = 480;
        const BLOCKS_W: usize = STORAGE_W / 16;
        const BLOCKS_H: usize = STORAGE_H / 16;
        const BLOCKS_TOTAL: usize = BLOCKS_W * BLOCKS_H;

        const Y_SIZE: usize = 256;
        const U_SIZE: usize = 64;
        const V_SIZE: usize = 64;

        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(99999);

        let mut all_y_blocks = Vec::new();
        let mut all_u_blocks = Vec::new();
        let mut all_v_blocks = Vec::new();

        let raw_size = BLOCKS_TOTAL * (Y_SIZE + U_SIZE + V_SIZE) * 2;

        for _ in 0..BLOCKS_TOTAL {
            let mut y_block = vec![0i16; Y_SIZE];
            if rng.gen_bool(0.8) {
                y_block[0] = rng.gen_range(-200..=200);
            }
            for i in 1..30.min(Y_SIZE) {
                let prob = 0.5 * (1.0 - (i as f64 / 30.0));
                if rng.gen_bool(prob) {
                    let max_val = (200.0 * (1.0 - i as f64 / 30.0)) as i16;
                    y_block[i] = rng.gen_range(-max_val..=max_val);
                }
            }
            for i in 30..Y_SIZE {
                if rng.gen_bool(0.05) {
                    y_block[i] = rng.gen_range(-20..=20);
                }
            }

            let mut u_block = vec![0i16; U_SIZE];
            if rng.gen_bool(0.7) {
                u_block[0] = rng.gen_range(-150..=150);
            }
            for i in 1..15.min(U_SIZE) {
                let prob = 0.4 * (1.0 - (i as f64 / 15.0));
                if rng.gen_bool(prob) {
                    let max_val = (150.0 * (1.0 - i as f64 / 15.0)) as i16;
                    u_block[i] = rng.gen_range(-max_val..=max_val);
                }
            }
            for i in 15..U_SIZE {
                if rng.gen_bool(0.05) {
                    u_block[i] = rng.gen_range(-15..=15);
                }
            }

            let mut v_block = vec![0i16; V_SIZE];
            if rng.gen_bool(0.7) {
                v_block[0] = rng.gen_range(-150..=150);
            }
            for i in 1..15.min(V_SIZE) {
                let prob = 0.4 * (1.0 - (i as f64 / 15.0));
                if rng.gen_bool(prob) {
                    let max_val = (150.0 * (1.0 - i as f64 / 15.0)) as i16;
                    v_block[i] = rng.gen_range(-max_val..=max_val);
                }
            }
            for i in 15..V_SIZE {
                if rng.gen_bool(0.05) {
                    v_block[i] = rng.gen_range(-15..=15);
                }
            }

            all_y_blocks.push(y_block);
            all_u_blocks.push(u_block);
            all_v_blocks.push(v_block);
        }

        let mut encoder = DctRansEncoder::new();
        for i in 0..BLOCKS_TOTAL {
            encoder.encode_block(
                &all_y_blocks[i],
                &all_u_blocks[i],
                &all_v_blocks[i],
                Y_SIZE,
                U_SIZE,
                V_SIZE,
            );
        }
        let encoded = encoder.finish();

        let encoded_size = encoded.len();
        let compression_ratio = raw_size as f64 / encoded_size as f64;

        println!("\n=== RANS Compression Statistics ===");
        println!(
            "Frame size: {}x{} ({} blocks)",
            STORAGE_W, STORAGE_H, BLOCKS_TOTAL
        );
        println!("Raw size: {} bytes", raw_size);
        println!("Encoded size: {} bytes", encoded_size);
        println!("Compression ratio: {:.2}:1", compression_ratio);
        println!(
            "Space savings: {:.1}%",
            (1.0 - encoded_size as f64 / raw_size as f64) * 100.0
        );

        let bytes_to_print = encoded_size.min(100);
        print!("First {} bytes (hex): ", bytes_to_print);
        for i in 0..bytes_to_print {
            print!("{:02x}", encoded[i]);
            if (i + 1) % 16 == 0 && i + 1 < bytes_to_print {
                print!("\n                                      ");
            } else if i + 1 < bytes_to_print {
                print!(" ");
            }
        }
        println!();

        let mut decoder = DctRansDecoder::new();
        decoder.consume(&encoded);

        for i in 0..BLOCKS_TOTAL {
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            assert_eq!(all_y_blocks[i], decoded_y, "Y block {} mismatch", i);
            assert_eq!(all_u_blocks[i], decoded_u, "U block {} mismatch", i);
            assert_eq!(all_v_blocks[i], decoded_v, "V block {} mismatch", i);
        }
    }
}
