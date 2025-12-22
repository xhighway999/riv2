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

use cabac::{
    CabacReader, CabacWriter,
    rans32::{RansReader32, RansWriter32},
    vp8::VP8Context,
};

struct SharedBuffer(Rc<RefCell<Vec<u8>>>);

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.borrow_mut().flush()
    }
}

/// RANS encoder for DCT coefficients
///
/// This encoder accumulates encoded blocks and produces the final output
/// when `finish()` is called.
///
///

pub struct RansEncoder {
    output: Rc<RefCell<Vec<u8>>>,
    writer: RansWriter32<SharedBuffer>,
    // We separate Luma and Chroma contexts for better specialization
    luma_contexts: [VP8Context; 256],
    chroma_contexts: [VP8Context; 64],
    luma_mag: VP8Context,
    chroma_mag: VP8Context,
    eob_context: VP8Context,
}

impl RansEncoder {
    /// Create a new RANS encoder
    ///
    /// # Returns
    /// A new `RansEncoder` ready for encoding operations
    pub fn new() -> Self {
        // TODO: Initialize RANS encoding state]
        let buf = Rc::new(RefCell::new(Vec::new()));
        let writer_handle = SharedBuffer(Rc::clone(&buf));
        Self {
            output: buf,
            writer: RansWriter32::new(writer_handle),
            luma_contexts: [VP8Context::default(); 256],
            chroma_contexts: [VP8Context::default(); 64],
            luma_mag: VP8Context::default(),
            chroma_mag: VP8Context::default(),
            eob_context: VP8Context::default(),
        }
    }

    /// Encode a block of DCT coefficients
    ///
    /// This function encodes Y, U, and V coefficient arrays and accumulates
    /// the encoded data internally. It does not return anything.
    ///
    /// # Arguments
    /// * `y` - Y plane coefficients
    /// * `u` - U plane coefficients
    /// * `v` - V plane coefficients
    /// * `y_size` - Size of the Y block
    /// * `u_size` - Size of the U block
    /// * `v_size` - Size of the V block
    pub fn encode_block(
        &mut self,
        y: &[i16],
        u: &[i16],
        v: &[i16],
        y_size: usize,
        u_size: usize,
        v_size: usize,
    ) {
        self.encode_single_plane(y, y_size);
        self.encode_single_plane(u, u_size);
        self.encode_single_plane(v, v_size);
    }

    /// Encode a single plane of DCT coefficients
    ///
    /// This function encodes a single plane (Y, U, or V) and accumulates
    /// the encoded data internally. It does not return anything.
    ///
    /// # Arguments
    /// * `plane` - Plane coefficients (Y, U, or V)
    /// * `size` - Size of the plane block
    fn encode_single_plane(&mut self, plane: &[i16], size: usize) {
        // 1. Find the last non-zero coefficient to determine the EOB (End of Block)
        // If the block is all zeros, last_nz will be 0.
        let last_nz = plane.iter().rposition(|&x| x != 0).unwrap_or(0);

        // 2. Encode the EOB index
        // This tells the decoder exactly when to stop processing this block.
        self.writer
            .put_unary_encoded(last_nz, &mut [self.eob_context])
            .expect("RANS writer failure on EOB");

        let is_luma = size == 256;

        // Resolve type mismatch by coercing arrays to slices
        let contexts: &mut [VP8Context] = if is_luma {
            &mut self.luma_contexts
        } else {
            &mut self.chroma_contexts
        };

        let mag_context: &mut VP8Context = if is_luma {
            &mut self.luma_mag
        } else {
            &mut self.chroma_mag
        };

        // 3. Encode coefficients up to the last non-zero
        for i in 0..=last_nz {
            let val = plane[i];

            // Significance bit: "Is there a non-zero value at this position?"
            self.writer
                .put(val != 0, &mut contexts[i])
                .expect("RANS writer failure on significance bit");

            if val != 0 {
                // Sign bit: We use put_bypass because signs are statistically 50/50
                self.writer
                    .put_bypass(val < 0)
                    .expect("RANS writer failure on sign bit");

                // Magnitude: Encoder stores (abs(val) - 1)
                // because we already know the value is at least 1.
                let abs_val = (val.abs() as usize) - 1;
                self.writer
                    .put_unary_encoded(abs_val, &mut [*mag_context])
                    .expect("RANS writer failure on magnitude");
            }
        }
    }
    /// Finish encoding and return the encoded bytes
    ///
    /// This function finalizes the encoding process and returns all
    /// accumulated encoded data as a byte vector.
    ///
    /// # Returns
    /// Vector of encoded bytes representing all encoded blocks
    pub fn finish(&mut self) -> Vec<u8> {
        // TODO: Implement RANS finalization
        // This should finalize the encoding and return all accumulated bytes
        self.writer.finish().unwrap();

        self.output.borrow_mut().clone()
    }
}

/// RANS decoder for DCT coefficients
///
/// This decoder consumes encoded bytes incrementally and can decode
/// blocks on demand.
pub struct RansDecoder {
    reader: Option<RansReader32<Cursor<Vec<u8>>>>,
    luma_contexts: [VP8Context; 256],
    chroma_contexts: [VP8Context; 64],
    luma_mag: VP8Context,
    chroma_mag: VP8Context,
    eob_context: VP8Context,
}

impl RansDecoder {
    /// Create a new RANS decoder
    ///
    /// # Returns
    /// A new `RansDecoder` ready for decoding operations
    pub fn new() -> Self {
        // TODO: Initialize RANS decoding state
        Self {
            reader: None, // We don't have the bitstream yet!
            luma_contexts: [VP8Context::default(); 256],
            chroma_contexts: [VP8Context::default(); 64],
            luma_mag: VP8Context::default(),
            chroma_mag: VP8Context::default(),
            eob_context: VP8Context::default(),
        }
    }

    /// Consume encoded bytes
    ///
    /// This function feeds encoded bytes into the decoder's internal buffer.
    /// The bytes are consumed and stored for later decoding operations.
    ///
    /// # Arguments
    /// * `bytes` - Slice of encoded bytes to consume
    pub fn consume(&mut self, bytes: &[u8]) {
        // TODO: Implement RANS byte consumption
        // This should store the bytes in the decoder's internal buffer
        let cursor = Cursor::new(bytes.to_vec());
        self.reader = Some(RansReader32::new(cursor).expect("RANS Init Failed"));
    }

    /// Decode a block of DCT coefficients
    ///
    /// This function decodes the next block from the consumed bytes and
    /// returns the Y, U, and V coefficient arrays.
    ///
    /// # Arguments
    /// * `y_size` - Expected size of the Y block
    /// * `u_size` - Expected size of the U block
    /// * `v_size` - Expected size of the V block
    ///
    /// # Returns
    /// Tuple of (Y coefficients, U coefficients, V coefficients), or error if decoding fails
    pub fn decode_block(
        &mut self,
        y_size: usize,
        u_size: usize,
        v_size: usize,
    ) -> (Vec<i16>, Vec<i16>, Vec<i16>) {
        let y = self.decode_single_plane(y_size);
        let u = self.decode_single_plane(u_size);
        let v = self.decode_single_plane(v_size);
        (y, u, v)
    }

    /// Decode a single plane of DCT coefficients
    ///
    /// This function decodes the next plane (Y, U, or V) from the consumed bytes
    /// and returns the coefficient array.
    ///
    /// # Arguments
    /// * `size` - Expected size of the plane block
    ///
    /// # Returns
    /// Vector of plane coefficients, or error if decoding fails
    /// Decodes a single plane (Y, U, or V) using expect for error handling
    fn decode_single_plane(&mut self, size: usize) -> Vec<i16> {
        let mut block = vec![0i16; size];

        // Get the reader or crash with a helpful message
        let reader = self
            .reader
            .as_mut()
            .expect("Decoder called before consume()");

        // 1. Decode EOB index (Last Non-Zero)
        let end_idx = reader
            .get_unary_encoded(&mut [self.eob_context])
            .expect("Bitstream ended prematurely while reading EOB");

        // Safety check for corrupt bitstreams
        assert!(
            end_idx < size,
            "Decoded EOB index {} exceeds block size {}",
            end_idx,
            size
        );

        let is_luma = size == 256;

        // Coerce arrays to slices to resolve the type mismatch
        let contexts: &mut [VP8Context] = if is_luma {
            &mut self.luma_contexts
        } else {
            &mut self.chroma_contexts
        };

        let mag_context: &mut VP8Context = if is_luma {
            &mut self.luma_mag
        } else {
            &mut self.chroma_mag
        };

        for i in 0..=end_idx {
            // 2. Decode Significance Bit
            let is_nz = reader
                .get(&mut contexts[i])
                .expect("Failed to read significance bit");

            if is_nz {
                // 3. Decode Sign (Bypass)
                let is_negative = reader.get_bypass().expect("Failed to read sign bit");

                // 4. Decode Magnitude
                // The encoder used (val.abs() - 1), so we add 1 back.
                let abs_val = (reader
                    .get_unary_encoded(&mut [*mag_context])
                    .expect("Failed to read magnitude")
                    + 1) as i16;

                block[i] = if is_negative { -abs_val } else { abs_val };
            }
        }

        block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rans_roundtrip_single_block() {
        // Create test data: Y (256), U (64), V (64) blocks with at least one non-zero value
        const Y_SIZE: usize = 256;
        const U_SIZE: usize = 64;
        const V_SIZE: usize = 64;

        // Create Y block with some non-zero values
        let mut y_block = vec![0i16; Y_SIZE];
        y_block[0] = 42; // DC coefficient
        y_block[1] = -15; // AC coefficient
        y_block[10] = 7; // Another AC coefficient

        // Create U block with some non-zero values
        let mut u_block = vec![0i16; U_SIZE];
        u_block[0] = 20;
        u_block[5] = -8;

        // Create V block with some non-zero values
        let mut v_block = vec![0i16; V_SIZE];
        v_block[0] = -12;
        v_block[3] = 5;

        // Encode
        let mut encoder = RansEncoder::new();
        encoder.encode_block(&y_block, &u_block, &v_block, Y_SIZE, U_SIZE, V_SIZE);
        let encoded = encoder.finish();

        // Decode
        let mut decoder = RansDecoder::new();
        decoder.consume(&encoded);
        let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

        // Verify roundtrip
        assert_eq!(y_block, decoded_y, "Y block mismatch");
        assert_eq!(u_block, decoded_u, "U block mismatch");
        assert_eq!(v_block, decoded_v, "V block mismatch");
    }

    #[test]
    fn test_rans_roundtrip_640x480_residual() {
        // 640x480 frame: 40x30 blocks of 16x16
        const STORAGE_W: usize = 640;
        const STORAGE_H: usize = 480;
        const BLOCKS_W: usize = STORAGE_W / 16; // 40
        const BLOCKS_H: usize = STORAGE_H / 16; // 30
        const BLOCKS_TOTAL: usize = BLOCKS_W * BLOCKS_H; // 1200

        const Y_SIZE: usize = 256; // 16x16
        const U_SIZE: usize = 64; // 8x8
        const V_SIZE: usize = 64; // 8x8

        // Generate quantized DCT coefficients for all blocks
        // Simulate what encode_dct_quantize_zigzag_block would produce
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(12345);

        let mut all_y_blocks = Vec::new();
        let mut all_u_blocks = Vec::new();
        let mut all_v_blocks = Vec::new();

        // Generate blocks with at least some non-zero values
        let mut non_zero_blocks = 0;
        for _ in 0..BLOCKS_TOTAL {
            // Y block: simulate quantized DCT coefficients
            let mut y_block = vec![0i16; Y_SIZE];
            // DC coefficient is often non-zero
            if rng.gen_bool(0.8) {
                y_block[0] = rng.gen_range(-200..=200);
            }
            // Some AC coefficients
            for i in 1..Y_SIZE {
                if rng.gen_bool(0.1) {
                    y_block[i] = rng.gen_range(-100..=100);
                }
            }

            // U block
            let mut u_block = vec![0i16; U_SIZE];
            if rng.gen_bool(0.7) {
                u_block[0] = rng.gen_range(-150..=150);
            }
            for i in 1..U_SIZE {
                if rng.gen_bool(0.15) {
                    u_block[i] = rng.gen_range(-80..=80);
                }
            }

            // V block
            let mut v_block = vec![0i16; V_SIZE];
            if rng.gen_bool(0.7) {
                v_block[0] = rng.gen_range(-150..=150);
            }
            for i in 1..V_SIZE {
                if rng.gen_bool(0.15) {
                    v_block[i] = rng.gen_range(-80..=80);
                }
            }

            // Ensure at least one block has non-zero values
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

        // Ensure we have at least some non-zero blocks
        assert!(
            non_zero_blocks > 0,
            "Test requires at least one block with non-zero values"
        );

        // Encode all blocks
        let mut encoder = RansEncoder::new();
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

        // Decode all blocks
        let mut decoder = RansDecoder::new();
        decoder.consume(&encoded);

        for i in 0..BLOCKS_TOTAL {
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            // Verify roundtrip
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

            let mut encoder = RansEncoder::new();
            encoder.encode_block(&y_block, &u_block, &v_block, Y_SIZE, U_SIZE, V_SIZE);
            let encoded = encoder.finish();

            let mut decoder = RansDecoder::new();
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

            let mut encoder = RansEncoder::new();
            encoder.encode_block(&y_block, &u_block, &v_block, Y_SIZE, U_SIZE, V_SIZE);
            let encoded = encoder.finish();

            let mut decoder = RansDecoder::new();
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

            let mut encoder = RansEncoder::new();
            encoder.encode_block(&y_block, &u_block, &v_block, Y_SIZE, U_SIZE, V_SIZE);
            let encoded = encoder.finish();

            let mut decoder = RansDecoder::new();
            decoder.consume(&encoded);
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            assert_eq!(y_block, decoded_y, "Y block mismatch (should be zero)");
            assert_eq!(u_block, decoded_u, "U block mismatch (should be zero)");
            assert_eq!(v_block, decoded_v, "V block mismatch (Y and U zero)");
        }
    }

    #[test]
    fn test_rans_roundtrip_640x480_all_zero_except_last_block() {
        // 640x480 frame: 40x30 blocks of 16x16
        const STORAGE_W: usize = 640;
        const STORAGE_H: usize = 480;
        const BLOCKS_W: usize = STORAGE_W / 16; // 40
        const BLOCKS_H: usize = STORAGE_H / 16; // 30
        const BLOCKS_TOTAL: usize = BLOCKS_W * BLOCKS_H; // 1200

        const Y_SIZE: usize = 256; // 16x16
        const U_SIZE: usize = 64; // 8x8
        const V_SIZE: usize = 64; // 8x8

        // Generate all zero blocks except the last one
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(54321);

        let mut all_y_blocks = Vec::new();
        let mut all_u_blocks = Vec::new();
        let mut all_v_blocks = Vec::new();

        // All blocks except the last are zero
        for i in 0..BLOCKS_TOTAL {
            if i == BLOCKS_TOTAL - 1 {
                // Last block: generate random non-zero data
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
                // All other blocks are zero
                all_y_blocks.push(vec![0i16; Y_SIZE]);
                all_u_blocks.push(vec![0i16; U_SIZE]);
                all_v_blocks.push(vec![0i16; V_SIZE]);
            }
        }

        // Encode all blocks
        let mut encoder = RansEncoder::new();
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

        // Decode all blocks
        let mut decoder = RansDecoder::new();
        decoder.consume(&encoded);

        for i in 0..BLOCKS_TOTAL {
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            // Verify roundtrip
            assert_eq!(all_y_blocks[i], decoded_y, "Y block {} mismatch", i);
            assert_eq!(all_u_blocks[i], decoded_u, "U block {} mismatch", i);
            assert_eq!(all_v_blocks[i], decoded_v, "V block {} mismatch", i);
        }
    }

    #[test]
    fn test_rans_roundtrip_640x480_plausible_residual() {
        // 640x480 frame: 40x30 blocks of 16x16
        const STORAGE_W: usize = 640;
        const STORAGE_H: usize = 480;
        const BLOCKS_W: usize = STORAGE_W / 16; // 40
        const BLOCKS_H: usize = STORAGE_H / 16; // 30
        const BLOCKS_TOTAL: usize = BLOCKS_W * BLOCKS_H; // 1200

        const Y_SIZE: usize = 256; // 16x16
        const U_SIZE: usize = 64; // 8x8
        const V_SIZE: usize = 64; // 8x8

        // Generate plausible residual data: coefficients taper off from DC to high frequencies
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(99999);

        let mut all_y_blocks = Vec::new();
        let mut all_u_blocks = Vec::new();
        let mut all_v_blocks = Vec::new();

        // Calculate raw size (uncompressed)
        let raw_size = BLOCKS_TOTAL * (Y_SIZE + U_SIZE + V_SIZE) * 2; // i16 = 2 bytes

        for _ in 0..BLOCKS_TOTAL {
            // Y block: DC coefficient is often non-zero, AC coefficients taper off
            let mut y_block = vec![0i16; Y_SIZE];
            // DC coefficient (index 0) - 80% chance of being non-zero
            if rng.gen_bool(0.8) {
                y_block[0] = rng.gen_range(-200..=200);
            }
            // Early AC coefficients (indices 1-30) - probability decreases with index
            for i in 1..30.min(Y_SIZE) {
                let prob = 0.5 * (1.0 - (i as f64 / 30.0)); // Taper from 50% to 0%
                if rng.gen_bool(prob) {
                    let max_val = (200.0 * (1.0 - i as f64 / 30.0)) as i16;
                    y_block[i] = rng.gen_range(-max_val..=max_val);
                }
            }
            // Later coefficients (indices 30+) are rarely non-zero
            for i in 30..Y_SIZE {
                if rng.gen_bool(0.05) {
                    y_block[i] = rng.gen_range(-20..=20);
                }
            }

            // U block: similar pattern but smaller range
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

            // V block: similar pattern
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

        // Encode all blocks
        let mut encoder = RansEncoder::new();
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

        // Calculate compression ratio
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

        // Print first 100 bytes of compressed data
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

        // Decode all blocks
        let mut decoder = RansDecoder::new();
        decoder.consume(&encoded);

        for i in 0..BLOCKS_TOTAL {
            let (decoded_y, decoded_u, decoded_v) = decoder.decode_block(Y_SIZE, U_SIZE, V_SIZE);

            // Verify roundtrip
            assert_eq!(all_y_blocks[i], decoded_y, "Y block {} mismatch", i);
            assert_eq!(all_u_blocks[i], decoded_u, "U block {} mismatch", i);
            assert_eq!(all_v_blocks[i], decoded_v, "V block {} mismatch", i);
        }
    }
}
