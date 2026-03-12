use super::*;

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
            let subpel_x = Subpel::from_flag_bits(flags & 0x03);
            let subpel_y = Subpel::from_flag_bits((flags >> 2) & 0x03);
            let skip = (flags & 0x40) != 0;
            MvCodedBlock {
                mode,
                new_base: 0,
                delta_x: ddx,
                delta_y: ddy,
                subpel_x,
                subpel_y,
                skip,
            }
        })
        .collect()
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
#[should_panic(expected = "raw -0.5 subpixel is disallowed")]
fn test_mv_rans_raw_minus_half_panics() {
    // Encoding a block with raw -0.5 should panic (no candidates path)
    let mut encoder = MvRansEncoder::new();
    let mv_interleaved = vec![
        0u8, 0u8, 0x08, // qy = 2 (MinusHalf), qx = 0
    ];
    let blocks = blocks_from_interleaved(&mv_interleaved);
    let _compressed = encoder.encode_frame_and_get_data(&blocks, 1, 1);
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
            0x05, // Block 1: ddx=5, ddy=-5, flags=0x05 (raw -0.5 disallowed)
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

    // Test allowed half-pixel code combinations (codes 0..1). Raw -0.5 is disallowed.
    for qx in 0..2 {
        for qy in 0..2 {
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
        0x45, // ddx=0, ddy=0, both axes +0.5 + skip (raw -0.5 disallowed)
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
        let qx = (i % 2) as u8; // raw -0.5 disallowed
        let qy = ((i / 3) % 2) as u8; // raw -0.5 disallowed
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
            let qx = ((frame_idx + block_idx) % 2) as u8; // raw -0.5 disallowed
            let qy = ((frame_idx * 2 + block_idx) % 2) as u8; // raw -0.5 disallowed

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
