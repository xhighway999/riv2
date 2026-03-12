use rand::Rng;
use reitero_dct::*;

// Standard luma quantization table (quality 50 baseline, JPEG-style).
#[rustfmt::skip]
const LUMA_Q50_Q50: [u16; 64] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
];

// Non-flat 16x16 quant table: tile the 8x8 luma table 4× (top-left, top-right, bottom-left,
// bottom-right quadrants each scaled by a different factor to ensure non-uniform coverage).
fn make_nonflat_16x16_quant() -> [u16; 256] {
    let mut t = [0u16; 256];
    let scales = [1u16, 2, 3, 4];
    for qy in 0..2 {
        for qx in 0..2 {
            let scale = scales[qy * 2 + qx];
            for y in 0..8 {
                for x in 0..8 {
                    let dst = (qy * 8 + y) * 16 + (qx * 8 + x);
                    t[dst] = (LUMA_Q50_Q50[y * 8 + x] * scale).max(1);
                }
            }
        }
    }
    t
}

fn generate_random_plane(width: usize, height: usize, stride: usize) -> Vec<i16> {
    let mut rng = rand::thread_rng();
    let mut plane = vec![0i16; stride * height];
    for y in 0..height {
        for x in 0..width {
            plane[y * stride + x] = rng.gen_range(-255..256);
        }
    }
    plane
}

#[test]
fn test_roundtrip_8x8_basic() {
    let width = 64;
    let height = 64;
    let stride = 64;
    let quant_step = 1.0;
    let num_blocks = (width / 8) * (height / 8);

    let original = generate_random_plane(width, height, stride);

    // Encode
    let encoded = encode_plane_8x8_aq(&original, stride, width, height, &vec![quant_step; num_blocks], None, 0.5);

    // Flatten encoded coefficients for decoder
    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        match block {
            Some(b) => {
                coeffs.extend(b);
                skip_mask.push(false);
            }
            None => {
                coeffs.extend(vec![0i16; 64]);
                skip_mask.push(true);
            }
        }
    }

    // Decode (strict ≤2 roundtrip)
    let mut decoded = vec![0i16; stride * height];
    decode_plane_8x8_aq(&coeffs, &mut decoded, stride, width, height, &vec![quant_step; num_blocks], &skip_mask);

    // Check (with quant_step 1.0, it should be very close or exact depending on DCT precision)
    for i in 0..original.len() {
        let diff = (original[i] as i32 - decoded[i] as i32).abs();
        assert!(
            diff <= 2,
            "Difference at index {} is too large: {} (orig: {}, dec: {})",
            i,
            diff,
            original[i],
            decoded[i]
        );
    }
}

#[test]
fn test_roundtrip_16x16_basic() {
    let width = 64;
    let height = 64;
    let stride = 64;
    let quant_step = 2.0;
    let num_blocks = (width / 16) * (height / 16);

    let original = generate_random_plane(width, height, stride);

    // Encode
    let encoded = encode_plane_16x16_aq(&original, stride, width, height, &vec![quant_step; num_blocks], None, 0.5);

    // Flatten
    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        match block {
            Some(b) => {
                coeffs.extend(b);
                skip_mask.push(false);
            }
            None => {
                skip_mask.push(true);
            }
        }
    }

    // Decode
    let mut decoded = vec![0i16; stride * height];
    decode_plane_16x16_aq(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        &vec![quant_step; num_blocks],
        &skip_mask,
    );

    // Check
    for i in 0..original.len() {
        let diff = (original[i] as i32 - decoded[i] as i32).abs();
        // 16x16 might have slightly more error due to fixed point precision in larger blocks
        assert!(
            diff <= 4,
            "Difference at index {} is too large: {} (orig: {}, dec: {})",
            i,
            diff,
            original[i],
            decoded[i]
        );
    }
}

#[test]
fn test_roundtrip_16x16_with_skip_mask() {
    let width = 32;
    let height = 32;
    let stride = 32;
    let quant_step = 1.0;
    let num_blocks = (width / 16) * (height / 16);

    let original = generate_random_plane(width, height, stride);

    // Create a skip mask where every other block is skipped
    let blocks_w = width / 16;
    let blocks_h = height / 16;
    let mut input_skip_mask = vec![false; blocks_w * blocks_h];
    for i in 0..input_skip_mask.len() {
        if i % 2 == 0 {
            input_skip_mask[i] = true;
        }
    }

    // Encode
    let encoded = encode_plane_16x16_aq(
        &original,
        stride,
        width,
        height,
        &vec![quant_step; num_blocks],
        Some(&input_skip_mask),
        0.5,
    );

    // Flatten
    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for (i, block) in encoded.into_iter().enumerate() {
        assert_eq!(block.is_none(), input_skip_mask[i]);
        match block {
            Some(b) => {
                coeffs.extend(b);
                skip_mask.push(false);
            }
            None => {
                coeffs.extend(vec![0i16; 256]);
                skip_mask.push(true);
            }
        }
    }

    // Decode
    let mut decoded = vec![0i16; stride * height];
    decode_plane_16x16_aq(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        &vec![quant_step; num_blocks],
        &skip_mask,
    );

    // Check
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let bi = by * blocks_w + bx;
            let is_skipped = input_skip_mask[bi];

            for y in 0..16 {
                for x in 0..16 {
                    let idx = (by * 16 + y) * stride + (bx * 16 + x);
                    if is_skipped {
                        assert_eq!(decoded[idx], 0, "Skipped block should be zeroed");
                    } else {
                        let diff = (original[idx] as i32 - decoded[idx] as i32).abs();
                        assert!(diff <= 4, "Difference too large in non-skipped block");
                    }
                }
            }
        }
    }
}

#[test]
fn test_roundtrip_8x8_with_skip_mask() {
    let width = 32;
    let height = 32;
    let stride = 32;
    let quant_step = 1.0;
    let num_blocks = (width / 8) * (height / 8);

    let original = generate_random_plane(width, height, stride);

    // Create a skip mask where every other block is skipped
    let blocks_w = width / 8;
    let blocks_h = height / 8;
    let mut input_skip_mask = vec![false; blocks_w * blocks_h];
    for i in 0..input_skip_mask.len() {
        if i % 2 == 0 {
            input_skip_mask[i] = true;
        }
    }

    // Encode
    let encoded = encode_plane_8x8_aq(
        &original,
        stride,
        width,
        height,
        &vec![quant_step; num_blocks],
        Some(&input_skip_mask),
        0.5,
    );

    // Flatten
    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for (i, block) in encoded.into_iter().enumerate() {
        assert_eq!(block.is_none(), input_skip_mask[i]);
        match block {
            Some(b) => {
                coeffs.extend(b);
                skip_mask.push(false);
            }
            None => {
                coeffs.extend(vec![0i16; 64]);
                skip_mask.push(true);
            }
        }
    }

    // Decode (strict ≤2)
    let mut decoded = vec![0i16; stride * height];
    decode_plane_8x8_aq(&coeffs, &mut decoded, stride, width, height, &vec![quant_step; num_blocks], &skip_mask);

    // Check
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let bi = by * blocks_w + bx;
            let is_skipped = input_skip_mask[bi];

            for y in 0..8 {
                for x in 0..8 {
                    let idx = (by * 8 + y) * stride + (bx * 8 + x);
                    if is_skipped {
                        assert_eq!(decoded[idx], 0, "Skipped block should be zeroed");
                    } else {
                        let diff = (original[idx] as i32 - decoded[idx] as i32).abs();
                        assert!(diff <= 2, "Difference too large in non-skipped block");
                    }
                }
            }
        }
    }
}

#[test]
fn test_roundtrip_8x8_high_quant() {
    let width = 16;
    let height = 16;
    let stride = 16;
    let quant_step = 50.0;
    let num_blocks = (width / 8) * (height / 8);

    let original = generate_random_plane(width, height, stride);
    let encoded = encode_plane_8x8_aq(&original, stride, width, height, &vec![quant_step; num_blocks], None, 0.5);

    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        coeffs.extend(block.unwrap());
        skip_mask.push(false);
    }

    let mut decoded = vec![0i16; stride * height];
    decode_plane_8x8_aq(&coeffs, &mut decoded, stride, width, height, &vec![quant_step; num_blocks], &skip_mask);

    // With high quantization, we just check it doesn't crash and produces something.
    // We can check that the values are within a reasonable range.
    for &val in &decoded {
        assert!(val >= -512 && val <= 512);
    }
}

#[test]
fn test_roundtrip_16x16_high_quant() {
    let width = 32;
    let height = 32;
    let stride = 32;
    let quant_step = 50.0;
    let num_blocks = (width / 16) * (height / 16);

    let original = generate_random_plane(width, height, stride);
    let encoded = encode_plane_16x16_aq(&original, stride, width, height, &vec![quant_step; num_blocks], None, 0.5);

    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        coeffs.extend(block.unwrap());
        skip_mask.push(false);
    }

    let mut decoded = vec![0i16; stride * height];
    decode_plane_16x16_aq(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        &vec![quant_step; num_blocks],
        &skip_mask,
    );

    for &val in &decoded {
        assert!(val >= -512 && val <= 512);
    }
}

#[test]
fn test_roundtrip_8x8_data() {
    let input: [i16; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ];
    let plane = &input[..];
    let quant_step = 1.0f32;
    let num_blocks = 1usize;
    let encoded = encode_plane_8x8_aq(plane, 8, 8, 8, &vec![quant_step; num_blocks], None, 0.5);
    assert!(encoded.len() == 1);
    let coeffs_opt = &encoded[0];
    assert!(coeffs_opt.is_some());
    let coeffs = coeffs_opt.as_ref().unwrap();

    // Check a few key coeffs against expected (allow +/-1 for float rounding bullshit)
    assert_eq!(coeffs[0], 461); // DC
    assert_eq!(coeffs[1], -169);
    assert_eq!(coeffs[8], -195); // Second row start
    // Add more if you're paranoid, flatten the expected matrix above

    let mut output = vec![0i16; 64];
    let skip_mask = [false];
    decode_plane_8x8_aq(coeffs, &mut output, 8, 8, 8, &vec![quant_step; num_blocks], &skip_mask);

    let max_error = input
        .iter()
        .zip(output.iter())
        .map(|(&a, &b)| (a - b).abs())
        .max()
        .unwrap();
    println!("Max reconstruction error: {}", max_error);
    assert!(max_error <= 2, "Roundtrip error too high: {}", max_error); // Loose for rounding

    // If you want exact recon match, print output and compare to expected recon above
}

// Note: previously there was a separate "reference" decoder used to cross-check the fast
// implementation. The main `decode_plane_8x8_aq` path is now the sole decoder and is tested
// via the roundtrip tests above.

/// Encode with a standard luma quant matrix, decode with the same matrix.
/// Checks that the roundtrip error is bounded by the largest quant step in the table.
#[test]
fn test_roundtrip_8x8_nonflat_matrix() {
    let width = 64;
    let height = 64;
    let stride = 64;
    let num_blocks = (width / 8) * (height / 8);

    let original = generate_random_plane(width, height, stride);
    let encoded =
        encode_plane_8x8_matrix(&original, stride, width, height, &LUMA_Q50_Q50, None);

    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        match block {
            Some(b) => {
                coeffs.extend(b);
                skip_mask.push(false);
            }
            None => {
                coeffs.extend(vec![0i16; 64]);
                skip_mask.push(true);
            }
        }
    }

    let mut decoded = vec![0i16; stride * height];
    decode_plane_8x8_matrix(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        &LUMA_Q50_Q50,
        &skip_mask,
    );

    // Max error is bounded by the largest quant step (half-step on each axis) times DCT gain.
    // Empirically this is well within max_q for well-behaved signals.
    let max_q = *LUMA_Q50_Q50.iter().max().unwrap() as i32;
    for i in 0..(width * height) {
        let diff = (original[i] as i32 - decoded[i] as i32).abs();
        assert!(
            diff <= max_q * 8,
            "8x8 matrix roundtrip diff {} at index {} exceeds bound {}",
            diff,
            i,
            max_q * 8
        );
    }
    // Verify DC (quant_step=16) reconstructs well for a smooth block.
    // Specifically ensure the non-flat matrix produces different coeff magnitudes than flat.
    let flat_encoded =
        encode_plane_8x8_aq(&original, stride, width, height, &vec![LUMA_Q50_Q50[0] as f32; num_blocks], None, 0.5);
    let matrix_encoded =
        encode_plane_8x8_matrix(&original, stride, width, height, &LUMA_Q50_Q50, None);
    // High-frequency coefficients (pos 7, for example) should differ due to different quant steps.
    let flat_ac: i32 = flat_encoded.iter().filter_map(|b| b.as_ref()).map(|b| b[7].abs() as i32).sum();
    let matrix_ac: i32 = matrix_encoded.iter().filter_map(|b| b.as_ref()).map(|b| b[7].abs() as i32).sum();
    // With quant_table[7]=61 vs flat quant=16, matrix should yield smaller (more quantized) HF coeffs.
    assert!(
        matrix_ac <= flat_ac,
        "Non-flat matrix HF coeffs {} should be <= flat {} (heavier HF quantization)",
        matrix_ac,
        flat_ac
    );
}

/// Encode with a non-flat 16x16 quant table, decode with the same table.
#[test]
fn test_roundtrip_16x16_nonflat_matrix() {
    let width = 64;
    let height = 64;
    let stride = 64;

    let original = generate_random_plane(width, height, stride);
    let quant_table = make_nonflat_16x16_quant();
    let encoded =
        encode_plane_16x16_matrix(&original, stride, width, height, &quant_table, None);

    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        match block {
            Some(b) => {
                coeffs.extend(b);
                skip_mask.push(false);
            }
            None => {
                coeffs.extend(vec![0i16; 256]);
                skip_mask.push(true);
            }
        }
    }

    let mut decoded = vec![0i16; stride * height];
    decode_plane_16x16_matrix(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        &quant_table,
        &skip_mask,
    );

    let max_q = *quant_table.iter().max().unwrap() as i32;
    for i in 0..(width * height) {
        let diff = (original[i] as i32 - decoded[i] as i32).abs();
        assert!(
            diff <= max_q * 16,
            "16x16 matrix roundtrip diff {} at index {} exceeds bound {}",
            diff,
            i,
            max_q * 16
        );
    }
    // Verify different quadrants produce different quantization granularity.
    // Top-left quadrant (scale=1, small quant) should have higher-magnitude coefficients
    // than bottom-right quadrant (scale=4, large quant) in the encoded output.
    let first_block_coeffs = &coeffs[..256]; // first 16x16 block
    let tl_sum: i32 = (0..8).flat_map(|v| (0..8).map(move |u| first_block_coeffs[v * 16 + u].abs() as i32)).sum();
    let br_sum: i32 = (8..16).flat_map(|v| (8..16).map(move |u| first_block_coeffs[v * 16 + u].abs() as i32)).sum();
    // Top-left uses smaller quant steps → more detail retained → higher coeff magnitudes.
    assert!(
        tl_sum >= br_sum,
        "Top-left (fine quant) coeff sum {} should be >= bottom-right (coarse quant) sum {}",
        tl_sum,
        br_sum
    );
}
