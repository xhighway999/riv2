use rand::Rng;
use reitero_dct::*;

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

    let original = generate_random_plane(width, height, stride);

    // Encode
    let encoded = encode_plane_8x8(&original, stride, width, height, quant_step, None);

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

    // Decode
    let mut decoded = vec![0i16; stride * height];
    decode_plane_8x8(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        quant_step,
        &skip_mask,
    );

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

    let original = generate_random_plane(width, height, stride);

    // Encode
    let encoded = encode_plane_16x16(&original, stride, width, height, quant_step, None);

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
    decode_plane_16x16(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        quant_step,
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
    let encoded = encode_plane_16x16(
        &original,
        stride,
        width,
        height,
        quant_step,
        Some(&input_skip_mask),
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
    decode_plane_16x16(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        quant_step,
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
    let encoded = encode_plane_8x8(
        &original,
        stride,
        width,
        height,
        quant_step,
        Some(&input_skip_mask),
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

    // Decode
    let mut decoded = vec![0i16; stride * height];
    decode_plane_8x8(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        quant_step,
        &skip_mask,
    );

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

    let original = generate_random_plane(width, height, stride);
    let encoded = encode_plane_8x8(&original, stride, width, height, quant_step, None);

    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        coeffs.extend(block.unwrap());
        skip_mask.push(false);
    }

    let mut decoded = vec![0i16; stride * height];
    decode_plane_8x8(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        quant_step,
        &skip_mask,
    );

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

    let original = generate_random_plane(width, height, stride);
    let encoded = encode_plane_16x16(&original, stride, width, height, quant_step, None);

    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        coeffs.extend(block.unwrap());
        skip_mask.push(false);
    }

    let mut decoded = vec![0i16; stride * height];
    decode_plane_16x16(
        &coeffs,
        &mut decoded,
        stride,
        width,
        height,
        quant_step,
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
    let encoded = encode_plane_8x8(plane, 8, 8, 8, quant_step, None);
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
    decode_plane_8x8(coeffs, &mut output, 8, 8, 8, quant_step, &skip_mask);

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
