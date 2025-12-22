use super::{BLOCK_SIZE, LumaPlane, MotionLuma, MotionVector, satd_scalar, scalar};
use reitero_video_common::decode_halfpel_i8;

#[cfg(feature = "simd")]
use super::satd_simd;
#[cfg(feature = "simd")]
use super::simd;

#[cfg(not(feature = "simd"))]
use super::backend;

fn solid_frame(width: usize, height: usize, color: [u8; 3]) -> Vec<u8> {
    let mut frame = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            frame[idx] = color[0];
            frame[idx + 1] = color[1];
            frame[idx + 2] = color[2];
        }
    }
    frame
}

fn gradient_frame(width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            frame[idx] = (x as u8).wrapping_mul(3);
            frame[idx + 1] = (y as u8).wrapping_mul(5);
            frame[idx + 2] = ((x + y) as u8).wrapping_mul(7);
        }
    }
    frame
}

fn expected_luma(r: u8, g: u8, b: u8) -> u16 {
    (r as u32 * 77 + g as u32 * 150 + b as u32 * 29) as u16
}

fn plane_to_fixed(samples: &[MotionLuma]) -> Vec<u16> {
    samples
        .iter()
        .map(|&sample| sample.as_fixed_point())
        .collect()
}

fn build_luma_plane(rgb: &[u8], width: usize, height: usize) -> LumaPlane {
    satd_scalar::rgb_to_luma_plane(rgb, width, height)
}

#[test]
fn satd_scalar_zero_on_identical_blocks() {
    let width = BLOCK_SIZE;
    let height = BLOCK_SIZE;
    let frame = solid_frame(width, height, [64, 128, 255]);
    let luma = build_luma_plane(&frame, width, height);
    let satd = satd_scalar::satd_block_int(&luma, &luma, width, height, 0, 0, 0, 0);
    assert_eq!(satd, 0);
}

#[test]
fn satd_scalar_detects_integer_shift() {
    let width = BLOCK_SIZE;
    let height = BLOCK_SIZE;
    let frame = gradient_frame(width * 2, height * 2);
    let luma = build_luma_plane(&frame, width * 2, height * 2);
    // Evaluate block at (0,0) but reference data shifted by 2px in both axes.
    let satd_same = satd_scalar::satd_block_int(&luma, &luma, width * 2, height * 2, 0, 0, 0, 0);
    let satd_shifted = satd_scalar::satd_block_int(&luma, &luma, width * 2, height * 2, 0, 0, 2, 2);
    assert_eq!(satd_same, 0);
    assert!(satd_shifted > 0);
}

#[test]
fn satd_scalar_respects_limit() {
    let width = BLOCK_SIZE;
    let height = BLOCK_SIZE;
    let prev = gradient_frame(width, height);
    let mut curr = prev.clone();
    // Introduce a difference so SATD is non-zero.
    for pixel in curr.chunks_mut(3).take(8) {
        pixel[0] = pixel[0].wrapping_add(50);
    }
    let prev_luma = build_luma_plane(&prev, width, height);
    let curr_luma = build_luma_plane(&curr, width, height);
    let full = satd_scalar::satd_block_int(&prev_luma, &curr_luma, width, height, 0, 0, 0, 0);
    assert!(full > 0);
    let limit = full / 4;
    let limited =
        satd_scalar::satd_block_int_limit(&prev_luma, &curr_luma, width, height, 0, 0, 0, 0, limit);
    assert!(limited >= limit);
    assert!(limited <= full);
}

#[cfg(not(feature = "simd"))]
#[test]
fn backend_satd_block_limit_matches_scalar() {
    let width = BLOCK_SIZE * 2;
    let height = BLOCK_SIZE * 2;
    let prev = gradient_frame(width, height);
    let mut curr = prev.clone();
    for (i, pixel) in curr.chunks_mut(3).enumerate() {
        if i % 4 == 0 {
            pixel[0] = pixel[0].wrapping_add(21);
        }
        if i % 5 == 0 {
            pixel[2] = pixel[2].wrapping_sub(13);
        }
    }
    let prev_luma = build_luma_plane(&prev, width, height);
    let curr_luma = build_luma_plane(&curr, width, height);
    let x0 = (BLOCK_SIZE / 4) as i32;
    let y0 = (BLOCK_SIZE / 5) as i32;
    let dx = 3;
    let dy = -2;

    let full = satd_scalar::satd_block_int(&prev_luma, &curr_luma, width, height, x0, y0, dx, dy);
    let limit = (full / 3).max(1);
    let scalar_limited = satd_scalar::satd_block_int_limit(
        &prev_luma, &curr_luma, width, height, x0, y0, dx, dy, limit,
    );
    let backend_limited =
        backend::satd_block_int_limit(&prev_luma, &curr_luma, width, height, x0, y0, dx, dy, limit);
    assert_eq!(scalar_limited, backend_limited);
    assert!(backend_limited >= limit);
    assert!(backend_limited <= full);
}

#[test]
fn satd_scalar_handles_out_of_bounds_shift() {
    let width = BLOCK_SIZE;
    let height = BLOCK_SIZE;
    let frame = solid_frame(width, height, [200, 10, 90]);
    let luma = build_luma_plane(&frame, width, height);
    // Large dx/dy should clamp to the frame edge and still match identical content.
    let satd = satd_scalar::satd_block_int(&luma, &luma, width, height, 0, 0, 64, 64);
    assert_eq!(satd, 0);
}

#[cfg(not(feature = "simd"))]
#[test]
fn backend_satd_block_matches_scalar() {
    let width = BLOCK_SIZE * 2;
    let height = BLOCK_SIZE * 2;
    let prev = gradient_frame(width, height);
    let mut curr = prev.clone();
    for (i, pixel) in curr.chunks_mut(3).enumerate() {
        if i % 3 == 0 {
            pixel[0] = pixel[0].wrapping_add(5);
        } else if i % 3 == 1 {
            pixel[1] = pixel[1].wrapping_sub(7);
        } else {
            pixel[2] = pixel[2].wrapping_add(11);
        }
    }

    let prev_luma = build_luma_plane(&prev, width, height);
    let curr_luma = build_luma_plane(&curr, width, height);
    let positions = [(0, 0), ((BLOCK_SIZE / 2) as i32, (BLOCK_SIZE / 3) as i32)];
    let offsets = [(-2, -1), (0, 0), (1, 2), (3, -2)];

    for &(x0, y0) in &positions {
        for &(dx, dy) in &offsets {
            let scalar =
                satd_scalar::satd_block_int(&prev_luma, &curr_luma, width, height, x0, y0, dx, dy);
            let backend_val =
                backend::satd_block_int(&prev_luma, &curr_luma, width, height, x0, y0, dx, dy);
            assert_eq!(scalar, backend_val, "x0={x0}, y0={y0}, dx={dx}, dy={dy}");
        }
    }
}

#[test]
fn rgb_to_luma_plane_scalar_matches_reference() {
    let width = 4;
    let height = 2;
    let rgb = vec![
        0, 0, 0, // 0
        255, 0, 0, // R
        0, 255, 0, // G
        0, 0, 255, // B
        50, 100, 150, 200, 150, 100, 12, 34, 56, 250, 250, 250,
    ];
    let plane = satd_scalar::rgb_to_luma_plane(&rgb, width, height);
    let expected: Vec<u16> = rgb
        .chunks_exact(3)
        .map(|c| expected_luma(c[0], c[1], c[2]))
        .collect();
    assert_eq!(plane_to_fixed(&plane.as_slice()), expected);
    assert_eq!(plane.as_slice().len(), width * height);
}

#[test]
fn rgb_to_luma_plane_scalar_gradient() {
    let width = BLOCK_SIZE * 2;
    let height = BLOCK_SIZE;
    let rgb = gradient_frame(width, height);
    let plane = satd_scalar::rgb_to_luma_plane(&rgb, width, height);
    assert_eq!(plane.as_slice().len(), width * height);
    // Ensure monotonicity along x axis for first row of gradient helper.
    for pair in plane
        .as_slice()
        .chunks_exact(width)
        .next()
        .unwrap()
        .windows(2)
    {
        assert!(pair[0] <= pair[1]);
    }
}

#[test]
fn rgb_to_luma_plane_scalar_handles_tail_pixels() {
    let width = 7;
    let height = 5;
    let rgb = gradient_frame(width, height);
    let plane = satd_scalar::rgb_to_luma_plane(&rgb, width, height);
    assert_eq!(plane.as_slice().len(), width * height);

    let expected: Vec<u16> = rgb
        .chunks_exact(3)
        .map(|c| expected_luma(c[0], c[1], c[2]))
        .collect();
    assert_eq!(plane_to_fixed(&plane.as_slice()), expected);
    // Spot check last pixel to make sure the tail path was exercised.
    assert_eq!(plane_to_fixed(&plane.as_slice()).last(), expected.last());
}

#[test]
fn sad_block_halfpel_scalar_detects_half_pixel_shift() {
    let width = BLOCK_SIZE * 2;
    let height = BLOCK_SIZE * 2;
    let frame = gradient_frame(width, height);
    let luma = build_luma_plane(&frame, width, height);
    let x0 = (BLOCK_SIZE / 2) as i32;
    let y0 = (BLOCK_SIZE / 2) as i32;

    let baseline = satd_scalar::sad_block_halfpel_limit_luma(
        &luma,
        &luma,
        width,
        height,
        x0,
        y0,
        0,
        0,
        i64::MAX,
    );
    assert_eq!(baseline, 0);

    let half_shift = satd_scalar::sad_block_halfpel_limit_luma(
        &luma,
        &luma,
        width,
        height,
        x0,
        y0,
        1,
        0,
        i64::MAX,
    );
    assert!(half_shift > 0);
}

#[test]
fn sad_block_halfpel_scalar_respects_limit() {
    let width = BLOCK_SIZE * 2;
    let height = BLOCK_SIZE * 2;
    let prev = gradient_frame(width, height);
    let mut curr = prev.clone();
    for (i, pixel) in curr.chunks_mut(3).enumerate().take(48) {
        if i % 2 == 0 {
            pixel[0] = pixel[0].wrapping_add(17);
        } else {
            pixel[2] = pixel[2].wrapping_add(23);
        }
    }

    let prev_luma = build_luma_plane(&prev, width, height);
    let curr_luma = build_luma_plane(&curr, width, height);
    let x0 = (BLOCK_SIZE / 2) as i32;
    let y0 = (BLOCK_SIZE / 3) as i32;
    let full = satd_scalar::sad_block_halfpel_limit_luma(
        &prev_luma,
        &curr_luma,
        width,
        height,
        x0,
        y0,
        1,
        1,
        i64::MAX,
    );
    assert!(full > 0);
    let limit = (full / 4).max(1);
    let limited = satd_scalar::sad_block_halfpel_limit_luma(
        &prev_luma, &curr_luma, width, height, x0, y0, 1, 1, limit,
    );
    assert!(limited >= limit);
    assert!(limited <= full);
}

#[cfg(feature = "simd")]
fn best_offset<F>(candidates: &[(i32, i32)], mut eval: F) -> ((i32, i32), i64)
where
    F: FnMut(i32, i32) -> i64,
{
    assert!(!candidates.is_empty());
    let mut best = candidates[0];
    let mut best_score = eval(best.0, best.1);
    for &(dx, dy) in &candidates[1..] {
        let score = eval(dx, dy);
        if score < best_score {
            best = (dx, dy);
            best_score = score;
        }
    }
    (best, best_score)
}

#[cfg(feature = "simd")]
#[test]
fn satd_simd_and_scalar_pick_same_best_offset() {
    let width = BLOCK_SIZE * 4;
    let height = BLOCK_SIZE * 4;
    let prev = gradient_frame(width, height);
    let curr = prev.clone();
    let prev_luma = build_luma_plane(&prev, width, height);
    let curr_luma = build_luma_plane(&curr, width, height);
    let x0 = BLOCK_SIZE as i32;
    let y0 = BLOCK_SIZE as i32;
    let candidates = [(-4, -2), (-2, -1), (-1, 0), (0, 0), (1, 1), (3, -1)];

    let (scalar_best, _) = best_offset(&candidates, |dx, dy| {
        satd_scalar::satd_block_int(&prev_luma, &curr_luma, width, height, x0, y0, dx, dy)
    });
    let (simd_best, _) = best_offset(&candidates, |dx, dy| {
        satd_simd::satd_block_int(&prev_luma, &curr_luma, width, height, x0, y0, dx, dy)
    });

    assert_eq!(scalar_best, simd_best);
}

#[cfg(feature = "simd")]
#[test]
fn sad_halfpel_simd_and_scalar_pick_same_best_offset() {
    let width = BLOCK_SIZE * 4;
    let height = BLOCK_SIZE * 3;
    let prev = gradient_frame(width, height);
    let curr = prev.clone();
    let prev_luma = build_luma_plane(&prev, width, height);
    let curr_luma = build_luma_plane(&curr, width, height);
    let x0 = BLOCK_SIZE as i32;
    let y0 = BLOCK_SIZE as i32;
    let candidates = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 2), (3, 1), (4, 2)];

    let (scalar_best, _) = best_offset(&candidates, |dx_hp, dy_hp| {
        satd_scalar::sad_block_halfpel_limit_luma(
            &prev_luma,
            &curr_luma,
            width,
            height,
            x0,
            y0,
            dx_hp,
            dy_hp,
            i64::MAX,
        )
    });

    let (simd_best, _) = best_offset(&candidates, |dx_hp, dy_hp| {
        satd_simd::sad_block_halfpel_limit_luma(
            &prev_luma,
            &curr_luma,
            width,
            height,
            x0,
            y0,
            dx_hp,
            dy_hp,
            i64::MAX,
        )
    });

    assert_eq!(scalar_best, simd_best);
}

#[cfg(feature = "simd")]
fn synthetic_motion_pair(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    let prev = gradient_frame(width, height);
    let mut curr = prev.clone();
    for (i, pixel) in curr.chunks_mut(3).enumerate() {
        if i % 5 == 0 {
            pixel[0] = pixel[0].wrapping_add(13);
        }
        if i % 7 == 0 {
            pixel[1] = pixel[1].wrapping_sub(9);
        }
        if i % 11 == 0 {
            pixel[2] = pixel[2].wrapping_add(5);
        }
    }
    (prev, curr)
}

#[cfg(feature = "simd")]
#[test]
fn diamond_search_simd_matches_scalar_vectors_and_scores() {
    let width = BLOCK_SIZE * 4;
    let height = BLOCK_SIZE * 3;
    let (prev_rgb, curr_rgb) = synthetic_motion_pair(width, height);
    let prev_luma = LumaPlane::from_rgb(&prev_rgb, width, height);
    let curr_luma = LumaPlane::from_rgb(&curr_rgb, width, height);
    let blocks = ((width + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    let mut prev_mvs = vec![MotionVector::from_raw(0, 0, 0); blocks];
    for (idx, mv) in prev_mvs.iter_mut().enumerate() {
        let dx = ((idx as i32 % 5) - 2).clamp(-4, 4);
        let dy = ((((idx / 3) as i32) % 5) - 2).clamp(-4, 4);
        mv.set_dx(dx as i8);
        mv.set_dy(dy as i8);
        // Manually set flags via raw access for test setup if needed, or use new setters
        // Here we are setting arbitrary flags for testing
        let flags = (idx as u8) & 0x0F;
        mv.set_subpixel_x(decode_halfpel_i8(flags));
        mv.set_subpixel_y(decode_halfpel_i8(flags >> 2));
    }

    let search_range = 8u8;
    let prev_slice = prev_mvs.as_slice();

    let (scalar_mvs, scalar_scores) = scalar::hex_search_sad_with_scores_luma(
        width,
        height,
        search_range,
        Some(prev_slice),
        &prev_luma,
        &curr_luma,
        0,
        0,
        0.0,
    );
    let (simd_mvs, simd_scores) = simd::hex_search_sad_with_scores_luma(
        width,
        height,
        search_range,
        Some(prev_slice),
        &prev_luma,
        &curr_luma,
        0,
        0,
        0.0,
    );

    assert_eq!(scalar_scores, simd_scores);
    assert_eq!(scalar_mvs.len(), simd_mvs.len());
    for (idx, (lhs, rhs)) in scalar_mvs.iter().zip(simd_mvs.iter()).enumerate() {
        assert_eq!(lhs.dx(), rhs.dx(), "dx mismatch at block {idx}");
        assert_eq!(lhs.dy(), rhs.dy(), "dy mismatch at block {idx}");
        assert_eq!(
            lhs.raw_flags(),
            rhs.raw_flags(),
            "flags mismatch at block {idx}"
        );
    }
}

#[cfg(feature = "simd")]
#[test]
fn rgb_to_luma_plane_simd_matches_scalar_and_alignment() {
    let width = BLOCK_SIZE * 2;
    let height = BLOCK_SIZE * 2;
    let rgb = gradient_frame(width, height);
    let scalar = satd_scalar::rgb_to_luma_plane(&rgb, width, height);
    let simd = satd_simd::rgb_to_luma_plane(&rgb, width, height);
    assert_eq!(scalar.as_slice(), simd.as_slice());
    assert_eq!(simd.as_slice().len(), width * height);
    let align = std::mem::align_of::<MotionLuma>();
    assert_eq!((scalar.as_slice().as_ptr() as usize) % align, 0);
    assert_eq!((simd.as_slice().as_ptr() as usize) % align, 0);
}

#[cfg(feature = "simd")]
#[test]
fn rgb_to_luma_plane_simd_handles_tail_pixels() {
    let width = 13;
    let height = 3;
    let rgb = gradient_frame(width, height);
    let scalar = satd_scalar::rgb_to_luma_plane(&rgb, width, height);
    let simd = satd_simd::rgb_to_luma_plane(&rgb, width, height);
    assert_eq!(scalar.as_slice(), simd.as_slice());
    assert_eq!(scalar.as_slice().len(), width * height);
    assert_eq!(simd.as_slice().len(), width * height);
    let expected: Vec<u16> = rgb
        .chunks_exact(3)
        .map(|c| expected_luma(c[0], c[1], c[2]))
        .collect();
    assert_eq!(plane_to_fixed(&scalar.as_slice()), expected);
    assert_eq!(plane_to_fixed(&simd.as_slice()), expected);
}
