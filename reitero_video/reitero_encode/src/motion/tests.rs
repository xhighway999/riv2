use super::{BLOCK_SIZE, LumaPlane, MotionVector, satd_scalar, scalar};
use reitero_video_common::decode_halfpel_i8;

#[cfg(feature = "simd")]
use super::satd_simd;
#[cfg(feature = "simd")]
use super::simd;

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

fn build_luma_plane(rgb: &[u8], width: usize, height: usize) -> LumaPlane {
    let y: Vec<u8> = rgb
        .chunks_exact(3)
        .map(|c| ((c[0] as u32 * 77 + c[1] as u32 * 150 + c[2] as u32 * 29) >> 8) as u8)
        .collect();
    LumaPlane::from_y_plane(&y, width, height)
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
        satd_scalar::sad_block_int_limit(&prev_luma, &curr_luma, width, height, x0, y0, dx, dy, i64::MAX)
    });
    let (simd_best, _) = best_offset(&candidates, |dx, dy| {
        satd_simd::sad_block_int_limit(&prev_luma, &curr_luma, width, height, x0, y0, dx, dy, i64::MAX)
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
    let prev_luma = build_luma_plane(&prev_rgb, width, height);
    let curr_luma = build_luma_plane(&curr_rgb, width, height);
    let blocks = ((width + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    let mut prev_mvs = vec![MotionVector::new(0, 0, 0, 0, false); blocks];
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
        assert_eq!(lhs, rhs, "motion vector mismatch at block {idx}");
    }
}
