//! Placeholder SIMD backend.
//!
//! The SIMD version currently just forwards to the scalar SATD routines so we
//! can start wiring up the feature-gated module structure without changing
//! behavior.

use crate::BLOCK_SIZE;

use super::{LumaPlane, MotionLuma};
use wide::i32x8;

#[cfg(feature = "bench-luma")]
use std::hint::black_box;

#[inline(always)]
pub fn deinterleave_8_pixels_wide(rgb: &[u8]) -> (i32x8, i32x8, i32x8) {
    // We need 24 bytes for 8 pixels.
    // We load two 16-byte chunks (32 bytes total).
    let c: &[u8; 24] = rgb[..24].try_into().unwrap();

    // This is where the time is spent. The CPU picks these out one-by-one.
    let r = i32x8::new([
        c[0] as i32,
        c[3] as i32,
        c[6] as i32,
        c[9] as i32,
        c[12] as i32,
        c[15] as i32,
        c[18] as i32,
        c[21] as i32,
    ]);

    let g = i32x8::new([
        c[1] as i32,
        c[4] as i32,
        c[7] as i32,
        c[10] as i32,
        c[13] as i32,
        c[16] as i32,
        c[19] as i32,
        c[22] as i32,
    ]);

    let b = i32x8::new([
        c[2] as i32,
        c[5] as i32,
        c[8] as i32,
        c[11] as i32,
        c[14] as i32,
        c[17] as i32,
        c[20] as i32,
        c[23] as i32,
    ]);

    (r, g, b)
}

pub(super) fn rgb_to_luma_plane(rgb: &[u8], width: usize, height: usize) -> LumaPlane {
    let pixels = width * height;
    let mut luma = Vec::with_capacity(pixels);
    unsafe {
        luma.set_len(pixels);
    }

    let mut luma_ptr = luma.as_mut_ptr() as *mut u32; // Treat destination as u32 for SIMD
    let mut rgb_ptr = rgb.as_ptr();

    let mut luma_idx = 0usize;

    while luma_idx + 8 <= pixels {
        // 1. Efficient Load (still using your deinterleave logic)
        let chunk = unsafe { std::slice::from_raw_parts(rgb_ptr, 24) };
        let (r, g, b) = deinterleave_8_pixels_wide(chunk);

        // 2. Math
        let luminance = r * i32x8::splat(77) + g * i32x8::splat(150) + b * i32x8::splat(29);

        // 3. THE FIX: Direct Vector Store
        // We cast the pointer and write all 8 pixels (32 bytes) in one go
        unsafe {
            *((luma_ptr.add(luma_idx)) as *mut [i32; 8]) = luminance.to_array();
            rgb_ptr = rgb_ptr.add(24);
        }

        luma_idx += 8;
    }

    // Scalar fallback for remaining pixels
    for i in luma_idx..pixels {
        let base = i * 3;
        let r = rgb[base] as u32;
        let g = rgb[base + 1] as u32;
        let b = rgb[base + 2] as u32;
        luma[i] = MotionLuma::from_fixed_point((r * 77 + g * 150 + b * 29) as i32);
    }

    luma.into()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn satd_block_int(
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    dx: i32,
    dy: i32,
) -> i64 {
    satd_block_int_limit(
        prev_luma,
        curr_luma,
        width,
        height,
        x0,
        y0,
        dx,
        dy,
        i64::MAX,
    )
}

pub(super) fn sad_block_int(
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    dx: i32,
    dy: i32,
) -> i64 {
    sad_block_int_limit(
        prev_luma,
        curr_luma,
        width,
        height,
        x0,
        y0,
        dx,
        dy,
        i64::MAX,
    )
}

#[inline(always)]
fn satd_4x4_simd_core(curr_ptr: *const i32, prev_ptr: *const i32, stride: usize) -> i32x8 {
    unsafe {
        let c_01 = load_two_rows(curr_ptr, stride);
        let c_23 = load_two_rows(curr_ptr.add(2 * stride), stride);
        let p_01 = load_two_rows(prev_ptr, stride);
        let p_23 = load_two_rows(prev_ptr.add(2 * stride), stride);

        let diff_01 = c_01 - p_01;
        let diff_23 = c_23 - p_23;

        let h_01 = hadamard_8_step(diff_01, diff_23);
        h_01.0.abs() + h_01.1.abs()
    }
}

#[inline(always)]
unsafe fn load_two_rows(base: *const i32, stride: usize) -> i32x8 {
    unsafe {
        let values = [
            *base.add(0),
            *base.add(1),
            *base.add(2),
            *base.add(3),
            *base.add(stride),
            *base.add(stride + 1),
            *base.add(stride + 2),
            *base.add(stride + 3),
        ];

        i32x8::from(values) >> 8
    }
}
#[inline(always)]
fn hadamard_8_step(a: i32x8, b: i32x8) -> (i32x8, i32x8) {
    (a + b, a - b)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn satd_block_int_limit(
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    dx: i32,
    dy: i32,
    limit: i64,
) -> i64 {
    let ref_x = x0 + dx;
    let ref_y = y0 + dy;

    // 1. Check if the entire 16x16 block is safely inside the frame
    // This removes 16 individual 'if' checks inside the loop
    let in_bounds = x0 >= 0
        && y0 >= 0
        && (x0 + 16) <= width as i32
        && (y0 + 16) <= height as i32
        && ref_x >= 0
        && ref_y >= 0
        && (ref_x + 16) <= width as i32
        && (ref_y + 16) <= height as i32;

    if in_bounds {
        let mut total_satd: i32x8 = i32x8::ZERO;

        let c_slice = curr_luma.as_i32_slice();
        let p_slice = prev_luma.as_i32_slice();
        let c_start = y0 as usize * width + x0 as usize;
        let r_start = ref_y as usize * width + ref_x as usize;

        let c_ptr = c_slice.as_ptr();
        let p_ptr = p_slice.as_ptr();

        for by in 0..4 {
            let row_offset = by * 4 * width;
            for bx in 0..4 {
                let col_offset = bx * 4;
                let curr_idx = c_start + row_offset + col_offset;
                let prev_idx = r_start + row_offset + col_offset;
                let curr_ptr = unsafe { c_ptr.add(curr_idx) };
                let prev_ptr = unsafe { p_ptr.add(prev_idx) };

                total_satd += satd_4x4_simd_core(curr_ptr, prev_ptr, width);
            }
        }
        total_satd.reduce_add() as i64 / 2
    } else {
        // Fallback for frame edges
        super::satd_scalar::satd_block_int_limit(
            prev_luma, curr_luma, width, height, x0, y0, dx, dy, limit,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sad_block_int_limit(
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    dx: i32,
    dy: i32,
    limit: i64,
) -> i64 {
    let ref_x = x0 + dx;
    let ref_y = y0 + dy;
    let block_in_curr = x0 >= 0
        && y0 >= 0
        && (x0 + BLOCK_SIZE as i32) <= width as i32
        && (y0 + BLOCK_SIZE as i32) <= height as i32;
    let block_in_prev = ref_x >= 0
        && ref_y >= 0
        && (ref_x + BLOCK_SIZE as i32) <= width as i32
        && (ref_y + BLOCK_SIZE as i32) <= height as i32;

    if block_in_curr && block_in_prev {
        return sad_block_int_limit_in_bounds(
            prev_luma,
            curr_luma,
            width,
            x0 as usize,
            y0 as usize,
            ref_x as usize,
            ref_y as usize,
            limit,
        );
    }

    super::satd_scalar::sad_block_int_limit(
        prev_luma, curr_luma, width, height, x0, y0, dx, dy, limit,
    )
}

#[inline(always)]
fn sad_block_int_limit_in_bounds(
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    curr_x: usize,
    curr_y: usize,
    ref_x: usize,
    ref_y: usize,
    _limit: i64,
) -> i64 {
    let curr_ptr = unsafe {
        curr_luma
            .as_i32_slice()
            .as_ptr()
            .add(curr_y * width + curr_x)
    };
    let ref_ptr = unsafe { prev_luma.as_i32_slice().as_ptr().add(ref_y * width + ref_x) };

    let mut sad_acc = i32x8::ZERO;

    for row in 0..BLOCK_SIZE as usize {
        let offset = row * width;
        unsafe {
            let cp = curr_ptr.add(offset);
            let rp = ref_ptr.add(offset);

            // Load both 8-pixel chunks so we only shift once after the loop
            let c0 = load_i32x8(cp);
            let c1 = load_i32x8(cp.add(8));
            let r0 = load_i32x8(rp);
            let r1 = load_i32x8(rp.add(8));

            sad_acc += (c0 - r0).abs();
            sad_acc += (c1 - r1).abs();
        }
    }

    (sad_acc.reduce_add() as i64) >> 8
}

#[inline(always)]
fn load_i32x8(ptr: *const i32) -> i32x8 {
    unsafe {
        i32x8::from([
            *ptr.add(0),
            *ptr.add(1),
            *ptr.add(2),
            *ptr.add(3),
            *ptr.add(4),
            *ptr.add(5),
            *ptr.add(6),
            *ptr.add(7),
        ])
    }
}

#[inline]
fn clamp_hp(v_hp: i32, max_px: i32) -> i32 {
    v_hp.clamp(0, max_px * 2)
}

#[inline]
fn idx_luma(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

fn sample_luma_halfpel(prev_luma: &LumaPlane, width: usize, dx_hp: i32, dy_hp: i32) -> i32x8 {
    let height = prev_luma.as_slice().len() / width;
    let x_hp = clamp_hp(dx_hp, width as i32 - 1);
    let y_hp = clamp_hp(dy_hp, height as i32 - 1);
    let x_odd = (x_hp & 1) != 0;
    let y_odd = (y_hp & 1) != 0;

    // Convert hp coordinates to integer base
    let x0 = (x_hp >> 1) as usize;
    let y0 = (y_hp >> 1) as usize;
    let idx = y0 * width + x0;

    // Load the necessary 8-pixel wide blocks
    match (x_odd, y_odd) {
        (false, false) => load_i32x8_unaligned(&prev_luma.as_slice()[idx..]),
        (true, false) => {
            let a = load_i32x8_unaligned(&prev_luma.as_slice()[idx..]);
            let b = load_i32x8_unaligned(&prev_luma.as_slice()[idx + 1..]);
            (a + b + i32x8::ONE) >> 1
        }
        (false, true) => {
            let a = load_i32x8_unaligned(&prev_luma.as_slice()[idx..]);
            let d = load_i32x8_unaligned(&prev_luma.as_slice()[idx + width..]);
            (a + d + i32x8::ONE) >> 1
        }
        (true, true) => {
            let a = load_i32x8_unaligned(&prev_luma.as_slice()[idx..]);
            let b = load_i32x8_unaligned(&prev_luma.as_slice()[idx + 1..]);
            let d = load_i32x8_unaligned(&prev_luma.as_slice()[idx + width..]);
            let e = load_i32x8_unaligned(&prev_luma.as_slice()[idx + width + 1..]);
            (a + b + d + e + i32x8::splat(2)) >> 2
        }
    }
}

#[inline(always)]
fn load_i32x8_unaligned(slice: &[MotionLuma]) -> i32x8 {
    let mut tmp = [0i32; 8];
    unsafe {
        // Since MotionLuma is i32, this is a direct 32-byte copy
        std::ptr::copy_nonoverlapping(slice.as_ptr() as *const i32, tmp.as_mut_ptr(), 8);
    }
    i32x8::new(tmp)
}

#[cfg(feature = "bench-luma")]
pub(super) fn bench_sample_luma_halfpel(plane: &LumaPlane, width: usize, height: usize) {
    let mut acc: u32 = 0;
    let width = width.max(1);
    let height = height.max(1);
    for y in (0..height).step_by(2) {
        let y_hp = ((y as i32) * 2) + ((y as i32) & 1);
        for x in (0..width).step_by(2) {
            let x_hp = ((x as i32) * 2) + (((x + y) as i32) & 1);
            let sample = sample_luma_halfpel(plane, width, x_hp, y_hp);
            acc ^= sample.reduce_add() as u32;
        }
    }
    black_box(acc);
}

#[cfg(feature = "bench-luma")]
pub(super) fn bench_sad_block_halfpel(
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    positions: &[(i32, i32)],
    offsets: &[(i32, i32)],
) -> i64 {
    let mut acc: i64 = 0;
    for &(x0, y0) in positions {
        for &(dx_hp, dy_hp) in offsets {
            let sad = sad_block_halfpel_limit_luma(
                prev_luma,
                curr_luma,
                width,
                height,
                x0,
                y0,
                dx_hp,
                dy_hp,
                i64::MAX,
            );
            acc ^= sad;
        }
    }
    acc
}

#[inline(always)]
fn is_in_bounds(x0: i32, y0: i32, width: usize, height: usize, dx_hp: i32, dy_hp: i32) -> bool {
    // 1. Convert half-pel offsets to integer "extra" requirements
    // If dx_hp is odd, we need x + BLOCK_SIZE + 1 pixels
    let x_extra = if (dx_hp & 1) != 0 { 1 } else { 0 };
    // If dy_hp is odd, we need y + BLOCK_SIZE + 1 pixels
    let y_extra = if (dy_hp & 1) != 0 { 1 } else { 0 };

    // 2. Check the top-left corner
    if x0 < 0 || y0 < 0 {
        return false;
    }

    // 3. Check the bottom-right corner
    // We need to be able to load BLOCK_SIZE pixels plus the interpolation neighbor
    let x_end = x0 as usize + BLOCK_SIZE + x_extra;
    let y_end = y0 as usize + BLOCK_SIZE + y_extra;

    x_end <= width && y_end <= height
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sad_block_halfpel_limit_luma(
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    dx_hp: i32,
    dy_hp: i32,
    limit: i64,
) -> i64 {
    let mut total_sad: i64 = 0;

    let width_i32 = width as i32;
    let height_i32 = height as i32;
    let max_x_hp = (width_i32 - 1) * 2;
    let max_y_hp = (height_i32 - 1) * 2;
    let x_hp_min = (x0 * 2) + dx_hp;
    let x_hp_max = ((x0 + (BLOCK_SIZE as i32 - 1)) * 2) + dx_hp;
    let y_hp_min = (y0 * 2) + dy_hp;
    let y_hp_max = ((y0 + (BLOCK_SIZE as i32 - 1)) * 2) + dy_hp;
    let x_in = x_hp_min >= 0 && x_hp_max <= max_x_hp;
    let y_in = y_hp_min >= 0 && y_hp_max <= max_y_hp;

    // We only use the SIMD path if we are away from the image edges
    // to avoid complex clamping inside the SIMD loop.
    if is_in_bounds(x0, y0, width, height, dx_hp, dy_hp) && x_in && y_in {
        let mut sad_acc = i32x8::ZERO;
        let lanes: i32 = 8;

        for yy in 0..BLOCK_SIZE as i32 {
            let cy = (y0 + yy) as usize;
            let ry_hp = (cy as i32 * 2) + dy_hp;

            for chunk in 0..(BLOCK_SIZE as i32 / lanes) {
                let cx = x0 + chunk * lanes;
                let rx_hp = (cx * 2) + dx_hp;

                // 1. Get 8 interpolated pixels at once
                let ref_vec: i32x8 = sample_luma_halfpel(prev_luma, width, rx_hp, ry_hp) >> 8;

                // 2. Get 8 current pixels at once
                let curr_idx = cy * width + cx as usize;
                let curr_vec: i32x8 = load_i32x8_unaligned(&curr_luma.as_slice()[curr_idx..]) >> 8;

                // 3. Accumulate absolute differences
                sad_acc += (curr_vec - ref_vec).abs();
            }

            //deliberatly no early exit on limit here; we want to keep the SIMD loop simple
            //if branching does not gain speed here, it costs complexity
        }
        total_sad += sad_acc.reduce_add() as i64;
    } else {
        // Fallback to your original scalar code for the 1-pixel border
        // around the edge of the frame.
        return super::satd_scalar::sad_block_halfpel_limit_luma(
            prev_luma, curr_luma, width, height, x0, y0, dx_hp, dy_hp, limit,
        );
    }

    total_sad
}
