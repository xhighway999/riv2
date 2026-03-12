//! Placeholder SIMD backend.
//!
//! The SIMD version currently just forwards to the scalar SATD routines so we
//! can start wiring up the feature-gated module structure without changing
//! behavior.

use super::BLOCK_SIZE;

use super::{LumaPlane, MotionLuma};
use wide::i32x8;



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
