use super::{BLOCK_SIZE, LumaPlane, MotionLuma};

#[inline]
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
    let mut sad: i64 = 0;
    debug_assert!(x0 >= 0 && y0 >= 0);
    debug_assert_eq!(width % BLOCK_SIZE, 0);
    debug_assert_eq!(height % BLOCK_SIZE, 0);

    let curr = curr_luma.as_slice();
    let prev = prev_luma.as_slice();
    debug_assert_eq!(curr.len(), prev.len());
    let w_i32 = width as i32;
    let h_i32 = height as i32;

    for by in 0..4 {
        for bx in 0..4 {
            let sub_x0 = x0 + (bx * 4);
            let sub_y0 = y0 + (by * 4);

            if sub_x0 < 0 || sub_y0 < 0 || sub_x0 >= w_i32 || sub_y0 >= h_i32 {
                continue;
            }

            let mut ref_x0 = sub_x0 + dx;
            let mut ref_y0 = sub_y0 + dy;

            if ref_x0 < 0 {
                ref_x0 = 0;
            }
            if ref_y0 < 0 {
                ref_y0 = 0;
            }
            if ref_x0 + 4 > w_i32 {
                ref_x0 = (w_i32 - 4).max(0);
            }
            if ref_y0 + 4 > h_i32 {
                ref_y0 = (h_i32 - 4).max(0);
            }

            let curr_idx = ((sub_y0 as usize) * width) + (sub_x0 as usize);
            let prev_idx = ((ref_y0 as usize) * width) + (ref_x0 as usize);

            let block_in_curr = sub_y0 + 4 <= h_i32 && sub_x0 + 4 <= w_i32;
            let block_in_prev = ref_y0 + 4 <= h_i32 && ref_x0 + 4 <= w_i32;

            if block_in_curr && block_in_prev {
                for sy in 0..4 {
                    let curr_row = curr_idx + (sy as usize) * width;
                    let prev_row = prev_idx + (sy as usize) * width;
                    for sx in 0..4 {
                        let ci = curr_row + sx as usize;
                        let pi = prev_row + sx as usize;
                        let diff = curr[ci].to_satd_sample() - prev[pi].to_satd_sample();
                        sad += diff.abs() as i64;
                    }
                }
            } else {
                for sy in 0..4 {
                    for sx in 0..4 {
                        let cy = sub_y0 + sy;
                        let cx = sub_x0 + sx;
                        let ry = ref_y0 + sy;
                        let rx = ref_x0 + sx;
                        if cy >= 0
                            && cy < h_i32
                            && cx >= 0
                            && cx < w_i32
                            && ry >= 0
                            && ry < h_i32
                            && rx >= 0
                            && rx < w_i32
                        {
                            let ci = ((cy as usize) * width) + (cx as usize);
                            let ri = ((ry as usize) * width) + (rx as usize);
                            let diff = curr[ci].to_satd_sample() - prev[ri].to_satd_sample();
                            sad += diff.abs() as i64;
                        }
                    }
                }
            }

            if sad > limit {
                return sad;
            }
        }
    }

    sad
}

#[inline]
fn idx_luma(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

#[inline]
fn clamp_hp(v_hp: i32, max_px: i32) -> i32 {
    v_hp.clamp(0, max_px * 2)
}

#[inline]
fn sample_luma_halfpel(
    plane: &LumaPlane,
    width: usize,
    height: usize,
    x_hp: i32,
    y_hp: i32,
) -> MotionLuma {
    let x_hp = clamp_hp(x_hp, width as i32 - 1);
    let y_hp = clamp_hp(y_hp, height as i32 - 1);
    sample_luma_halfpel_unchecked(plane, width, height, x_hp, y_hp)
}

#[inline]
fn sample_luma_halfpel_unchecked(
    plane: &LumaPlane,
    width: usize,
    height: usize,
    x_hp: i32,
    y_hp: i32,
) -> MotionLuma {
    debug_assert!(width > 0 && height > 0);
    debug_assert!(x_hp >= 0 && x_hp <= (width as i32 - 1) * 2);
    debug_assert!(y_hp >= 0 && y_hp <= (height as i32 - 1) * 2);

    let slice = plane.as_slice();
    let x0 = (x_hp / 2) as usize;
    let y0 = (y_hp / 2) as usize;
    let x_odd = (x_hp & 1) != 0;
    let y_odd = (y_hp & 1) != 0;

    let x1 = if x0 + 1 < width { x0 + 1 } else { x0 };
    let y1 = if y0 + 1 < height { y0 + 1 } else { y0 };

    let i00 = idx_luma(x0, y0, width);
    if !x_odd && !y_odd {
        return slice[i00];
    }

    let i10 = idx_luma(x1, y0, width);
    let i01 = idx_luma(x0, y1, width);
    let i11 = idx_luma(x1, y1, width);

    let a: i32 = slice[i00].into();
    let b: i32 = slice[i10].into();
    let d: i32 = slice[i01].into();
    let e: i32 = slice[i11].into();

    let fixed = match (x_odd, y_odd) {
        (true, false) => (a + b + 1) / 2,
        (false, true) => (a + d + 1) / 2,
        (true, true) => (a + b + d + e + 2) / 4,
        _ => a,
    };

    MotionLuma::from_fixed_point(fixed)
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
    let mut sad: i64 = 0;
    debug_assert!(x0 >= 0 && y0 >= 0);
    debug_assert_eq!(width % BLOCK_SIZE, 0);
    debug_assert_eq!(height % BLOCK_SIZE, 0);

    let prev = prev_luma.as_slice();
    let curr = curr_luma.as_slice();
    debug_assert_eq!(prev.len(), curr.len());

    let max_x_hp = (width as i32 - 1) * 2;
    let max_y_hp = (height as i32 - 1) * 2;
    let x_hp_min = (x0 * 2) + dx_hp;
    let x_hp_max = ((x0 + (BLOCK_SIZE as i32 - 1)) * 2) + dx_hp;
    let x_in = x_hp_min >= 0 && x_hp_max <= max_x_hp;

    for yy in 0..BLOCK_SIZE as i32 {
        let cy = y0 + yy;
        let y_hp = (cy * 2) + dy_hp;
        let y_in = y_hp >= 0 && y_hp <= max_y_hp;
        for xx in 0..BLOCK_SIZE as i32 {
            let cx = x0 + xx;
            let ci = idx_luma(cx as usize, cy as usize, width);

            let rx_hp = (cx * 2) + dx_hp;
            let ry_hp = (cy * 2) + dy_hp;
            let ref_luma = if x_in && y_in {
                sample_luma_halfpel_unchecked(prev_luma, width, height, rx_hp, ry_hp)
            } else {
                sample_luma_halfpel(prev_luma, width, height, rx_hp, ry_hp)
            };

            let curr_sample = curr[ci].to_satd_sample();
            let ref_sample = ref_luma.to_satd_sample();
            sad += (curr_sample - ref_sample).abs() as i64;
            if sad > limit {
                return sad;
            }
        }
    }
    sad
}

