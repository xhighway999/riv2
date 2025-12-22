use super::{BLOCK_SIZE, LumaPlane, MotionLuma};

#[cfg(feature = "bench-luma")]
use std::hint::black_box;

#[inline]
pub(super) fn rgb_to_luma_sample(r: u8, g: u8, b: u8) -> MotionLuma {
    // Fixed-point approximation of 0.299R + 0.587G + 0.114B without truncating precision.
    let fixed = r as u32 * 77 + g as u32 * 150 + b as u32 * 29;
    fixed.into()
}

pub(super) fn rgb_to_luma_plane(rgb: &[u8], width: usize, height: usize) -> LumaPlane {
    let pixels = width
        .checked_mul(height)
        .expect("width*height must fit into usize");
    let expected_bytes = pixels.checked_mul(3).expect("RGB buffer size overflow");
    assert_eq!(rgb.len(), expected_bytes, "RGB buffer size mismatch");

    let mut luma = Vec::with_capacity(pixels);
    for chunk in rgb.chunks_exact(3) {
        luma.push(rgb_to_luma_sample(chunk[0], chunk[1], chunk[2]));
    }
    luma.into()
}

// 4x4 Hadamard transform (used for SATD)
#[inline]
fn hadamard_4x4(input: &[i32; 16]) -> [i32; 16] {
    let mut temp = [0i32; 16];
    let mut output = [0i32; 16];

    // First pass: horizontal transform
    for i in 0..4 {
        let base = i * 4;
        temp[base + 0] = input[base + 0] + input[base + 3];
        temp[base + 1] = input[base + 1] + input[base + 2];
        temp[base + 2] = input[base + 1] - input[base + 2];
        temp[base + 3] = input[base + 0] - input[base + 3];
    }

    // Second pass: vertical transform
    for j in 0..4 {
        output[j + 0] = temp[j + 0] + temp[j + 12];
        output[j + 4] = temp[j + 4] + temp[j + 8];
        output[j + 8] = temp[j + 4] - temp[j + 8];
        output[j + 12] = temp[j + 0] - temp[j + 12];
    }

    output
}

// Compute SATD for a 4x4 block using Hadamard transform on precomputed luma samples.
#[inline]
fn satd_4x4_luma(
    curr: &[MotionLuma],
    prev: &[MotionLuma],
    curr_idx: usize,
    prev_idx: usize,
    curr_stride: usize,
    prev_stride: usize,
) -> i64 {
    let mut diff = [0i32; 16];
    for y in 0..4 {
        for x in 0..4 {
            let ci = curr_idx + y * curr_stride + x;
            let pi = prev_idx + y * prev_stride + x;
            let cy = curr[ci].to_satd_sample();
            let py = prev[pi].to_satd_sample();
            diff[y * 4 + x] = cy - py;
        }
    }

    let transformed = hadamard_4x4(&diff);
    transformed.iter().map(|&x| x.abs() as i64).sum::<i64>() / 2
}

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

#[inline]
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
    let mut satd: i64 = 0;
    debug_assert!(x0 >= 0 && y0 >= 0);
    debug_assert_eq!(width % BLOCK_SIZE, 0);
    debug_assert_eq!(height % BLOCK_SIZE, 0);
    debug_assert_eq!(prev_luma.as_slice().len(), curr_luma.as_slice().len());
    debug_assert_eq!(prev_luma.as_slice().len(), width * height);

    let w_i32 = width as i32;
    let h_i32 = height as i32;
    let x0_i32 = x0;

    for by in 0..4 {
        for bx in 0..4 {
            let sub_x0 = x0_i32 + (bx * 4);
            let sub_y0 = y0 + (by * 4);

            if sub_x0 < 0 || sub_y0 < 0 || sub_x0 + 4 > w_i32 || sub_y0 + 4 > h_i32 {
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
                ref_x0 = w_i32 - 4;
            }
            if ref_y0 + 4 > h_i32 {
                ref_y0 = h_i32 - 4;
            }

            let curr_idx = ((sub_y0 as usize) * width) + (sub_x0 as usize);
            let prev_idx = ((ref_y0 as usize) * width) + (ref_x0 as usize);

            let block_in_curr = sub_y0 + 4 <= h_i32 && sub_x0 + 4 <= w_i32;
            let block_in_prev = ref_y0 + 4 <= h_i32 && ref_x0 + 4 <= w_i32;

            if block_in_curr && block_in_prev {
                satd += satd_4x4_luma(
                    curr_luma.as_slice(),
                    prev_luma.as_slice(),
                    curr_idx,
                    prev_idx,
                    width,
                    width,
                );
            } else {
                let mut sub_sad = 0i64;
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
                            let diff = curr_luma.as_slice()[ci].to_satd_sample()
                                - prev_luma.as_slice()[ri].to_satd_sample();
                            sub_sad += diff.abs() as i64;
                        }
                    }
                }
                satd += sub_sad;
            }

            if satd > limit {
                return satd;
            }
        }
    }
    satd
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

#[cfg(feature = "bench-luma")]
pub(super) fn bench_sample_luma_halfpel(plane: &LumaPlane, width: usize, height: usize) {
    let mut acc: u32 = 0;
    let width = width.max(1);
    let height = height.max(1);
    for y in (0..height).step_by(2) {
        let y_hp = ((y as i32) * 2) + ((y as i32) & 1);
        for x in (0..width).step_by(2) {
            let x_hp = ((x as i32) * 2) + (((x + y) as i32) & 1);
            let sample = sample_luma_halfpel(plane, width, height, x_hp, y_hp);
            acc ^= sample.as_fixed_point() as u32;
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
