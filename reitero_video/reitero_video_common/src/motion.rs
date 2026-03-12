use crate::yuv::Yuv420Frame;
use crate::motion_vector::MotionVector;

const BLOCK_SIZE: usize = 16;

// MotionVector struct and impl moved to motion_vector.rs


#[inline]
pub fn decode_halfpel_i8(code: u8) -> i8 {
    match code & 0x03 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

// encode_halfpel removed; external code should pass subpixel as -1,0,1

// Old unpack_mv_delta_hp removed - using new 3-byte format

#[inline(always)]
fn clamp_hp(v_hp: i32, max_px: i32) -> i32 {
    v_hp.clamp(0, max_px * 2)
}

#[inline(always)]
pub(crate) fn sample_plane_halfpel(plane: &[u8], width: usize, height: usize, x_hp: i32, y_hp: i32) -> u8 {
    let x_hp = clamp_hp(x_hp, width as i32 - 1);
    let y_hp = clamp_hp(y_hp, height as i32 - 1);

    let x0 = (x_hp / 2) as usize;
    let y0 = (y_hp / 2) as usize;
    let x_odd = (x_hp & 1) != 0;
    let y_odd = (y_hp & 1) != 0;

    let x1 = if x0 + 1 < width { x0 + 1 } else { x0 };
    let y1 = if y0 + 1 < height { y0 + 1 } else { y0 };

    let i00 = y0 * width + x0;
    if !x_odd && !y_odd {
        return plane[i00];
    }

    let i10 = y0 * width + x1;
    let i01 = y1 * width + x0;
    let i11 = y1 * width + x1;

    let a = plane[i00] as u16;
    let b = plane[i10] as u16;
    let d = plane[i01] as u16;
    let e = plane[i11] as u16;

    match (x_odd, y_odd) {
        (true, false) => ((a + b + 1) / 2) as u8,
        (false, true) => ((a + d + 1) / 2) as u8,
        (true, true) => ((a + b + d + e + 2) / 4) as u8,
        _ => a as u8,
    }
}

// Combined chroma sampler: sample both U and V planes at same half-pel coords to reuse clamp/index math
#[inline(always)]
pub fn sample_uv_halfpel(
    u_plane: &[u8],
    v_plane: &[u8],
    width: usize,
    height: usize,
    x_hp: i32,
    y_hp: i32,
) -> (u8, u8) {
    let x_hp = clamp_hp(x_hp, width as i32 - 1);
    let y_hp = clamp_hp(y_hp, height as i32 - 1);

    let x0 = (x_hp / 2) as usize;
    let y0 = (y_hp / 2) as usize;
    let x_odd = (x_hp & 1) != 0;
    let y_odd = (y_hp & 1) != 0;

    let x1 = if x0 + 1 < width { x0 + 1 } else { x0 };
    let y1 = if y0 + 1 < height { y0 + 1 } else { y0 };

    let i00 = y0 * width + x0;
    if !x_odd && !y_odd {
        return (u_plane[i00], v_plane[i00]);
    }

    let i10 = y0 * width + x1;
    let i01 = y1 * width + x0;
    let i11 = y1 * width + x1;

    let au = u_plane[i00] as u16;
    let bu = u_plane[i10] as u16;
    let du = u_plane[i01] as u16;
    let eu = u_plane[i11] as u16;

    let av = v_plane[i00] as u16;
    let bv = v_plane[i10] as u16;
    let dv = v_plane[i01] as u16;
    let ev = v_plane[i11] as u16;

    let u_res = match (x_odd, y_odd) {
        (true, false) => ((au + bu + 1) / 2) as u8,
        (false, true) => ((au + du + 1) / 2) as u8,
        (true, true) => ((au + bu + du + eu + 2) / 4) as u8,
        _ => au as u8,
    };

    let v_res = match (x_odd, y_odd) {
        (true, false) => ((av + bv + 1) / 2) as u8,
        (false, true) => ((av + dv + 1) / 2) as u8,
        (true, true) => ((av + bv + dv + ev + 2) / 4) as u8,
        _ => av as u8,
    };

    (u_res, v_res)
}

/// Sample YUV420 with half-pixel precision using bilinear interpolation on each plane.
#[inline]
pub fn sample_rgb_halfpel(
    prev: &Yuv420Frame,
    width: usize,
    height: usize,
    x_hp: i32,
    y_hp: i32,
) -> [u8; 3] {
    debug_assert_eq!(prev.width(), width);
    debug_assert_eq!(prev.height(), height);

    let y = sample_plane_halfpel(prev.y_plane(), width, height, x_hp, y_hp);
    let chroma_width = width / 2;
    let chroma_height = height / 2;
    let u = sample_plane_halfpel(
        prev.u_plane(),
        chroma_width,
        chroma_height,
        x_hp / 2,
        y_hp / 2,
    );
    let v = sample_plane_halfpel(
        prev.v_plane(),
        chroma_width,
        chroma_height,
        x_hp / 2,
        y_hp / 2,
    );
    [y, u, v]
}

/// Reference (scalar) predicted frame builder
pub fn reference_build_predicted(
    prev: &Yuv420Frame,
    width: usize,
    height: usize,
    mvs: &[MotionVector],
) -> Yuv420Frame {
    // Instrumentation: coarse measure for reference path
    crate::Instrument::start_measure("10_reference_build_predicted");
    let blocks_w = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let blocks_h = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    debug_assert_eq!(prev.width(), width);
    debug_assert_eq!(prev.height(), height);

    let mut predicted_y = vec![0u8; width * height];
    let mut predicted_u = vec![0u8; (width / 2) * (height / 2)];
    let mut predicted_v = vec![0u8; (width / 2) * (height / 2)];

    // Optimize: sample Y for every pixel, but sample U/V only once per 2x2 luma area.
    // Precompute plane references and chroma sizes to avoid repeated computations.
    let y_plane = prev.y_plane();
    let u_plane = prev.u_plane();
    let v_plane = prev.v_plane();
    let chroma_width = width / 2;
    let chroma_height = height / 2;
    let height_i32 = height as i32;
    let width_i32 = width as i32;

    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let mv = mvs[by * blocks_w + bx];
            let mv_dx_hp = mv.dx_hp();
            let mv_dy_hp = mv.dy_hp();
            let x0 = (bx * BLOCK_SIZE) as i32;
            let y0 = (by * BLOCK_SIZE) as i32;

            // Fast path: if motion vector has no half-pixel component (integer-pixel shift), we can copy blocks directly
            if (mv_dx_hp & 1) == 0 && (mv_dy_hp & 1) == 0 {
                let dx = mv_dx_hp / 2;
                let dy = mv_dy_hp / 2;
                let src_x0 = x0 + dx;
                let src_y0 = y0 + dy;

                // Check if entire block falls inside source frame bounds and destination bounds
                if src_x0 >= 0
                    && src_y0 >= 0
                    && src_x0 + (BLOCK_SIZE as i32) - 1 < width_i32
                    && src_y0 + (BLOCK_SIZE as i32) - 1 < height_i32
                {
                    // Luma block copy
                    for row in 0..BLOCK_SIZE {
                        let dst_y = (y0 as usize) + row;
                        let dst_x = x0 as usize;
                        let src_y = (src_y0 as usize) + row;
                        let src_x = src_x0 as usize;
                        let dst_base = dst_y * width + dst_x;
                        let src_base = src_y * width + src_x;
                        predicted_y[dst_base..dst_base + BLOCK_SIZE]
                            .copy_from_slice(&y_plane[src_base..src_base + BLOCK_SIZE]);
                    }

                    // Chroma block copy (8x8)
                    // Safer compute: integer division
                    let chroma_src_x0 = (src_x0 as i32 / 2) as usize;
                    let chroma_src_y0 = (src_y0 as i32 / 2) as usize;
                    if chroma_src_x0 + (BLOCK_SIZE / 2) <= chroma_width && chroma_src_y0 + (BLOCK_SIZE / 2) <= chroma_height {
                        for row in 0..(BLOCK_SIZE / 2) {
                            let dst_y = (y0 as usize) / 2 + row;
                            let dst_x = x0 as usize / 2;
                            let src_y = chroma_src_y0 + row;
                            let src_x = chroma_src_x0;
                            let dst_base = dst_y * chroma_width + dst_x;
                            let src_base = src_y * chroma_width + src_x;
                            predicted_u[dst_base..dst_base + (BLOCK_SIZE / 2)]
                                .copy_from_slice(&u_plane[src_base..src_base + (BLOCK_SIZE / 2)]);
                            predicted_v[dst_base..dst_base + (BLOCK_SIZE / 2)]
                                .copy_from_slice(&v_plane[src_base..src_base + (BLOCK_SIZE / 2)]);
                        }
                    }

                    continue; // next block
                }
            }

            for yy in 0..BLOCK_SIZE as i32 {
                let y = (y0 + yy).clamp(0, height_i32 - 1);
                let y_usize = y as usize;
                let y_idx_base = y_usize * width;
                let even_y = (y_usize & 1) == 0;
                // Precompute ry_hp for this row
                let ry_hp = y * 2 + mv_dy_hp;

                // Start rx_hp at leftmost x for the block
                let mut rx_hp = (x0 * 2) + mv_dx_hp;

                for xx in 0..BLOCK_SIZE as i32 {
                    let x = (x0 + xx).clamp(0, width_i32 - 1);
                    let x_usize = x as usize;

                    // Luma sample using incremental rx_hp
                    let y_sample = sample_plane_halfpel(y_plane, width, height, rx_hp, ry_hp);
                    predicted_y[y_idx_base + x_usize] = y_sample;

                    // Chroma sample only on even luma coordinates (maps to one chroma sample)
                    if even_y && (x_usize & 1) == 0 {
                        // sample both chroma planes together to reuse clamps/index math
                        let (u_sample, v_sample) = sample_uv_halfpel(
                            u_plane,
                            v_plane,
                            chroma_width,
                            chroma_height,
                            rx_hp >> 1,
                            ry_hp >> 1,
                        );
                        let chroma_x = x_usize / 2;
                        let chroma_y = y_usize / 2;
                        let chroma_idx = chroma_y * chroma_width + chroma_x;
                        predicted_u[chroma_idx] = u_sample;
                        predicted_v[chroma_idx] = v_sample;
                    }

                    // advance rx_hp by 2 half-pixels for next x
                    rx_hp += 2;
                }
            }
        }
    }

    let out = Yuv420Frame::from_planes(width, height, predicted_y, predicted_u, predicted_v)
        .expect("predicted YUV420 frame dimensions are always valid; this is a bug");
    crate::Instrument::stop_measure("10_reference_build_predicted");
    out
}
