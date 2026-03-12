use crate::MotionVector;
use crate::yuv::Yuv420Frame;
// Local constant mirrors motion.rs
const BLOCK_SIZE: usize = 16;



#[inline(always)]
fn clamp_hp(v_hp: i32, max_px: i32) -> i32 { v_hp.clamp(0, max_px * 2) }

#[inline(always)]
fn avg2_u8(a: &[u8], b: &[u8], dst: &mut [u8]) {
    let len = a.len().min(b.len()).min(dst.len());
    // Unchecked, unrolled pointer loop; lets LLVM auto-vectorize.
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0usize;
        // 64-byte chunks
        while i + 64 <= len {
            let mut k = 0;
            while k < 64 {
                let ai = *ap.add(i + k) as u16;
                let bi = *bp.add(i + k) as u16;
                *dp.add(i + k) = ((ai + bi + 1) >> 1) as u8;
                k += 1;
            }
            i += 64;
        }
        // 16-byte chunks
        while i + 16 <= len {
            let mut k = 0;
            while k < 16 {
                let ai = *ap.add(i + k) as u16;
                let bi = *bp.add(i + k) as u16;
                *dp.add(i + k) = ((ai + bi + 1) >> 1) as u8;
                k += 1;
            }
            i += 16;
        }
        // tail
        while i < len {
            let ai = *ap.add(i) as u16;
            let bi = *bp.add(i) as u16;
            *dp.add(i) = ((ai + bi + 1) >> 1) as u8;
            i += 1;
        }
    }
}

#[inline(always)]
fn avg4_u8(a: &[u8], b: &[u8], d: &[u8], e: &[u8], dst: &mut [u8]) {
    let len = a.len().min(b.len()).min(d.len()).min(e.len()).min(dst.len());
    // Unchecked, unrolled pointer loop; lets LLVM auto-vectorize.
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let dp1 = d.as_ptr();
        let ep = e.as_ptr();
        let outp = dst.as_mut_ptr();
        let mut i = 0usize;
        // 64-byte chunks
        while i + 64 <= len {
            let mut k = 0;
            while k < 64 {
                let a0 = *ap.add(i + k) as u16;
                let b0 = *bp.add(i + k) as u16;
                let d0 = *dp1.add(i + k) as u16;
                let e0 = *ep.add(i + k) as u16;
                *outp.add(i + k) = ((a0 + b0 + d0 + e0 + 2) >> 2) as u8;
                k += 1;
            }
            i += 64;
        }
        // 16-byte chunks
        while i + 16 <= len {
            let mut k = 0;
            while k < 16 {
                let a0 = *ap.add(i + k) as u16;
                let b0 = *bp.add(i + k) as u16;
                let d0 = *dp1.add(i + k) as u16;
                let e0 = *ep.add(i + k) as u16;
                *outp.add(i + k) = ((a0 + b0 + d0 + e0 + 2) >> 2) as u8;
                k += 1;
            }
            i += 16;
        }
        // tail
        while i < len {
            let a0 = *ap.add(i) as u16;
            let b0 = *bp.add(i) as u16;
            let d0 = *dp1.add(i) as u16;
            let e0 = *ep.add(i) as u16;
            *outp.add(i) = ((a0 + b0 + d0 + e0 + 2) >> 2) as u8;
            i += 1;
        }
    }
}

/// SIMD-accelerated predictor builder. Vectorizes luma half-pel interpolation across rows.
#[inline(always)]
pub fn build_predicted(
    prev: &Yuv420Frame,
    width: usize,
    height: usize,
    mvs: &[MotionVector],
) -> Yuv420Frame {
    let blocks_w = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let blocks_h = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    debug_assert_eq!(prev.width(), width);
    debug_assert_eq!(prev.height(), height);

    let mut predicted_y = vec![0u8; width * height];
    let mut predicted_u = vec![0u8; (width / 2) * (height / 2)];
    let mut predicted_v = vec![0u8; (width / 2) * (height / 2)];

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

            // Integer-pixel fast path: direct copy
            if (mv_dx_hp & 1) == 0 && (mv_dy_hp & 1) == 0 {
                let dx = mv_dx_hp / 2;
                let dy = mv_dy_hp / 2;
                let src_x0 = x0 + dx;
                let src_y0 = y0 + dy;
                if src_x0 >= 0
                    && src_y0 >= 0
                    && src_x0 + (BLOCK_SIZE as i32) - 1 < width_i32
                    && src_y0 + (BLOCK_SIZE as i32) - 1 < height_i32
                {
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
                    // Chroma block copy identical to reference
                    let chroma_src_x0 = (src_x0 as i32 / 2) as usize;
                    let chroma_src_y0 = (src_y0 as i32 / 2) as usize;
                    if chroma_src_x0 + (BLOCK_SIZE / 2) <= chroma_width && chroma_src_y0 + (BLOCK_SIZE / 2) <= chroma_height {
                        for row in 0..(BLOCK_SIZE / 2) {
                            let dst_yc = (y0 as usize) / 2 + row;
                            let dst_xc = x0 as usize / 2;
                            let src_yc = chroma_src_y0 + row;
                            let src_xc = chroma_src_x0;
                            let dst_base = dst_yc * chroma_width + dst_xc;
                            let src_base = src_yc * chroma_width + src_xc;
                            predicted_u[dst_base..dst_base + (BLOCK_SIZE / 2)]
                                .copy_from_slice(&u_plane[src_base..src_base + (BLOCK_SIZE / 2)]);
                            predicted_v[dst_base..dst_base + (BLOCK_SIZE / 2)]
                                .copy_from_slice(&v_plane[src_base..src_base + (BLOCK_SIZE / 2)]);
                        }
                    } else {
                        // Fallback to exact sampling per 2x2 area for edge blocks
                        for yy in 0..BLOCK_SIZE as i32 {
                            let y = (y0 + yy).clamp(0, height_i32 - 1) as usize;
                            if (y & 1) != 0 { continue; }
                            let ry_hp_raw = (y as i32) * 2 + mv_dy_hp;
                            let mut rx_hp_raw = x0 * 2 + mv_dx_hp;
                            for xx in (0..BLOCK_SIZE as usize).step_by(2) {
                                let x = (x0 as usize) + xx;
                                let chroma_x = x / 2;
                                let chroma_y = y / 2;
                                let (u, v) = crate::motion::sample_uv_halfpel(
                                    u_plane,
                                    v_plane,
                                    chroma_width,
                                    chroma_height,
                                    rx_hp_raw >> 1,
                                    ry_hp_raw >> 1,
                                );
                                let chroma_idx = chroma_y * chroma_width + chroma_x;
                                predicted_u[chroma_idx] = u;
                                predicted_v[chroma_idx] = v;
                                rx_hp_raw += 2;
                            }
                        }
                    }
                    continue;
                }
            }

            for yy in 0..BLOCK_SIZE as i32 {
                let y = (y0 + yy).clamp(0, height_i32 - 1);
                let y_usize = y as usize;
                let dst_row_base = y_usize * width + x0 as usize;

                let ry_hp = clamp_hp(y * 2 + mv_dy_hp, height as i32 - 1);
                let y0p = (ry_hp / 2) as usize;
                let y1p = if y0p + 1 < height { y0p + 1 } else { y0p };
                let y_odd = (ry_hp & 1) != 0;

                // Compute starting x half-pel and derived positions
                let rx_hp0_raw = x0 * 2 + mv_dx_hp;
                let rx_hp0 = clamp_hp(rx_hp0_raw, width as i32 - 1);
                let x0p = (rx_hp0 / 2) as usize;
                let x_odd = (rx_hp0 & 1) != 0;
                // When within bounds (no right-edge clamp), the whole row is contiguous
                let can_use_contig = rx_hp0_raw >= 0
                    && x0p + BLOCK_SIZE <= width
                    && (!x_odd || (x0p + BLOCK_SIZE < width));

                if mv_dx_hp < 0 || !can_use_contig {
                    // Fallback to scalar sampling near borders or for negative half-pel (parity-sensitive)
                    // Start from raw half-pel position; clamp inside sampler to preserve oddness near borders
                    let mut rx_hp = x0 * 2 + mv_dx_hp;
                    for xx in 0..BLOCK_SIZE as i32 {
                        let x = (x0 + xx).clamp(0, width_i32 - 1);
                        let x_usize = x as usize;
                        let y_sample = crate::motion::sample_plane_halfpel(
                            y_plane,
                            width,
                            height,
                            rx_hp,
                            ry_hp,
                        );
                        predicted_y[dst_row_base + x_usize - x0 as usize] = y_sample;
                        rx_hp += 2;
                    }
                } else {
                    // Build slices for the needed neighbors
                    let a_row = &y_plane[y0p * width..y0p * width + width];
                    let d_row = &y_plane[y1p * width..y1p * width + width];
                    let out_row = &mut predicted_y[dst_row_base..dst_row_base + BLOCK_SIZE];
                    let a = &a_row[x0p..x0p + BLOCK_SIZE];
                    if !y_odd && !x_odd {
                        // Direct copy
                        out_row.copy_from_slice(a);
                    } else if !y_odd && x_odd {
                        // Horizontal avg: (a+b+1)/2 where b = a shifted by +1
                        let b = &a_row[x0p + 1..x0p + 1 + BLOCK_SIZE];
                        avg2_u8(a, b, out_row);
                    } else if y_odd && !x_odd {
                        // Vertical avg: (a+d+1)/2
                        let d = &d_row[x0p..x0p + BLOCK_SIZE];
                        avg2_u8(a, d, out_row);
                    } else {
                        // Both odd: avg of four neighbors
                        let b = &a_row[x0p + 1..x0p + 1 + BLOCK_SIZE];
                        let d = &d_row[x0p..x0p + BLOCK_SIZE];
                        let e = &d_row[x0p + 1..x0p + 1 + BLOCK_SIZE];
                        avg4_u8(a, b, d, e, out_row);
                    }
                }

                // Fast chroma: vectorize interior; fallback to scalar sampling at edges
                if (y_usize & 1) == 0 {
                    // RAW (unclamped) chroma half-pel positions mirror reference behavior
                    let rx_hp_raw0 = x0 * 2 + mv_dx_hp;
                    let ry_hp_raw = y * 2 + mv_dy_hp;
                    let crx_hp0 = rx_hp_raw0 >> 1; // chroma half-pel x
                    let cry_hp = ry_hp_raw >> 1;   // chroma half-pel y
                    let cx0 = crx_hp0 >> 1;
                    let cy0 = cry_hp >> 1;
                    let x_odd_c = (crx_hp0 & 1) != 0;
                    let y_odd_c = (cry_hp & 1) != 0;

                    let chroma_x_dst = (x0 as usize) / 2;
                    let chroma_y_dst = y_usize / 2;

                    let need_right = if x_odd_c { 1 } else { 0 };
                    let need_down = if y_odd_c { 1 } else { 0 };

                    let in_bounds = cx0 >= 0 && cy0 >= 0
                        && (cx0 as usize) + (BLOCK_SIZE / 2) + need_right <= chroma_width
                        && (cy0 as usize) + need_down < chroma_height;

                    if in_bounds {
                        // Operate on 8 chroma samples across row
                        let cw = chroma_width;
                        let src_u_row0 = &u_plane[(cy0 as usize) * cw..];
                        let src_u_row1 = if y_odd_c { &u_plane[((cy0 as usize) + 1) * cw..] } else { &u_plane[(cy0 as usize) * cw..] };
                        let src_v_row0 = &v_plane[(cy0 as usize) * cw..];
                        let src_v_row1 = if y_odd_c { &v_plane[((cy0 as usize) + 1) * cw..] } else { &v_plane[(cy0 as usize) * cw..] };

                        let a_u = &src_u_row0[cx0 as usize..cx0 as usize + (BLOCK_SIZE / 2)];
                        let a_v = &src_v_row0[cx0 as usize..cx0 as usize + (BLOCK_SIZE / 2)];
                        let dst_u = &mut predicted_u[chroma_y_dst * cw + chroma_x_dst..chroma_y_dst * cw + chroma_x_dst + (BLOCK_SIZE / 2)];
                        let dst_v = &mut predicted_v[chroma_y_dst * cw + chroma_x_dst..chroma_y_dst * cw + chroma_x_dst + (BLOCK_SIZE / 2)];

                        if !x_odd_c && !y_odd_c {
                            // Copy
                            dst_u.copy_from_slice(a_u);
                            dst_v.copy_from_slice(a_v);
                        } else if x_odd_c && !y_odd_c {
                            // Horizontal avg
                            let b_u = &src_u_row0[cx0 as usize + 1..cx0 as usize + 1 + (BLOCK_SIZE / 2)];
                            let b_v = &src_v_row0[cx0 as usize + 1..cx0 as usize + 1 + (BLOCK_SIZE / 2)];
                            avg2_u8(a_u, b_u, dst_u);
                            avg2_u8(a_v, b_v, dst_v);
                        } else if !x_odd_c && y_odd_c {
                            // Vertical avg
                            let d_u = &src_u_row1[cx0 as usize..cx0 as usize + (BLOCK_SIZE / 2)];
                            let d_v = &src_v_row1[cx0 as usize..cx0 as usize + (BLOCK_SIZE / 2)];
                            avg2_u8(a_u, d_u, dst_u);
                            avg2_u8(a_v, d_v, dst_v);
                        } else {
                            // Both odd: four-tap average
                            let b_u = &src_u_row0[cx0 as usize + 1..cx0 as usize + 1 + (BLOCK_SIZE / 2)];
                            let b_v = &src_v_row0[cx0 as usize + 1..cx0 as usize + 1 + (BLOCK_SIZE / 2)];
                            let d_u = &src_u_row1[cx0 as usize..cx0 as usize + (BLOCK_SIZE / 2)];
                            let d_v = &src_v_row1[cx0 as usize..cx0 as usize + (BLOCK_SIZE / 2)];
                            let e_u = &src_u_row1[cx0 as usize + 1..cx0 as usize + 1 + (BLOCK_SIZE / 2)];
                            let e_v = &src_v_row1[cx0 as usize + 1..cx0 as usize + 1 + (BLOCK_SIZE / 2)];
                            avg4_u8(a_u, b_u, d_u, e_u, dst_u);
                            avg4_u8(a_v, b_v, d_v, e_v, dst_v);
                        }
                    } else {
                        // Edge fallback: sample per 2x2 area with clamping
                        for xx in (0..BLOCK_SIZE as usize).step_by(2) {
                            let x = (x0 as usize) + xx;
                            let chroma_x = x / 2;
                            let chroma_y = y_usize / 2;
                            let (u, v) = crate::motion::sample_uv_halfpel(
                                u_plane,
                                v_plane,
                                chroma_width,
                                chroma_height,
                                (rx_hp_raw0 + (xx as i32) * 2) >> 1,
                                ry_hp_raw >> 1,
                            );
                            let chroma_idx = chroma_y * chroma_width + chroma_x;
                            predicted_u[chroma_idx] = u;
                            predicted_v[chroma_idx] = v;
                        }
                    }
                }
            }
        }
    }

    Yuv420Frame::from_planes(width, height, predicted_y, predicted_u, predicted_v)
        .expect("fast predicted YUV420 frame")
}
