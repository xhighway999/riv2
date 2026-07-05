//! In-loop deblocking filter (format v5+).
//!
//! Applied to the reconstructed YUV frame on BOTH encoder and decoder side,
//! before the frame is stored as the motion-compensation reference. The filter
//! is fully derived from data both sides already share (frame quality byte +
//! reconstructed MV/skip field), so nothing is transmitted.
//!
//! Algorithm: the VP8 "normal loop filter" for macroblock edges (RFC 6386
//! §15.2/§15.3). For each block-grid edge, up to four pixels on each side
//! (p3..p0 | q0..q3) are examined. An edge is filtered only when the step
//! across it is small (quantization noise, not content) AND both sides are
//! locally smooth (interior limit). Locally sharp neighborhoods (high edge
//! variance, "hev") get only the narrow p0/q0 correction; smooth ones get a
//! wide taper across p2..q2 with weights 27/18/9 ÷ 128, which is what removes
//! visible block seams at coarse quantization. Luma is filtered on the 16px
//! macroblock grid, chroma on its corresponding 8px grid.
//!
//! Blocks that are pure copies of the previous frame (skip + zero MV) already
//! carry filtered content; re-filtering them every frame would progressively
//! blur static areas, so their edges are excluded via the block mask.

use crate::Yuv420Frame;

/// Macroblock size in luma pixels; the chroma grid is half of this.
const MB: usize = 16;

/// Scale from the frame's dequantization step to the VP8 filter level (0..=63).
/// Tuned on the bench footage set; see SPEC §14.
const LEVEL_SCALE: f32 = 0.6;

/// Derive the loop-filter level from the dequantization step of the frame.
/// Larger quant steps produce larger false steps across block edges, so the
/// filter gets proportionally more aggressive. Level 0 disables the filter.
pub fn deblock_level_from_quant_step(quant_step: f32) -> i32 {
    (quant_step * LEVEL_SCALE).round().clamp(0.0, 63.0) as i32
}

/// RFC 6386 threshold derivation for macroblock edges, sharpness = 0.
/// Returns (edge_limit, interior_limit, hev_threshold).
fn thresholds(level: i32, intra: bool) -> (i32, i32, i32) {
    let interior = level.max(1);
    let edge = (level + 2) * 2 + interior;
    let hev = if intra {
        match level {
            l if l >= 40 => 2,
            l if l >= 15 => 1,
            _ => 0,
        }
    } else {
        match level {
            l if l >= 40 => 3,
            l if l >= 20 => 2,
            l if l >= 15 => 1,
            _ => 0,
        }
    };
    (edge, interior, hev)
}

#[inline(always)]
fn c(x: i32) -> i32 {
    x.clamp(-128, 127)
}

/// One edge-crossing line of 8 pixels: `px[0..4]` = p3..p0, `px[4..8]` = q0..q3.
/// Filters in place per RFC 6386 §15.3 (macroblock-edge normal filter).
#[inline(always)]
fn mb_filter_line(px: &mut [i32; 8], edge_limit: i32, interior_limit: i32, hev_threshold: i32) {
    let (p3, p2, p1, p0, q0, q1, q2, q3) =
        (px[0], px[1], px[2], px[3], px[4], px[5], px[6], px[7]);

    // Filter mask: small step across the edge AND locally smooth on both sides.
    let mask = 2 * (p0 - q0).abs() + (q1 - p1).abs() / 2 <= edge_limit
        && (p3 - p2).abs() <= interior_limit
        && (p2 - p1).abs() <= interior_limit
        && (p1 - p0).abs() <= interior_limit
        && (q1 - q0).abs() <= interior_limit
        && (q2 - q1).abs() <= interior_limit
        && (q3 - q2).abs() <= interior_limit;
    if !mask {
        return;
    }

    // Work on signed-biased values like the reference implementation.
    let (ps2, ps1, ps0, qs0, qs1, qs2) =
        (p2 - 128, p1 - 128, p0 - 128, q0 - 128, q1 - 128, q2 - 128);

    let hev = (p1 - p0).abs() > hev_threshold || (q1 - q0).abs() > hev_threshold;
    let w = c(c(ps1 - qs1) + 3 * (qs0 - ps0));

    if hev {
        // Locally sharp: narrow correction of the edge pixels only.
        let f1 = c(w + 4) >> 3;
        let f2 = c(w + 3) >> 3;
        px[4] = (c(qs0 - f1) + 128).clamp(0, 255);
        px[3] = (c(ps0 + f2) + 128).clamp(0, 255);
    } else {
        // Smooth: wide taper across three pixels each side.
        let a = c((27 * w + 63) >> 7);
        let qs0 = c(qs0 - a);
        let ps0 = c(ps0 + a);
        let a = c((18 * w + 63) >> 7);
        let qs1 = c(qs1 - a);
        let ps1 = c(ps1 + a);
        let a = c((9 * w + 63) >> 7);
        let qs2 = c(qs2 - a);
        let ps2 = c(ps2 + a);
        px[1] = (ps2 + 128).clamp(0, 255);
        px[2] = (ps1 + 128).clamp(0, 255);
        px[3] = (ps0 + 128).clamp(0, 255);
        px[4] = (qs0 + 128).clamp(0, 255);
        px[5] = (qs1 + 128).clamp(0, 255);
        px[6] = (qs2 + 128).clamp(0, 255);
    }
}

#[inline(always)]
fn edge_active(mask: Option<&[bool]>, blocks_w: usize, a_bx: usize, a_by: usize, b_bx: usize, b_by: usize) -> bool {
    match mask {
        None => true,
        Some(m) => m[a_by * blocks_w + a_bx] || m[b_by * blocks_w + b_bx],
    }
}

/// Filter all interior block-grid edges of one plane.
///
/// `grid` is the block size in this plane's pixels (16 for Y, 8 for chroma —
/// both map 1:1 onto the macroblock grid, so `mask` indexing is shared).
fn deblock_plane(
    plane: &mut [u8],
    w: usize,
    h: usize,
    grid: usize,
    limits: (i32, i32, i32),
    mask: Option<&[bool]>,
) {
    let (e, i_lim, hev) = limits;
    let blocks_w = w / grid;
    let mut px = [0i32; 8];
    // Vertical edges (left boundary of each block column).
    let mut x = grid;
    while x < w {
        let bx = x / grid;
        for y in 0..h {
            if !edge_active(mask, blocks_w, bx - 1, y / grid, bx, y / grid) {
                continue;
            }
            let i = y * w + x;
            for (k, p) in px.iter_mut().enumerate() {
                *p = plane[i - 4 + k] as i32;
            }
            mb_filter_line(&mut px, e, i_lim, hev);
            for (k, p) in px.iter().enumerate() {
                plane[i - 4 + k] = *p as u8;
            }
        }
        x += grid;
    }
    // Horizontal edges (top boundary of each block row).
    let mut y = grid;
    while y < h {
        let by = y / grid;
        for x in 0..w {
            if !edge_active(mask, blocks_w, x / grid, by - 1, x / grid, by) {
                continue;
            }
            for (k, p) in px.iter_mut().enumerate() {
                *p = plane[(y - 4 + k) * w + x] as i32;
            }
            mb_filter_line(&mut px, e, i_lim, hev);
            for (k, p) in px.iter().enumerate() {
                plane[(y - 4 + k) * w + x] = *p as u8;
            }
        }
        y += grid;
    }
}

/// Deblock a reconstructed YUV420 frame in place.
///
/// `level` is the VP8 loop-filter level (0..=63; 0 = no-op), `intra` selects
/// the keyframe hev-threshold table. `block_mask`, if given, has one entry per
/// 16×16 macroblock (row-major, `(w/16)*(h/16)` entries); an edge is filtered
/// iff either adjacent block is `true`. `None` filters every interior edge
/// (intra frames).
///
/// Normative: encoder and decoder MUST call this with identical inputs, or the
/// prediction references drift.
pub fn deblock_yuv420(frame: &mut Yuv420Frame, level: i32, intra: bool, block_mask: Option<&[bool]>) {
    if level <= 0 {
        return;
    }
    let w = frame.width();
    let h = frame.height();
    if let Some(m) = block_mask {
        debug_assert_eq!(m.len(), (w / MB) * (h / MB), "block mask len mismatch");
    }
    let limits = thresholds(level, intra);
    deblock_plane(frame.y_plane_mut(), w, h, MB, limits, block_mask);
    deblock_plane(frame.u_plane_mut(), w / 2, h / 2, MB / 2, limits, block_mask);
    deblock_plane(frame.v_plane_mut(), w / 2, h / 2, MB / 2, limits, block_mask);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(w: usize, h: usize, left: u8, right: u8) -> Yuv420Frame {
        // Left half `left`, right half `right` → step exactly on the x=w/2 block edge.
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                y[row * w + col] = if col < w / 2 { left } else { right };
            }
        }
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        Yuv420Frame::from_planes(w, h, y, u, v).unwrap()
    }

    #[test]
    fn small_step_is_smoothed_wide() {
        let mut f = flat_frame(32, 32, 100, 106);
        deblock_yuv420(&mut f, 20, false, None);
        let y = f.y_plane();
        // Flat sides → !hev → wide taper: p0/q0 move most, p1/q1 and p2/q2 too.
        assert!(y[15] > 100, "p0 should move up, got {}", y[15]);
        assert!(y[16] < 106, "q0 should move down, got {}", y[16]);
        assert!(y[14] >= 100 && y[14] <= y[15], "p1 tapered, got {}", y[14]);
        assert!(y[17] <= 106 && y[17] >= y[16], "q1 tapered, got {}", y[17]);
        // Pixels beyond the taper untouched.
        assert_eq!(y[8], 100);
        assert_eq!(y[24], 106);
    }

    #[test]
    fn large_step_is_preserved() {
        let mut f = flat_frame(32, 32, 40, 200);
        deblock_yuv420(&mut f, 20, false, None);
        let y = f.y_plane();
        assert_eq!(y[15], 40, "real content edge must not be filtered");
        assert_eq!(y[16], 200);
    }

    #[test]
    fn rough_interior_blocks_filtering() {
        // Same small edge step, but a busy stripe pattern next to the edge —
        // the interior limit must reject the filter to protect texture.
        let w = 32;
        let mut f = flat_frame(w, 32, 100, 106);
        {
            let y = f.y_plane_mut();
            for row in 0..32 {
                y[row * w + 13] = 160; // |p2-p1| large
            }
        }
        deblock_yuv420(&mut f, 8, false, None);
        let y = f.y_plane();
        assert_eq!(y[15], 100, "textured neighborhood must not be filtered");
        assert_eq!(y[16], 106);
    }

    #[test]
    fn masked_static_blocks_are_untouched() {
        let mut f = flat_frame(32, 32, 100, 106);
        // 2x2 macroblocks, all static → nothing filtered.
        let mask = vec![false; 4];
        deblock_yuv420(&mut f, 20, false, Some(&mask));
        assert_eq!(f.y_plane()[15], 100);
        assert_eq!(f.y_plane()[16], 106);
    }

    #[test]
    fn edge_filtered_if_either_side_coded() {
        let mut f = flat_frame(32, 32, 100, 106);
        // Only the right column of macroblocks was coded.
        let mask = vec![false, true, false, true];
        deblock_yuv420(&mut f, 20, false, Some(&mask));
        assert!(f.y_plane()[15] > 100);
        assert!(f.y_plane()[16] < 106);
    }

    #[test]
    fn zero_level_is_noop() {
        let mut f = flat_frame(32, 32, 100, 106);
        let before = f.y_plane().to_vec();
        deblock_yuv420(&mut f, 0, false, None);
        assert_eq!(f.y_plane(), &before[..]);
    }
}
