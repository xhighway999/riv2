use wide::i32x8;
use crate::common::CONST_SCALE;

/// 8×8 forward DCT transform matrix (rows = frequency basis vectors, columns = spatial
/// samples), scaled by `1 << CONST_SCALE` for fixed‑point arithmetic.
const FDCT_TRANSPOSE_8: [i32x8; 8] = [
    i32x8::new([92682, 128553, 121095, 108982, 92682, 72820, 50159, 25571]),
    i32x8::new([
        92682, 108982, 50159, -25571, -92682, -128553, -121095, -72820,
    ]),
    i32x8::new([92682, 72820, -50159, -128553, -92682, 25571, 121095, 108982]),
    i32x8::new([
        92682, 25571, -121095, -72820, 92682, 108982, -50159, -128553,
    ]),
    i32x8::new([
        92682, -25571, -121095, 72820, 92682, -108982, -50159, 128553,
    ]),
    i32x8::new([
        92682, -72820, -50159, 128553, -92682, -25571, 121095, -108982,
    ]),
    i32x8::new([92682, -108982, 50159, 25571, -92682, 128553, -121095, 72820]),
    i32x8::new([
        92682, -128553, 121095, -108982, 92682, -72820, 50159, -25571,
    ]),
];

/// 1‑D 8‑point forward DCT on a single row/column, using the fixed‑point basis above.
///
/// This uses a "matmul" formulation: each output coefficient is the dot‑product of the
/// input vector with one pre‑scaled basis row from `FDCT_TRANSPOSE_8`. We accumulate
/// in i64 to keep plenty of headroom before shifting back down by `CONST_SCALE`.
fn fdct_1d_8(input: i32x8) -> i32x8 {
    let vals = input.to_array();
    let mut accum: [i64; 8] = [0; 8];
    for i in 0..8 {
        let basis_row = FDCT_TRANSPOSE_8[i].to_array();
        let v = vals[i] as i64;
        for j in 0..8 {
            accum[j] += v * (basis_row[j] as i64);
        }
    }
    // Bias used for rounding after the fixed‑point multiply/accumulate.
    let round = 1i64 << (CONST_SCALE as i64 - 1);
    let mut out_arr: [i32; 8] = [0; 8];
    for j in 0..8 {
        // Shift back down to the original scale and clamp into i32. In practice the
        // DCT value range is far from the i32 limits; the clamp is defensive only.
        out_arr[j] = ((accum[j] + round) >> CONST_SCALE as i64)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    i32x8::new(out_arr)
}

/// Transpose an 8×8 matrix stored as `[i32x8; 8]` (row‑major).
fn transpose_8(mat: [i32x8; 8]) -> [i32x8; 8] {
    // Go through plain arrays to keep the implementation simple and avoid wide‑SIMD
    // shuffle instructions for the transpose. Benchmarking shows this is faster than
    // trying to force the wide crate to behave.
    let mut in_arrays: [[i32; 8]; 8] = [[0; 8]; 8];
    for i in 0..8 {
        in_arrays[i] = mat[i].to_array();
    }
    let mut out_arrays: [[i32; 8]; 8] = [[0; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            out_arrays[i][j] = in_arrays[j][i];
        }
    }
    let mut out: [i32x8; 8] = [i32x8::splat(0); 8];
    for i in 0..8 {
        out[i] = i32x8::new(out_arrays[i]);
    }
    out
}

/// Encode a full plane as 8×8 DCT blocks with per-block quantization steps.
///
/// Each block uses its own scalar quant step from `quant_steps[block_index]`.
/// Used for adaptive quantization.
pub fn encode_plane_8x8_aq(
    plane: &[i16],
    stride: usize,
    width: usize,
    height: usize,
    quant_steps: &[f32],
    skip_mask: Option<&[bool]>,
    dead_zone: f32,
) -> Vec<Option<Vec<i16>>> {
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let num_blocks = blocks_x * blocks_y;
    let skips = skip_mask
        .map(|m| m.to_vec())
        .unwrap_or(vec![false; num_blocks]);
    let mut result: Vec<Option<Vec<i16>>> = Vec::with_capacity(num_blocks);
    let mut mask_idx = 0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            if skips[mask_idx] {
                result.push(None);
            } else {
                let qs = quant_steps[mask_idx];
                let mut row_vecs: [i32x8; 8] = [i32x8::splat(0); 8];
                for y in 0..8 {
                    let offset = (by * 8 + y) * stride + bx * 8;
                    let mut arr: [i32; 8] = [0; 8];
                    for x in 0..8 {
                        arr[x] = plane[offset + x] as i32;
                    }
                    row_vecs[y] = i32x8::new(arr);
                }
                for i in 0..8 {
                    row_vecs[i] = fdct_1d_8(row_vecs[i]);
                }
                let mut col_vecs = transpose_8(row_vecs);
                for i in 0..8 {
                    col_vecs[i] = fdct_1d_8(col_vecs[i]);
                }
                let final_vecs = transpose_8(col_vecs);
                let mut coeffs: Vec<i16> = Vec::with_capacity(64);
                for v in 0..8 {
                    let row = final_vecs[v].to_array();
                    for u in 0..8 {
                        let r = row[u] as f32 / qs;
                        coeffs.push(if r.abs() < dead_zone { 0 } else { r.round() as i16 });
                    }
                }
                result.push(Some(coeffs));
            }
            mask_idx += 1;
        }
    }
    result
}

/// Encode a full plane as 8×8 DCT blocks using a per‑coefficient quantization table.
///
/// - `quant_table[v*8+u]` holds the step for coefficient (u,v); values are clamped
///   to at least 1 to avoid division by zero.
/// - `skip_mask`, if present, marks blocks that should be treated as skipped.
pub fn encode_plane_8x8_matrix(
    plane: &[i16],
    stride: usize,
    width: usize,
    height: usize,
    quant_table: &[u16; 64],
    skip_mask: Option<&[bool]>,
) -> Vec<Option<Vec<i16>>> {
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let num_blocks = blocks_x * blocks_y;
    let skips = skip_mask
        .map(|m| m.to_vec())
        .unwrap_or(vec![false; num_blocks]);
    let mut result: Vec<Option<Vec<i16>>> = Vec::with_capacity(num_blocks);
    let mut mask_idx = 0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            if skips[mask_idx] {
                result.push(None);
            } else {
                let mut row_vecs: [i32x8; 8] = [i32x8::splat(0); 8];
                for y in 0..8 {
                    let offset = (by * 8 + y) * stride + bx * 8;
                    let mut arr: [i32; 8] = [0; 8];
                    for x in 0..8 {
                        arr[x] = plane[offset + x] as i32;
                    }
                    row_vecs[y] = i32x8::new(arr);
                }
                for i in 0..8 {
                    row_vecs[i] = fdct_1d_8(row_vecs[i]);
                }
                let mut col_vecs = transpose_8(row_vecs);
                for i in 0..8 {
                    col_vecs[i] = fdct_1d_8(col_vecs[i]);
                }
                let final_vecs = transpose_8(col_vecs);
                let mut coeffs: Vec<i16> = Vec::with_capacity(64);
                for v in 0..8 {
                    let row = final_vecs[v].to_array();
                    for u in 0..8 {
                        let q_step = quant_table[v * 8 + u].max(1) as f32;
                        let q = (row[u] as f32 / q_step).round() as i16;
                        coeffs.push(q);
                    }
                }
                result.push(Some(coeffs));
            }
            mask_idx += 1;
        }
    }
    result
}

