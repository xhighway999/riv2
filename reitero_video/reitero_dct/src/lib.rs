// ============================================================================
// Common constants and helpers
// ============================================================================

pub const MAX_BLOCK_SIZE: usize = 32;
pub const MAX_BLOCK_AREA: usize = MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;

use wide::i32x8;

const CONST_SCALE: u32 = 18;
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
const IDCT_ROWS_8: [i32x8; 8] = [
    i32x8::new([92682, 92682, 92682, 92682, 92682, 92682, 92682, 92682]),
    i32x8::new([
        128553, 108982, 72820, 25571, -25571, -72820, -108982, -128553,
    ]),
    i32x8::new([
        121095, 50159, -50159, -121095, -121095, -50159, 50159, 121095,
    ]),
    i32x8::new([
        108982, -25571, -128553, -72820, 72820, 128553, 25571, -108982,
    ]),
    i32x8::new([92682, -92682, -92682, 92682, 92682, -92682, -92682, 92682]),
    i32x8::new([
        72820, -128553, 25571, 108982, -108982, -25571, 128553, -72820,
    ]),
    i32x8::new([
        50159, -121095, 121095, -50159, -50159, 121095, -121095, 50159,
    ]),
    i32x8::new([
        25571, -72820, 108982, -128553, 128553, -108982, 72820, -25571,
    ]),
];
const FDCT_TRANSPOSE_16: [[i32x8; 2]; 16] = [
    [
        i32x8::new([65536, 92236, 90901, 88691, 85627, 81738, 77062, 71644]),
        i32x8::new([65536, 58797, 51491, 43690, 35468, 26904, 18081, 9084]),
    ],
    [
        i32x8::new([65536, 88691, 77062, 58797, 35468, 9084, -18081, -43690]),
        i32x8::new([
            -65536, -81738, -90901, -92236, -85627, -71644, -51491, -26904,
        ]),
    ],
    [
        i32x8::new([65536, 81738, 51491, 9084, -35468, -71644, -90901, -88691]),
        i32x8::new([-65536, -26904, 18081, 58797, 85627, 92236, 77062, 43690]),
    ],
    [
        i32x8::new([65536, 71644, 18081, -43690, -85627, -88691, -51491, 9084]),
        i32x8::new([65536, 92236, 77062, 26904, -35468, -81738, -90901, -58797]),
    ],
    [
        i32x8::new([65536, 58797, -18081, -81738, -85627, -26904, 51491, 92236]),
        i32x8::new([65536, -9084, -77062, -88691, -35468, 43690, 90901, 71644]),
    ],
    [
        i32x8::new([65536, 43690, -51491, -92236, -35468, 58797, 90901, 26904]),
        i32x8::new([-65536, -88691, -18081, 71644, 85627, 9084, -77062, -81738]),
    ],
    [
        i32x8::new([65536, 26904, -77062, -71644, 35468, 92236, 18081, -81738]),
        i32x8::new([-65536, 43690, 90901, 9084, -85627, -58797, 51491, 88691]),
    ],
    [
        i32x8::new([65536, 9084, -90901, -26904, 85627, 43690, -77062, -58797]),
        i32x8::new([65536, 71644, -51491, -81738, 35468, 88691, -18081, -92236]),
    ],
    [
        i32x8::new([65536, -9084, -90901, 26904, 85627, -43690, -77062, 58797]),
        i32x8::new([65536, -71644, -51491, 81738, 35468, -88691, -18081, 92236]),
    ],
    [
        i32x8::new([65536, -26904, -77062, 71644, 35468, -92236, 18081, 81738]),
        i32x8::new([-65536, -43690, 90901, -9084, -85627, 58797, 51491, -88691]),
    ],
    [
        i32x8::new([65536, -43690, -51491, 92236, -35468, -58797, 90901, -26904]),
        i32x8::new([-65536, 88691, -18081, -71644, 85627, -9084, -77062, 81738]),
    ],
    [
        i32x8::new([65536, -58797, -18081, 81738, -85627, 26904, 51491, -92236]),
        i32x8::new([65536, 9084, -77062, 88691, -35468, -43690, 90901, -71644]),
    ],
    [
        i32x8::new([65536, -71644, 18081, 43690, -85627, 88691, -51491, -9084]),
        i32x8::new([65536, -92236, 77062, -26904, -35468, 81738, -90901, 58797]),
    ],
    [
        i32x8::new([65536, -81738, 51491, -9084, -35468, 71644, -90901, 88691]),
        i32x8::new([-65536, 26904, 18081, -58797, 85627, -92236, 77062, -43690]),
    ],
    [
        i32x8::new([65536, -88691, 77062, -58797, 35468, -9084, -18081, 43690]),
        i32x8::new([-65536, 81738, -90901, 92236, -85627, 71644, -51491, 26904]),
    ],
    [
        i32x8::new([65536, -92236, 90901, -88691, 85627, -81738, 77062, -71644]),
        i32x8::new([65536, -58797, 51491, -43690, 35468, -26904, 18081, -9084]),
    ],
];
const IDCT_ROWS_16: [[i32x8; 2]; 16] = [
    [
        i32x8::new([65536, 65536, 65536, 65536, 65536, 65536, 65536, 65536]),
        i32x8::new([65536, 65536, 65536, 65536, 65536, 65536, 65536, 65536]),
    ],
    [
        i32x8::new([92236, 88691, 81738, 71644, 58797, 43690, 26904, 9084]),
        i32x8::new([
            -9084, -26904, -43690, -58797, -71644, -81738, -88691, -92236,
        ]),
    ],
    [
        i32x8::new([90901, 77062, 51491, 18081, -18081, -51491, -77062, -90901]),
        i32x8::new([-90901, -77062, -51491, -18081, 18081, 51491, 77062, 90901]),
    ],
    [
        i32x8::new([88691, 58797, 9084, -43690, -81738, -92236, -71644, -26904]),
        i32x8::new([26904, 71644, 92236, 81738, 43690, -9084, -58797, -88691]),
    ],
    [
        i32x8::new([85627, 35468, -35468, -85627, -85627, -35468, 35468, 85627]),
        i32x8::new([85627, 35468, -35468, -85627, -85627, -35468, 35468, 85627]),
    ],
    [
        i32x8::new([81738, 9084, -71644, -88691, -26904, 58797, 92236, 43690]),
        i32x8::new([-43690, -92236, -58797, 26904, 88691, 71644, -9084, -81738]),
    ],
    [
        i32x8::new([77062, -18081, -90901, -51491, 51491, 90901, 18081, -77062]),
        i32x8::new([-77062, 18081, 90901, 51491, -51491, -90901, -18081, 77062]),
    ],
    [
        i32x8::new([71644, -43690, -88691, 9084, 92236, 26904, -81738, -58797]),
        i32x8::new([58797, 81738, -26904, -92236, -9084, 88691, 43690, -71644]),
    ],
    [
        i32x8::new([65536, -65536, -65536, 65536, 65536, -65536, -65536, 65536]),
        i32x8::new([65536, -65536, -65536, 65536, 65536, -65536, -65536, 65536]),
    ],
    [
        i32x8::new([58797, -81738, -26904, 92236, -9084, -88691, 43690, 71644]),
        i32x8::new([-71644, -43690, 88691, 9084, -92236, 26904, 81738, -58797]),
    ],
    [
        i32x8::new([51491, -90901, 18081, 77062, -77062, -18081, 90901, -51491]),
        i32x8::new([-51491, 90901, -18081, -77062, 77062, 18081, -90901, 51491]),
    ],
    [
        i32x8::new([43690, -92236, 58797, 26904, -88691, 71644, 9084, -81738]),
        i32x8::new([81738, -9084, -71644, 88691, -26904, -58797, 92236, -43690]),
    ],
    [
        i32x8::new([35468, -85627, 85627, -35468, -35468, 85627, -85627, 35468]),
        i32x8::new([35468, -85627, 85627, -35468, -35468, 85627, -85627, 35468]),
    ],
    [
        i32x8::new([26904, -71644, 92236, -81738, 43690, 9084, -58797, 88691]),
        i32x8::new([-88691, 58797, -9084, -43690, 81738, -92236, 71644, -26904]),
    ],
    [
        i32x8::new([18081, -51491, 77062, -90901, 90901, -77062, 51491, -18081]),
        i32x8::new([-18081, 51491, -77062, 90901, -90901, 77062, -51491, 18081]),
    ],
    [
        i32x8::new([9084, -26904, 43690, -58797, 71644, -81738, 88691, -92236]),
        i32x8::new([92236, -88691, 81738, -71644, 58797, -43690, 26904, -9084]),
    ],
];
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
    let round = 1i64 << (CONST_SCALE as i64 - 1);
    let mut out_arr: [i32; 8] = [0; 8];
    for j in 0..8 {
        out_arr[j] = ((accum[j] + round) >> CONST_SCALE as i64)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    i32x8::new(out_arr)
}

fn idct_1d_8(input: i32x8) -> i32x8 {
    let mut accum = i32x8::splat(0);
    let vals = input.to_array();
    for i in 0..8 {
        accum = accum + i32x8::splat(vals[i]) * IDCT_ROWS_8[i];
    }
    let round = i32x8::splat(1 << (CONST_SCALE as i32 - 1));
    (accum + round) >> CONST_SCALE as i32
}

fn fdct_1d_16(input: [i32x8; 2]) -> [i32x8; 2] {
    let low_vals = input[0].to_array();
    let high_vals = input[1].to_array();
    let vals: [i32; 16] = [
        low_vals[0],
        low_vals[1],
        low_vals[2],
        low_vals[3],
        low_vals[4],
        low_vals[5],
        low_vals[6],
        low_vals[7],
        high_vals[0],
        high_vals[1],
        high_vals[2],
        high_vals[3],
        high_vals[4],
        high_vals[5],
        high_vals[6],
        high_vals[7],
    ];
    let mut accum: [i64; 16] = [0; 16];
    for i in 0..16 {
        let basis_low = FDCT_TRANSPOSE_16[i][0].to_array();
        let basis_high = FDCT_TRANSPOSE_16[i][1].to_array();
        let v = vals[i] as i64;
        for j in 0..8 {
            accum[j] += v * (basis_low[j] as i64);
            accum[j + 8] += v * (basis_high[j] as i64);
        }
    }
    let round = 1i64 << (CONST_SCALE as i64 - 1);
    let mut out_low: [i32; 8] = [0; 8];
    let mut out_high: [i32; 8] = [0; 8];
    for j in 0..8 {
        out_low[j] = ((accum[j] + round) >> CONST_SCALE as i64)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        out_high[j] = ((accum[j + 8] + round) >> CONST_SCALE as i64)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    [i32x8::new(out_low), i32x8::new(out_high)]
}

fn idct_1d_16(input: [i32x8; 2]) -> [i32x8; 2] {
    let low_arr = input[0].to_array();
    let high_arr = input[1].to_array();
    let vals: [i32; 16] = [
        low_arr[0],
        low_arr[1],
        low_arr[2],
        low_arr[3],
        low_arr[4],
        low_arr[5],
        low_arr[6],
        low_arr[7],
        high_arr[0],
        high_arr[1],
        high_arr[2],
        high_arr[3],
        high_arr[4],
        high_arr[5],
        high_arr[6],
        high_arr[7],
    ];
    let mut accum_low = i32x8::splat(0);
    let mut accum_high = i32x8::splat(0);
    for i in 0..16 {
        let s = i32x8::splat(vals[i]);
        accum_low = accum_low + s * IDCT_ROWS_16[i][0];
        accum_high = accum_high + s * IDCT_ROWS_16[i][1];
    }
    let round = i32x8::splat(1 << (CONST_SCALE as i32 - 1));
    let out_low = (accum_low + round) >> CONST_SCALE as i32;
    let out_high = (accum_high + round) >> CONST_SCALE as i32;
    [out_low, out_high]
}

fn transpose_8(mat: [i32x8; 8]) -> [i32x8; 8] {
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

fn transpose_16(mat: [[i32x8; 2]; 16]) -> [[i32x8; 2]; 16] {
    let mut in_arrays: [[i32; 16]; 16] = [[0; 16]; 16];
    for i in 0..16 {
        let low = mat[i][0].to_array();
        let high = mat[i][1].to_array();
        for j in 0..8 {
            in_arrays[i][j] = low[j];
            in_arrays[i][j + 8] = high[j];
        }
    }
    let mut out_arrays: [[i32; 16]; 16] = [[0; 16]; 16];
    for i in 0..16 {
        for j in 0..16 {
            out_arrays[i][j] = in_arrays[j][i];
        }
    }
    let mut out: [[i32x8; 2]; 16] = [[i32x8::splat(0); 2]; 16];
    for i in 0..16 {
        let arr = out_arrays[i];
        let mut low_arr: [i32; 8] = [0; 8];
        let mut high_arr: [i32; 8] = [0; 8];
        for j in 0..8 {
            low_arr[j] = arr[j];
            high_arr[j] = arr[j + 8];
        }
        out[i][0] = i32x8::new(low_arr);
        out[i][1] = i32x8::new(high_arr);
    }
    out
}

pub fn encode_plane_8x8(
    plane: &[i16],
    stride: usize,
    width: usize,
    height: usize,
    quant_step: f32,
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
                // Row FDCT
                for i in 0..8 {
                    row_vecs[i] = fdct_1d_8(row_vecs[i]);
                }
                // Transpose
                let mut col_vecs = transpose_8(row_vecs);
                // Column FDCT
                for i in 0..8 {
                    col_vecs[i] = fdct_1d_8(col_vecs[i]);
                }
                // Transpose back
                let final_vecs = transpose_8(col_vecs);
                // Quantize and collect
                let mut coeffs: Vec<i16> = Vec::with_capacity(64);
                for v in 0..8 {
                    let row = final_vecs[v].to_array();
                    for u in 0..8 {
                        let q = (row[u] as f32 / quant_step).round() as i16;
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

pub fn decode_plane_8x8(
    coeffs: &[i16],
    output_plane: &mut [i16],
    stride: usize,
    width: usize,
    height: usize,
    quant_step: f32,
    skip_mask: &[bool],
) {
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let mut coeff_idx = 0;
    let mut mask_idx = 0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            if skip_mask[mask_idx] {
                // Zero out block
                for y in 0..8 {
                    let offset = (by * 8 + y) * stride + bx * 8;
                    for x in 0..8 {
                        output_plane[offset + x] = 0;
                    }
                }
                coeff_idx += 64;
            } else {
                let mut coeff_vecs: [i32x8; 8] = [i32x8::splat(0); 8];
                // Use fixed-point quant multiplication to avoid per-coefficient f32 ops
                let q_fp: i32 = (quant_step * (1<<14) as f32).round() as i32;
                let q_round = 1 << 13;
                for v in 0..8 {
                    let mut arr: [i32; 8] = [0; 8];
                    for u in 0..8 {
                        arr[u] = ((coeffs[coeff_idx] as i32) * q_fp + q_round) >> 14;
                        coeff_idx += 1;
                    }
                    coeff_vecs[v] = i32x8::new(arr);
                }
                // Row IDCT (horizontal)
                for i in 0..8 {
                    coeff_vecs[i] = idct_1d_8(coeff_vecs[i]);
                }
                // Transpose
                let mut col_vecs = transpose_8(coeff_vecs);
                // Column IDCT (vertical)
                for i in 0..8 {
                    col_vecs[i] = idct_1d_8(col_vecs[i]);
                }
                // Transpose back
                let final_vecs = transpose_8(col_vecs);
                // Write to output
                for y in 0..8 {
                    let row = final_vecs[y].to_array();
                    let offset = (by * 8 + y) * stride + bx * 8;
                    for x in 0..8 {
                        let val = row[x];
                        output_plane[offset + x] =
                            val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                }
            }
            mask_idx += 1;
        }
    }
}

pub fn encode_plane_16x16(
    plane: &[i16],
    stride: usize,
    width: usize,
    height: usize,
    quant_step: f32,
    skip_mask: Option<&[bool]>,
) -> Vec<Option<Vec<i16>>> {
    let blocks_x = width / 16;
    let blocks_y = height / 16;
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
                let mut row_vecs: [[i32x8; 2]; 16] = [[i32x8::splat(0); 2]; 16];
                for y in 0..16 {
                    let offset = (by * 16 + y) * stride + bx * 16;
                    let mut arr: [i32; 16] = [0; 16];
                    for x in 0..16 {
                        arr[x] = plane[offset + x] as i32;
                    }
                    row_vecs[y][0] = i32x8::new([
                        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
                    ]);
                    row_vecs[y][1] = i32x8::new([
                        arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15],
                    ]);
                }
                // Row FDCT
                for i in 0..16 {
                    row_vecs[i] = fdct_1d_16(row_vecs[i]);
                }
                // Transpose
                let mut col_vecs = transpose_16(row_vecs);
                // Column FDCT
                for i in 0..16 {
                    col_vecs[i] = fdct_1d_16(col_vecs[i]);
                }
                // Transpose back
                let final_vecs = transpose_16(col_vecs);
                // Quantize and collect
                let mut coeffs: Vec<i16> = Vec::with_capacity(256);
                for v in 0..16 {
                    let row_low = final_vecs[v][0].to_array();
                    let row_high = final_vecs[v][1].to_array();
                    for u in 0..8 {
                        let q = (row_low[u] as f32 / quant_step).round() as i16;
                        coeffs.push(q);
                    }
                    for u in 0..8 {
                        let q = (row_high[u] as f32 / quant_step).round() as i16;
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

pub fn decode_plane_16x16(
    coeffs: &[i16],
    output_plane: &mut [i16],
    stride: usize,
    width: usize,
    height: usize,
    quant_step: f32,
    skip_mask: &[bool],
) {
    let blocks_x = width / 16;
    let blocks_y = height / 16;
    let mut coeff_idx = 0;
    let mut mask_idx = 0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            if skip_mask[mask_idx] {
                // Zero out block
                for y in 0..16 {
                    let offset = (by * 16 + y) * stride + bx * 16;
                    for x in 0..16 {
                        output_plane[offset + x] = 0;
                    }
                }
                coeff_idx += 256;
            } else {
                let mut coeff_vecs: [[i32x8; 2]; 16] = [[i32x8::splat(0); 2]; 16];
                // Fixed-point quant multiply for 16x16 blocks
                let q_fp: i32 = (quant_step * (1<<14) as f32).round() as i32;
                let q_round = 1 << 13;
                for v in 0..16 {
                    let mut arr: [i32; 16] = [0; 16];
                    for u in 0..16 {
                        arr[u] = ((coeffs[coeff_idx] as i32) * q_fp + q_round) >> 14;
                        coeff_idx += 1;
                    }
                    coeff_vecs[v][0] = i32x8::new([
                        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
                    ]);
                    coeff_vecs[v][1] = i32x8::new([
                        arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15],
                    ]);
                }
                // Row IDCT (horizontal)
                for i in 0..16 {
                    coeff_vecs[i] = idct_1d_16(coeff_vecs[i]);
                }
                // Transpose
                let mut col_vecs = transpose_16(coeff_vecs);
                // Column IDCT (vertical)
                for i in 0..16 {
                    col_vecs[i] = idct_1d_16(col_vecs[i]);
                }
                // Transpose back
                let final_vecs = transpose_16(col_vecs);
                // Write to output
                for y in 0..16 {
                    let row_low = final_vecs[y][0].to_array();
                    let row_high = final_vecs[y][1].to_array();
                    let offset = (by * 16 + y) * stride + bx * 16;
                    for x in 0..8 {
                        let val = row_low[x];
                        output_plane[offset + x] =
                            val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                    for x in 0..8 {
                        let val = row_high[x];
                        output_plane[offset + x + 8] =
                            val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                }
            }
            mask_idx += 1;
        }
    }
}
