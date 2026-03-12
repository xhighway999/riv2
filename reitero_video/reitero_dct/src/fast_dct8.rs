//! Fast 8×8 inverse DCT (IDCT) used for decoding.
//!
//! This implementation is **blatantly copied** from a ported version of the [jpeg] crate,
//! which to our knowledge uses [stb_image]'s (stbi's) IDCT code as its base. We keep
//! wrapping arithmetic so that malicious or out-of-range inputs cannot overflow.
//!
//! [jpeg]: https://crates.io/crates/jpeg-decoder
//! [stb_image]: https://github.com/nothings/stb

// Note: we have many values that are straight from a reference.
// Do not warn on them or try to automatically change them.
#![allow(clippy::excessive_precision)]
// Note: consistency for unrolled, scaled offset loops
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
// Some helpers (scalar JPEG path) are only reachable in test builds.
#![allow(dead_code)]
use core::num::Wrapping;

use wide::i32x8;

/// Decode a full plane of 8×8 blocks using the fast (JPEG-style) IDCT with a per-coefficient
/// quantization matrix. Writes i16 (centered at 0); skipped blocks are filled with 0.
pub fn decode_plane_8x8_matrix(
    coeffs: &[i16],
    output_plane: &mut [i16],
    stride: usize,
    width: usize,
    height: usize,
    quant_table: &[u16; 64],
    skip_mask: &[bool],
) {
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let mut coeff_idx = 0;
    let mut mask_idx = 0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_offset = (by * 8) * stride + (bx * 8);
            if skip_mask[mask_idx] {
                for y in 0..8 {
                    let offset = block_offset + y * stride;
                    for x in 0..8 {
                        output_plane[offset + x] = 0;
                    }
                }
                coeff_idx += 64;
            } else {
                let coeff_slice = &coeffs[coeff_idx..coeff_idx + 64];
                coeff_idx += 64;
                let mut block_i16 = [0i16; 64];
                idct_block_8x8_residual(coeff_slice, quant_table, &mut block_i16);
                for y in 0..8 {
                    let out_offset = block_offset + y * stride;
                    output_plane[out_offset..out_offset + 8]
                        .copy_from_slice(&block_i16[y * 8..y * 8 + 8]);
                }
            }
            mask_idx += 1;
        }
    }
}

/// Decode a full plane of 8×8 blocks with per-block quantization steps.
///
/// Each block uses its own scalar quant step from `quant_steps[block_index]`.
/// Used for adaptive quantization.
pub fn decode_plane_8x8_aq(
    coeffs: &[i16],
    output_plane: &mut [i16],
    stride: usize,
    width: usize,
    height: usize,
    quant_steps: &[f32],
    skip_mask: &[bool],
) {
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let mut coeff_idx = 0;
    let mut mask_idx = 0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_offset = (by * 8) * stride + (bx * 8);
            if skip_mask[mask_idx] {
                for y in 0..8 {
                    let offset = block_offset + y * stride;
                    for x in 0..8 {
                        output_plane[offset + x] = 0;
                    }
                }
                coeff_idx += 64;
            } else {
                let q = quant_steps[mask_idx].round().max(1.0) as u16;
                let quantization_table = [q; 64];
                let coeff_slice = &coeffs[coeff_idx..coeff_idx + 64];
                coeff_idx += 64;
                let mut block_i16 = [0i16; 64];
                idct_block_8x8_residual(coeff_slice, &quantization_table, &mut block_i16);
                for y in 0..8 {
                    let out_offset = block_offset + y * stride;
                    output_plane[out_offset..out_offset + 8]
                        .copy_from_slice(&block_i16[y * 8..y * 8 + 8]);
                }
            }
            mask_idx += 1;
        }
    }
}

/// IDCT for residual data: same kernel as the JPEG path but without the +128 level-unshift,
/// writing directly to i16. This allows the full [-255, 255] residual range to round-trip
/// correctly instead of being clamped to [-128, 127] by the u8 output of the JPEG path.
fn idct_block_8x8_residual(
    coefficients: &[i16],
    quantization_table: &[u16; 64],
    output: &mut [i16; 64],
) {
    idct_block_8x8_residual_simd(coefficients, quantization_table, output);
}

fn idct_block_8x8_residual_simd(
    coefficients: &[i16],
    quantization_table: &[u16; 64],
    output: &mut [i16; 64],
) {
    let mut col_vecs: [i32x8; 8] = [i32x8::splat(0); 8];
    for c in 0..8 {
        let mut arr: [i32; 8] = [0; 8];
        for r in 0..8 {
            arr[r] = coefficients[r * 8 + c] as i32 * quantization_table[r * 8 + c] as i32;
        }
        col_vecs[c] = i32x8::new(arr);
    }

    let (xs, ts) = kernel_simd(col_vecs, 512);
    let (x0, x1, x2, x3) = (xs[0], xs[1], xs[2], xs[3]);
    let (t0, t1, t2, t3) = (ts[0], ts[1], ts[2], ts[3]);

    let mut temp: [i32x8; 8] = [i32x8::splat(0); 8];
    temp[0] = (x0 + t3) >> 10;
    temp[7] = (x0 - t3) >> 10;
    temp[1] = (x1 + t2) >> 10;
    temp[6] = (x1 - t2) >> 10;
    temp[2] = (x2 + t1) >> 10;
    temp[5] = (x2 - t1) >> 10;
    temp[3] = (x3 + t0) >> 10;
    temp[4] = (x3 - t0) >> 10;

    let row_vecs = transpose_8x8_simd(temp);

    const X_SCALE: i32 = 65536; // no +128 bias
    let (xs2, ts2) = kernel_simd(row_vecs, X_SCALE);
    let (x0, x1, x2, x3) = (xs2[0], xs2[1], xs2[2], xs2[3]);
    let (t0, t1, t2, t3) = (ts2[0], ts2[1], ts2[2], ts2[3]);

    let rows: [i32x8; 8] = [
        (x0 + t3) >> 17,
        (x1 + t2) >> 17,
        (x2 + t1) >> 17,
        (x3 + t0) >> 17,
        (x3 - t0) >> 17,
        (x2 - t1) >> 17,
        (x1 - t2) >> 17,
        (x0 - t3) >> 17,
    ];
    for (row_vec, out_row) in rows.iter().zip(output.chunks_exact_mut(8)) {
        let arr = row_vec.to_array();
        for (i, &v) in arr.iter().enumerate() {
            out_row[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

struct Kernel {
    xs: [Wrapping<i32>; 4],
    ts: [Wrapping<i32>; 4],
}

#[inline]
fn kernel_x([s0, s2, s4, s6]: [Wrapping<i32>; 4], x_scale: i32) -> [Wrapping<i32>; 4] {
    // Even `chunk` indicies
    let (t2, t3);
    {
        let p2 = s2;
        let p3 = s6;

        let p1 = (p2 + p3) * stbi_f2f(0.5411961);
        t2 = p1 + p3 * stbi_f2f(-1.847759065);
        t3 = p1 + p2 * stbi_f2f(0.765366865);
    }

    let (t0, t1);
    {
        let p2 = s0;
        let p3 = s4;

        t0 = stbi_fsh(p2 + p3);
        t1 = stbi_fsh(p2 - p3);
    }

    let x0 = t0 + t3;
    let x3 = t0 - t3;
    let x1 = t1 + t2;
    let x2 = t1 - t2;

    let x_scale = Wrapping(x_scale);

    [x0 + x_scale, x1 + x_scale, x2 + x_scale, x3 + x_scale]
}

#[inline]
fn kernel_t([s1, s3, s5, s7]: [Wrapping<i32>; 4]) -> [Wrapping<i32>; 4] {
    // Odd `chunk` indicies
    let mut t0 = s7;
    let mut t1 = s5;
    let mut t2 = s3;
    let mut t3 = s1;

    let p3 = t0 + t2;
    let p4 = t1 + t3;
    let p1 = t0 + t3;
    let p2 = t1 + t2;
    let p5 = (p3 + p4) * stbi_f2f(1.175875602);

    t0 *= stbi_f2f(0.298631336);
    t1 *= stbi_f2f(2.053119869);
    t2 *= stbi_f2f(3.072711026);
    t3 *= stbi_f2f(1.501321110);

    let p1 = p5 + p1 * stbi_f2f(-0.899976223);
    let p2 = p5 + p2 * stbi_f2f(-2.562915447);
    let p3 = p3 * stbi_f2f(-1.961570560);
    let p4 = p4 * stbi_f2f(-0.390180644);

    t3 += p1 + p4;
    t2 += p2 + p3;
    t1 += p2 + p4;
    t0 += p1 + p3;

    [t0, t1, t2, t3]
}

#[inline]
fn kernel([s0, s1, s2, s3, s4, s5, s6, s7]: [Wrapping<i32>; 8], x_scale: i32) -> Kernel {
    Kernel {
        xs: kernel_x([s0, s2, s4, s6], x_scale),
        ts: kernel_t([s1, s3, s5, s7]),
    }
}

// SIMD 8x8 IDCT: same AAN kernel with i32x8 (8 columns/rows at once). Constants = stbi_f2f(x)*4096.
const F2F_0541: i32 = 2218;
const F2F_1848: i32 = -7572;
const F2F_0765: i32 = 3136;
const F2F_1176: i32 = 4817;
const F2F_0299: i32 = 1223;
const F2F_2053: i32 = 8411;
const F2F_3073: i32 = 12595;
const F2F_1501: i32 = 6150;
const F2F_0900: i32 = -3685;
const F2F_2563: i32 = -10496;
const F2F_1962: i32 = -8035;
const F2F_0390: i32 = -1598;

#[inline(always)]
fn mul_f2f(a: i32x8, c: i32) -> i32x8 {
    a * i32x8::splat(c)
}

#[inline]
fn kernel_x_simd([s0, s2, s4, s6]: [i32x8; 4], x_scale: i32) -> [i32x8; 4] {
    let p2 = s2;
    let p3 = s6;
    let p1 = mul_f2f(p2 + p3, F2F_0541);
    let t2 = p1 + mul_f2f(p3, F2F_1848);
    let t3 = p1 + mul_f2f(p2, F2F_0765);

    let p2 = s0;
    let p3 = s4;
    let t0 = (p2 + p3) << 12;
    let t1 = (p2 - p3) << 12;

    let x0 = t0 + t3;
    let x3 = t0 - t3;
    let x1 = t1 + t2;
    let x2 = t1 - t2;

    let xs = i32x8::splat(x_scale);
    [x0 + xs, x1 + xs, x2 + xs, x3 + xs]
}

#[inline]
fn kernel_t_simd([s1, s3, s5, s7]: [i32x8; 4]) -> [i32x8; 4] {
    let mut t0 = s7;
    let mut t1 = s5;
    let mut t2 = s3;
    let mut t3 = s1;

    let p3 = t0 + t2;
    let p4 = t1 + t3;
    let p1 = t0 + t3;
    let p2 = t1 + t2;
    let p5 = mul_f2f(p3 + p4, F2F_1176);

    t0 = mul_f2f(t0, F2F_0299);
    t1 = mul_f2f(t1, F2F_2053);
    t2 = mul_f2f(t2, F2F_3073);
    t3 = mul_f2f(t3, F2F_1501);

    let p1 = p5 + mul_f2f(p1, F2F_0900);
    let p2 = p5 + mul_f2f(p2, F2F_2563);
    let p3 = mul_f2f(p3, F2F_1962);
    let p4 = mul_f2f(p4, F2F_0390);

    t3 = t3 + p1 + p4;
    t2 = t2 + p2 + p3;
    t1 = t1 + p2 + p4;
    t0 = t0 + p1 + p3;

    [t0, t1, t2, t3]
}

#[inline]
fn kernel_simd(s: [i32x8; 8], x_scale: i32) -> ([i32x8; 4], [i32x8; 4]) {
    let xs = kernel_x_simd([s[0], s[2], s[4], s[6]], x_scale);
    let ts = kernel_t_simd([s[1], s[3], s[5], s[7]]);
    (xs, ts)
}

fn transpose_8x8_simd(mat: [i32x8; 8]) -> [i32x8; 8] {
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


#[inline(always)]
fn dequantize(c: i16, q: u16) -> Wrapping<i32> {
    Wrapping(i32::from(c) * i32::from(q))
}




fn stbi_f2f(x: f32) -> Wrapping<i32> {
    Wrapping((x * 4096.0 + 0.5) as i32)
}

fn stbi_fsh(x: Wrapping<i32>) -> Wrapping<i32> {
    x << 12
}

/// Encode an 8x8 block with dct8, decode with fast_dct8 (drop-in API). Different DCT
/// normalizations => decoded i16 is within tolerance of original, not bit-identical.
#[test]
fn test_dct8_encode_fast_dct8_decode() {
    use crate::dct8;

    let plane: Vec<i16> = (0..64).map(|i| ((i * 7) % 255) as i16 - 128).collect();
    let quant_step = 8.0f32;
    let num_blocks = 1usize;
    let encoded = dct8::encode_plane_8x8_aq(&plane, 8, 8, 8, &vec![quant_step; num_blocks], None, 0.5);
    let coeffs = encoded[0].as_ref().unwrap().as_slice();
    let skip_mask = [false];

    let mut decoded = vec![0i16; 64];
    decode_plane_8x8_aq(coeffs, &mut decoded, 8, 8, 8, &vec![quant_step; num_blocks], &skip_mask);

    for i in 0..64 {
        let diff = (plane[i] as i32 - decoded[i] as i32).abs();
        assert!(
            diff <= 25,
            "dct8 encode + fast_dct8 decode at {}: orig={} decoded={}",
            i,
            plane[i],
            decoded[i]
        );
    }
}
