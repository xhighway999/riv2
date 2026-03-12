use thiserror::Error;

#[cfg(feature = "threads")]
use rayon::prelude::*;

use reitero_dct;
use reitero_video_common::Yuv420Frame;

const BEST_QUALITY_QUANT_STEP: f32 = 1.0;
const WORST_QUALITY_QUANT_STEP: f32 = 112.0;

/// Calculate quantization step from quality parameter (1-100)
pub fn quant_step_from_quality(quality: u8) -> f32 {
    if quality == 100 {
        BEST_QUALITY_QUANT_STEP
    } else if quality == 1 {
        WORST_QUALITY_QUANT_STEP
    } else {
        let quality_f = quality as f32;
        let range = WORST_QUALITY_QUANT_STEP - BEST_QUALITY_QUANT_STEP;
        BEST_QUALITY_QUANT_STEP + (100.0 - quality_f) / 99.0 * range
    }
}

// ---------------------------------------------------------------------------
// JPEG-style perceptual quantization matrices for intra frames
// ---------------------------------------------------------------------------

/// Standard JPEG luminance quantization table (8×8, zigzag-independent row-major).
/// Values represent relative perceptual importance — higher = more aggressively quantized.
#[rustfmt::skip]
const JPEG_LUMA_8X8: [f32; 64] = [
    16.0, 11.0, 10.0, 16.0, 24.0,  40.0,  51.0,  61.0,
    12.0, 12.0, 14.0, 19.0, 26.0,  58.0,  60.0,  55.0,
    14.0, 13.0, 16.0, 24.0, 40.0,  57.0,  69.0,  56.0,
    14.0, 17.0, 22.0, 29.0, 51.0,  87.0,  80.0,  62.0,
    18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0,  77.0,
    24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0,  92.0,
    49.0, 64.0, 78.0, 87.0,103.0, 121.0, 120.0, 101.0,
    72.0, 92.0, 95.0, 98.0,112.0, 100.0, 103.0,  99.0,
];

/// Standard JPEG chrominance quantization table (8×8).
#[rustfmt::skip]
const JPEG_CHROMA_8X8: [f32; 64] = [
    17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0,
    18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0,
    24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
];

/// Build a 16×16 intra luma quantization table by bilinearly upscaling the JPEG 8×8 table,
/// then scaling by the user's quality-derived quant_step.
pub fn build_intra_y_quant_table(quant_step: f32) -> [u16; 256] {
    let mut table = [0u16; 256];
    // Bilinear interpolation from 8×8 → 16×16
    for v in 0..16u32 {
        for u in 0..16u32 {
            // Map 16×16 coordinate to 8×8 space
            let fx = (u as f32) * 7.0 / 15.0;
            let fy = (v as f32) * 7.0 / 15.0;
            let x0 = fx as usize;
            let y0 = fy as usize;
            let x1 = (x0 + 1).min(7);
            let y1 = (y0 + 1).min(7);
            let dx = fx - x0 as f32;
            let dy = fy - y0 as f32;

            let val = JPEG_LUMA_8X8[y0 * 8 + x0] * (1.0 - dx) * (1.0 - dy)
                + JPEG_LUMA_8X8[y0 * 8 + x1] * dx * (1.0 - dy)
                + JPEG_LUMA_8X8[y1 * 8 + x0] * (1.0 - dx) * dy
                + JPEG_LUMA_8X8[y1 * 8 + x1] * dx * dy;

            // Scale: JPEG table values are for quality ~50. Normalize so DC ≈ quant_step,
            // and higher frequencies get proportionally larger steps.
            let scaled = (val / JPEG_LUMA_8X8[0]) * quant_step;
            table[(v * 16 + u) as usize] = scaled.round().max(1.0) as u16;
        }
    }
    table
}

/// Build an 8×8 intra chroma quantization table from the JPEG chrominance table,
/// scaled by the user's quality-derived quant_step.
pub fn build_intra_uv_quant_table(quant_step: f32) -> [u16; 64] {
    let mut table = [0u16; 64];
    for i in 0..64 {
        let scaled = (JPEG_CHROMA_8X8[i] / JPEG_CHROMA_8X8[0]) * quant_step;
        table[i] = scaled.round().max(1.0) as u16;
    }
    table
}


// ---------------------------------------------------------------------------
// Adaptive Quantization (x264-style variance-based AQ)
// ---------------------------------------------------------------------------

/// Default AQ strength parameter.
pub const AQ_STRENGTH_DEFAULT: f32 = 0.8;

/// Minimum variance floor to avoid log(0) and extreme quant on flat blocks.
const AQ_MIN_VARIANCE: f32 = 4.0;

/// Compute per-block adaptive quant steps from a reference frame's Y plane.
///
/// Uses x264's formula: `offset = strength * (log2(block_var) - log2(avg_var))`
/// then `block_qs = base_qs * 2^(offset/6)`.
///
/// The reference frame should be the predicted frame (available to both encoder and decoder).
/// Returns per-block quant steps for 16×16 luma blocks.
pub fn compute_aq_quant_steps(
    y_plane: &[u8],
    y_stride: usize,
    width: usize,
    height: usize,
    base_quant_step: f32,
) -> Vec<f32> {
    let aq_strength = AQ_STRENGTH_DEFAULT;
    let blocks_w = width / 16;
    let blocks_h = height / 16;
    let num_blocks = blocks_w * blocks_h;

    // Fast path: AQ disabled
    if aq_strength <= 0.0 {
        return vec![base_quant_step; num_blocks];
    }

    // Compute per-block variance
    let mut variances = Vec::with_capacity(num_blocks);
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let mut sum: u64 = 0;
            let mut sum_sq: u64 = 0;
            for y in 0..16 {
                let row_start = (by * 16 + y) * y_stride + bx * 16;
                for x in 0..16 {
                    let v = y_plane[row_start + x] as u64;
                    sum += v;
                    sum_sq += v * v;
                }
            }
            let n = 256u64; // 16*16
            let mean_sq = (sum * sum) / n;
            let variance = ((sum_sq * n - sum * sum) as f32) / (n * n) as f32;
            let _ = mean_sq; // unused, variance computed directly
            variances.push(variance.max(AQ_MIN_VARIANCE));
        }
    }

    // Compute average log-variance
    let avg_log_var: f32 = variances.iter().map(|v| v.ln()).sum::<f32>() / num_blocks as f32;

    // Compute per-block quant steps using x264 formula
    variances
        .iter()
        .map(|&var| {
            let offset = aq_strength * (var.ln() - avg_log_var);
            // offset > 0 for busy blocks (coarser quant), < 0 for smooth blocks (finer quant)
            base_quant_step * (offset / 6.0).exp2()
        })
        .collect()
}

use crate::rans::{DctRansDecoder, DctRansEncoder};
use chrono::Utc;
use std::sync::atomic::{AtomicU64, Ordering};
use yuv::YuvChromaSubsampling;

pub type Result<T> = std::result::Result<T, ResidualError>;

// Lightweight in-process profiler counters (ns)
static RANS_DECODE_NS: AtomicU64 = AtomicU64::new(0);
static DEINTERLEAVE_NS: AtomicU64 = AtomicU64::new(0);
static DCT_Y_NS: AtomicU64 = AtomicU64::new(0);
static DCT_UV_NS: AtomicU64 = AtomicU64::new(0);
static APPLY_NS: AtomicU64 = AtomicU64::new(0);

/// Drain and return residual-phase counters (ns). Returns (rans, deinterleave, dct_y, dct_uv, apply)
pub fn drain_residual_phase_counters() -> (u64, u64, u64, u64, u64) {
    let r = RANS_DECODE_NS.swap(0, Ordering::Relaxed);
    let d = DEINTERLEAVE_NS.swap(0, Ordering::Relaxed);
    let y = DCT_Y_NS.swap(0, Ordering::Relaxed);
    let uv = DCT_UV_NS.swap(0, Ordering::Relaxed);
    let a = APPLY_NS.swap(0, Ordering::Relaxed);
    (r, d, y, uv, a)
}

#[derive(Debug, Error)]
pub enum ResidualError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Result for encoding an intra frame via DCT+RANS, including the reconstructed YUV the
/// decoder will reproduce.
#[derive(Debug, Clone)]
pub struct IntraResidualEncodeResult {
    pub residual_data: Vec<u8>,
    pub recon_yuv: Yuv420Frame,
}

#[derive(Debug, Clone)]
pub struct InterResidualEncodeParams<'a> {
    pub curr_yuv: &'a Yuv420Frame,      // storage-sized YUV420
    pub predicted_yuv: &'a Yuv420Frame, // storage-sized YUV420
    pub storage_width: u32,
    pub storage_height: u32,
    pub skip_mask: &'a [bool], // blocks_w * blocks_h
    pub inter_quality: u8,
    pub inter_dead_zone: f32,
}

#[derive(Debug, Clone)]
pub struct InterResidualEncodeResult {
    /// Opaque residual data bytes (internal format is hidden)
    pub residual_data: Vec<u8>,
    /// Reconstructed current frame (storage-sized YUV420), computed using the YUV residual.
    pub recon_current_yuv: Yuv420Frame,
    pub blocks_total: usize,
    pub blocks_kept: usize,
    /// RLE compression statistics (for diagnostics)
    pub rle_stats: Option<RleCompressionStats>,
    /// Optimized skip mask: combines original skip_mask with blocks that became zero after quantization
    /// This allows the encoder to further optimize by skipping blocks that quantize to all zeros
    pub optimized_skip_mask: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct RleCompressionStats {
    pub raw_size_bytes: usize,
    pub rle_size_bytes: usize,
    pub compression_ratio: f64,
    pub total_pairs: usize,
    pub pairs_per_block: f64,
    pub non_zero_coeffs: usize,
    pub total_coeffs: usize,
    pub zero_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct InterResidualDecodeParams<'a> {
    pub predicted_yuv: &'a Yuv420Frame, // storage-sized YUV420
    pub storage_width: u32,
    pub storage_height: u32,
    pub skip_mask: &'a [bool],
    /// Opaque residual data bytes (internal format is hidden)
    pub residual_data: &'a [u8],
    pub inter_quality: u8,
    /// If true, skip residual decoding and return predicted frame only
    pub skip_residuals: bool,
}

fn frame_is_all_zero(frame: &Yuv420Frame) -> bool {
    frame.y_plane().iter().all(|&b| b == 0)
        && frame.u_plane().iter().all(|&b| b == 0)
        && frame.v_plane().iter().all(|&b| b == 0)
}

pub struct ResidualEncoder;

impl ResidualEncoder {
    pub fn encode_intra(
        curr_yuv: &Yuv420Frame,
        storage_width: u32,
        storage_height: u32,
        quality: u8,
    ) -> Result<IntraResidualEncodeResult> {
        let storage_w = storage_width as usize;
        let storage_h = storage_height as usize;
        if curr_yuv.width() != storage_w || curr_yuv.height() != storage_h {
            return Err(ResidualError::InvalidInput(format!(
                "curr_yuv dims mismatch: expected {}x{}, got {}x{}",
                storage_w, storage_h, curr_yuv.width(), curr_yuv.height()
            )));
        }
        if storage_w % 16 != 0 || storage_h % 16 != 0 {
            return Err(ResidualError::InvalidInput(format!(
                "storage dims must be multiple of 16, got {}x{}",
                storage_w, storage_h
            )));
        }

        let y_len = storage_w * storage_h;
        let uv_w = storage_w / 2;
        let uv_h = storage_h / 2;
        let uv_len = uv_w * uv_h;
        let blocks_w = storage_w / 16;
        let blocks_h = storage_h / 16;
        let blocks_total = blocks_w * blocks_h;
        let quant_step = quant_step_from_quality(quality);
        // All blocks encoded for intra — no skipping.
        let skip_mask = vec![false; blocks_total];

        const Y_COEFFS_PER_BLOCK: usize = 16 * 16;
        const UV_COEFFS_PER_BLOCK: usize = 8 * 8;

        // Build residual planes: pixel - 128 (DC predictor), range [-128, 127].
        let mut res_y = vec![0i16; y_len];
        let mut res_u = vec![0i16; uv_len];
        let mut res_v = vec![0i16; uv_len];
        for (&y, r) in curr_yuv.y_plane().iter().zip(res_y.iter_mut()) {
            *r = y as i16 - 128;
        }
        for ((&u, &v), (ru, rv)) in curr_yuv.u_plane().iter()
            .zip(curr_yuv.v_plane().iter())
            .zip(res_u.iter_mut().zip(res_v.iter_mut()))
        {
            *ru = u as i16 - 128;
            *rv = v as i16 - 128;
        }

        // DCT encode all planes with perceptual quantization matrix (intra only).
        let y_qtable = build_intra_y_quant_table(quant_step);
        let uv_qtable = build_intra_uv_quant_table(quant_step);
        let y_coeffs_all = reitero_dct::encode_plane_16x16_matrix(
            &res_y, storage_w, storage_w, storage_h, &y_qtable, None,
        );
        let u_coeffs_all = reitero_dct::encode_plane_8x8_matrix(
            &res_u, uv_w, uv_w, uv_h, &uv_qtable, None,
        );
        let v_coeffs_all = reitero_dct::encode_plane_8x8_matrix(
            &res_v, uv_w, uv_w, uv_h, &uv_qtable, None,
        );

        // RANS-encode and collect interleaved DCT coefficients.
        let mut rans_encoder: Option<DctRansEncoder> = None;
        let mut dct_coefficients =
            Vec::with_capacity(blocks_total * (Y_COEFFS_PER_BLOCK + UV_COEFFS_PER_BLOCK * 2));

        for bi in 0..blocks_total {
            match (&y_coeffs_all[bi], &u_coeffs_all[bi], &v_coeffs_all[bi]) {
                (Some(yc), Some(uc), Some(vc)) => {
                    let enc = rans_encoder.get_or_insert_with(|| DctRansEncoder::with_dc_prediction(true));
                    enc.encode_block(yc, uc, vc, Y_COEFFS_PER_BLOCK, UV_COEFFS_PER_BLOCK, UV_COEFFS_PER_BLOCK);
                    dct_coefficients.extend_from_slice(yc);
                    dct_coefficients.extend_from_slice(uc);
                    dct_coefficients.extend_from_slice(vc);
                }
                _ => {
                    // Should never happen for intra (no skip_mask), but handle gracefully.
                    dct_coefficients.extend(std::iter::repeat(0i16).take(Y_COEFFS_PER_BLOCK + UV_COEFFS_PER_BLOCK * 2));
                }
            }
        }

        // Reconstruct: IDCT the interleaved coefficients, then add DC predictor back.
        let mut y_coeffs_recon = Vec::with_capacity(blocks_total * Y_COEFFS_PER_BLOCK);
        let mut u_coeffs_recon = Vec::with_capacity(blocks_total * UV_COEFFS_PER_BLOCK);
        let mut v_coeffs_recon = Vec::with_capacity(blocks_total * UV_COEFFS_PER_BLOCK);
        let mut ci = 0;
        for _ in 0..blocks_total {
            y_coeffs_recon.extend_from_slice(&dct_coefficients[ci..ci + Y_COEFFS_PER_BLOCK]);
            ci += Y_COEFFS_PER_BLOCK;
            u_coeffs_recon.extend_from_slice(&dct_coefficients[ci..ci + UV_COEFFS_PER_BLOCK]);
            ci += UV_COEFFS_PER_BLOCK;
            v_coeffs_recon.extend_from_slice(&dct_coefficients[ci..ci + UV_COEFFS_PER_BLOCK]);
            ci += UV_COEFFS_PER_BLOCK;
        }

        let mut recon_res_y = vec![0i16; y_len];
        let mut recon_res_u = vec![0i16; uv_len];
        let mut recon_res_v = vec![0i16; uv_len];
        reitero_dct::decode_plane_16x16_matrix(
            &y_coeffs_recon, &mut recon_res_y, storage_w, storage_w, storage_h,
            &y_qtable, &skip_mask,
        );
        reitero_dct::decode_plane_8x8_matrix(
            &u_coeffs_recon, &mut recon_res_u, uv_w, uv_w, uv_h, &uv_qtable, &skip_mask,
        );
        reitero_dct::decode_plane_8x8_matrix(
            &v_coeffs_recon, &mut recon_res_v, uv_w, uv_w, uv_h, &uv_qtable, &skip_mask,
        );

        let recon_y: Vec<u8> = recon_res_y.iter().map(|&r| (r as i32 + 128).clamp(0, 255) as u8).collect();
        let recon_u: Vec<u8> = recon_res_u.iter().map(|&r| (r as i32 + 128).clamp(0, 255) as u8).collect();
        let recon_v: Vec<u8> = recon_res_v.iter().map(|&r| (r as i32 + 128).clamp(0, 255) as u8).collect();
        let recon_yuv = Yuv420Frame::from_planes(storage_w, storage_h, recon_y, recon_u, recon_v)
            .map_err(|e| ResidualError::InvalidInput(format!("intra recon planes invalid: {e}")))?;

        let residual_data = rans_encoder.map(|mut enc| enc.finish()).unwrap_or_default();

        Ok(IntraResidualEncodeResult { residual_data, recon_yuv })
    }

    /// Residual system:
    /// - residual in YUV420 planar: r = current_yuv - predicted_yuv
    /// - encode residuals using DCT + quantization + zigzag (16x16 blocks for Y, 8x8 blocks for U/V)
    /// - store quantized DCT coefficients as i16 per block (zigzag order)
    /// - skip blocks (from skip_mask) are stored as all zeros
    /// - reconstruct current in YUV from quantized data, then convert back to RGB24 for reference/output
    pub fn encode_inter(p: InterResidualEncodeParams<'_>) -> Result<InterResidualEncodeResult> {
        // TEST-ONLY DEBUG: detect the all-zero RGB case very early so we can confirm
        // cfg!(test) behaviour and what kind of data goes through this path.
        let predicted_all_zero = frame_is_all_zero(p.predicted_yuv);
        let curr_all_zero = frame_is_all_zero(p.curr_yuv);

        if cfg!(test) && curr_all_zero && predicted_all_zero {
            // This will only print when running tests; in production builds cfg!(test) is false.
            println!(
                "[encode_inter][test] all-zero curr/pred RGB case: storage={}x{}, quality={}",
                p.storage_width, p.storage_height, p.inter_quality
            );
        }

        let storage_w = p.storage_width as usize;
        let storage_h = p.storage_height as usize;
        if p.curr_yuv.width() != storage_w || p.curr_yuv.height() != storage_h {
            return Err(ResidualError::InvalidInput(format!(
                "curr_yuv dims mismatch: expected {}x{}, got {}x{}",
                storage_w,
                storage_h,
                p.curr_yuv.width(),
                p.curr_yuv.height()
            )));
        }
        if p.predicted_yuv.width() != storage_w || p.predicted_yuv.height() != storage_h {
            return Err(ResidualError::InvalidInput(format!(
                "predicted_yuv dims mismatch: expected {}x{}, got {}x{}",
                storage_w,
                storage_h,
                p.predicted_yuv.width(),
                p.predicted_yuv.height()
            )));
        }
        if storage_w % 16 != 0 || storage_h % 16 != 0 {
            return Err(ResidualError::InvalidInput(format!(
                "storage dims must be multiple of 16, got {}x{}",
                storage_w, storage_h
            )));
        }
        let blocks_w = storage_w / 16;
        let blocks_h = storage_h / 16;
        let blocks_total = blocks_w * blocks_h;
        let base_quant_step = quant_step_from_quality(p.inter_quality);
        if p.skip_mask.len() != blocks_total {
            return Err(ResidualError::InvalidInput(format!(
                "skip_mask len mismatch: expected {blocks_total}, got {}",
                p.skip_mask.len()
            )));
        }

        // blocks_kept will be recalculated after encoding using optimized_skip_mask
        let _blocks_kept_initial = p.skip_mask.iter().filter(|s| !**s).count();

        // Convert YUV420 references to planar views.
        let curr = p.curr_yuv.as_planar();
        let pred = p.predicted_yuv.as_planar();
        curr.check_constraints(YuvChromaSubsampling::Yuv420)
            .map_err(|e| ResidualError::InvalidInput(format!("curr_yuv constraints: {e:?}")))?;
        pred.check_constraints(YuvChromaSubsampling::Yuv420)
            .map_err(|e| ResidualError::InvalidInput(format!("pred_yuv constraints: {e:?}")))?;

        // Build residual planes as signed i16 (self-centered around 0), full-frame YUV420 planar.
        // These are temporary planes used for DCT encoding.
        let y_len = (p.storage_width * p.storage_height) as usize;
        let uv_len = ((p.storage_width / 2) * (p.storage_height / 2)) as usize;
        let mut res_y = vec![0i16; y_len];
        let mut res_u = vec![0i16; uv_len];
        let mut res_v = vec![0i16; uv_len];

        #[cfg(feature = "threads")]
        {
            // Parallel residual computation
            // We split by rows of blocks (16px high strips)
            res_y
                .par_chunks_mut(storage_w * 16)
                .zip(res_u.par_chunks_mut((storage_w / 2) * 8))
                .zip(res_v.par_chunks_mut((storage_w / 2) * 8))
                .enumerate()
                .for_each(|(by, ((row_res_y, row_res_u), row_res_v))| {
                    for bx in 0..blocks_w {
                        let bi = by * blocks_w + bx;
                        if p.skip_mask[bi] {
                            continue;
                        }
                        let x0 = bx * 16;
                        // y0 is relative to the chunk start (always 0 for the chunk)
                        // but we need absolute y0 for source indexing
                        let abs_y0 = by * 16;

                        // Y plane
                        for yy in 0..16 {
                            let y = abs_y0 + yy;
                            let row = (y as u32 * curr.y_stride) as usize;
                            let prow = (y as u32 * pred.y_stride) as usize;
                            let drow = yy * storage_w; // relative to chunk
                            for xx in 0..16 {
                                let x = x0 + xx;
                                let i = row + x;
                                let pi = prow + x;
                                let r = curr.y_plane[i] as i16 - pred.y_plane[pi] as i16;
                                row_res_y[drow + x] = r;
                            }
                        }
                        // U/V planes
                        for yy in 0..8 {
                            let y_uv = (abs_y0 / 2) + yy;
                            let row = (y_uv as u32 * curr.u_stride) as usize;
                            let prow = (y_uv as u32 * pred.u_stride) as usize;
                            let drow = yy * (storage_w / 2); // relative to chunk
                            for xx in 0..8 {
                                let x = (x0 / 2) + xx;
                                let i = row + x;
                                let pi = prow + x;
                                let ru = curr.u_plane[i] as i16 - pred.u_plane[pi] as i16;
                                let rv = curr.v_plane[i] as i16 - pred.v_plane[pi] as i16;
                                row_res_u[drow + x] = ru;
                                row_res_v[drow + x] = rv;
                            }
                        }
                    }
                });
        }

        #[cfg(not(feature = "threads"))]
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                let bi = by * blocks_w + bx;
                if p.skip_mask[bi] {
                    continue;
                }
                let x0 = bx * 16;
                let y0 = by * 16;

                for yy in 0..16 {
                    let y = y0 + yy;
                    let row = (y as u32 * curr.y_stride) as usize;
                    let prow = (y as u32 * pred.y_stride) as usize;
                    let drow = y * storage_w;
                    for xx in 0..16 {
                        let x = x0 + xx;
                        let i = row + x;
                        let pi = prow + x;
                        let r = curr.y_plane[i] as i16 - pred.y_plane[pi] as i16;
                        res_y[drow + x] = r;
                    }
                }
                // U/V planes: 8x8 per 16x16 block (YUV420 halves both dimensions)
                for yy in 0..8 {
                    let y_uv = (y0 / 2) + yy;
                    let row = (y_uv as u32 * curr.u_stride) as usize;
                    let prow = (y_uv as u32 * pred.u_stride) as usize;
                    let drow = y_uv * (storage_w / 2);
                    for xx in 0..8 {
                        let x = (x0 / 2) + xx;
                        let i = row + x;
                        let pi = prow + x;
                        let ru = curr.u_plane[i] as i16 - pred.u_plane[pi] as i16;
                        let rv = curr.v_plane[i] as i16 - pred.v_plane[pi] as i16;
                        res_u[drow + x] = ru;
                        res_v[drow + x] = rv;
                    }
                }
            }
        }

        // Encode residuals using DCT + quantization + zigzag + RANS: process each block
        // Format: For each motion block (raster order): Y block (16x16), U block (8x8), V block (8x8)
        // Each block is RANS-encoded and serialized to bytes
        const Y_BLOCK_SIZE: usize = 16;
        const UV_BLOCK_SIZE: usize = 8;
        const Y_COEFFS_PER_BLOCK: usize = Y_BLOCK_SIZE * Y_BLOCK_SIZE; // 256
        const UV_COEFFS_PER_BLOCK: usize = UV_BLOCK_SIZE * UV_BLOCK_SIZE; // 64

        // Lazily create RANS encoder only if we actually have non-zero blocks to encode.
        // This avoids exercising the RANS writer in the "all blocks skipped/zero" case.
        let mut rans_encoder: Option<DctRansEncoder> = None;
        let mut dct_coefficients =
            Vec::with_capacity(blocks_total * (Y_COEFFS_PER_BLOCK + UV_COEFFS_PER_BLOCK * 2));
        // Build optimized skip mask: original skip_mask + blocks that become zero after quantization
        let mut optimized_skip_mask = Vec::with_capacity(blocks_total);
        // TEST-ONLY: track how many blocks become newly skipped after quantization
        let mut newly_skipped_after_quant: usize = 0;

        // Compute adaptive per-block quant steps from predicted frame variance
        let aq_y_steps = compute_aq_quant_steps(
            pred.y_plane, pred.y_stride as usize,
            storage_w, storage_h, base_quant_step,
        );
        // Chroma blocks are 8×8 but map 1:1 to 16×16 macroblocks, so reuse the same per-block steps
        let aq_uv_steps = &aq_y_steps;

        // Encode planes with per-block adaptive quantization
        let y_coeffs_all = reitero_dct::encode_plane_16x16_aq(
            &res_y,
            storage_w,
            storage_w,
            storage_h,
            &aq_y_steps,
            Some(p.skip_mask),
            p.inter_dead_zone,
        );
        let u_coeffs_all = reitero_dct::encode_plane_8x8_aq(
            &res_u,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            aq_uv_steps,
            Some(p.skip_mask),
            p.inter_dead_zone,
        );
        let v_coeffs_all = reitero_dct::encode_plane_8x8_aq(
            &res_v,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            aq_uv_steps,
            Some(p.skip_mask),
            p.inter_dead_zone,
        );

        // Serial RANS encoding + state update
        for bi in 0..blocks_total {
            let y_opt = &y_coeffs_all[bi];
            let u_opt = &u_coeffs_all[bi];
            let v_opt = &v_coeffs_all[bi];

            if let (Some(y_coeffs), Some(u_coeffs), Some(v_coeffs)) = (y_opt, u_opt, v_opt) {
                // Check if block became effectively zero after quantization
                let y_all_zero = y_coeffs.iter().all(|&c| c == 0);
                let u_all_zero = u_coeffs.iter().all(|&c| c == 0);
                let v_all_zero = v_coeffs.iter().all(|&c| c == 0);
                let block_became_zero = y_all_zero && u_all_zero && v_all_zero;

                // Test-only assertion: all-zero input must produce all-zero quantized coeffs.
                if cfg!(test) && curr_all_zero && predicted_all_zero && !block_became_zero {
                    let nz_y = y_coeffs.iter().filter(|&&c| c != 0).count();
                    let nz_u = u_coeffs.iter().filter(|&&c| c != 0).count();
                    let nz_v = v_coeffs.iter().filter(|&&c| c != 0).count();
                    let bx = bi % blocks_w;
                    let by = bi / blocks_w;
                    panic!(
                        "All-zero RGB test: block ({}, {}) produced non-zero quantized DCT coeffs: \
                         nz_y={}, nz_u={}, nz_v={}",
                        bx, by, nz_y, nz_u, nz_v
                    );
                }

                if block_became_zero {
                    optimized_skip_mask.push(true);
                    if cfg!(test) {
                        newly_skipped_after_quant += 1;
                    }
                    dct_coefficients.extend_from_slice(&vec![0i16; Y_COEFFS_PER_BLOCK]);
                    dct_coefficients.extend_from_slice(&vec![0i16; UV_COEFFS_PER_BLOCK]);
                    dct_coefficients.extend_from_slice(&vec![0i16; UV_COEFFS_PER_BLOCK]);
                } else {
                    let enc = rans_encoder.get_or_insert_with(DctRansEncoder::new);
                    enc.encode_block(
                        y_coeffs,
                        u_coeffs,
                        v_coeffs,
                        Y_COEFFS_PER_BLOCK,
                        UV_COEFFS_PER_BLOCK,
                        UV_COEFFS_PER_BLOCK,
                    );
                    dct_coefficients.extend_from_slice(y_coeffs);
                    dct_coefficients.extend_from_slice(u_coeffs);
                    dct_coefficients.extend_from_slice(v_coeffs);
                    optimized_skip_mask.push(false);
                }
            } else {
                // Original skip
                optimized_skip_mask.push(true);
                dct_coefficients.extend_from_slice(&vec![0i16; Y_COEFFS_PER_BLOCK]);
                dct_coefficients.extend_from_slice(&vec![0i16; UV_COEFFS_PER_BLOCK]);
                dct_coefficients.extend_from_slice(&vec![0i16; UV_COEFFS_PER_BLOCK]);
            }
        }

        // TEST-ONLY: if we're in the all-zero RGB test case, report how many
        // blocks became newly skipped after quantization. This helps verify
        // that the "optimized skip" logic behaves as expected in that scenario.
        if cfg!(test) && curr_all_zero && predicted_all_zero {
            println!(
                "[encode_inter][test] newly skipped after quantization: {} of {} blocks",
                newly_skipped_after_quant, blocks_total
            );
        }

        // Decode DCT coefficients back to residual planes for reconstruction
        // This ensures encoder and decoder use identical reconstruction
        let mut recon_res_y = vec![0i16; y_len];
        let mut recon_res_u = vec![0i16; uv_len];
        let mut recon_res_v = vec![0i16; uv_len];

        // De-interleave coefficients for reconstruction
        let mut y_coeffs_recon = Vec::with_capacity(blocks_total * Y_COEFFS_PER_BLOCK);
        let mut u_coeffs_recon = Vec::with_capacity(blocks_total * UV_COEFFS_PER_BLOCK);
        let mut v_coeffs_recon = Vec::with_capacity(blocks_total * UV_COEFFS_PER_BLOCK);

        let mut coeff_idx = 0;
        for _ in 0..blocks_total {
            y_coeffs_recon
                .extend_from_slice(&dct_coefficients[coeff_idx..coeff_idx + Y_COEFFS_PER_BLOCK]);
            coeff_idx += Y_COEFFS_PER_BLOCK;
            u_coeffs_recon
                .extend_from_slice(&dct_coefficients[coeff_idx..coeff_idx + UV_COEFFS_PER_BLOCK]);
            coeff_idx += UV_COEFFS_PER_BLOCK;
            v_coeffs_recon
                .extend_from_slice(&dct_coefficients[coeff_idx..coeff_idx + UV_COEFFS_PER_BLOCK]);
            coeff_idx += UV_COEFFS_PER_BLOCK;
        }

        reitero_dct::decode_plane_16x16_aq(
            &y_coeffs_recon,
            &mut recon_res_y,
            storage_w,
            storage_w,
            storage_h,
            &aq_y_steps,
            &optimized_skip_mask,
        );
        reitero_dct::decode_plane_8x8_aq(
            &u_coeffs_recon,
            &mut recon_res_u,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            aq_uv_steps,
            &optimized_skip_mask,
        );
        reitero_dct::decode_plane_8x8_aq(
            &v_coeffs_recon,
            &mut recon_res_v,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            aq_uv_steps,
            &optimized_skip_mask,
        );

        // Apply residual in YUV to predicted YUV, then convert to RGB24 for reference/output.
        // Copy pred planes to contiguous arrays (matching residual storage format)
        let (mut recon_y, mut recon_u, mut recon_v) = p.predicted_yuv.clone_planes();

        // Apply decoded residuals (residuals are stored with stride = width, matching recon layout)
        for i in 0..y_len {
            let residual = recon_res_y[i] as i32;
            recon_y[i] = (recon_y[i] as i32 + residual).clamp(0, 255) as u8;
        }
        for i in 0..uv_len {
            let ru = recon_res_u[i] as i32;
            let rv = recon_res_v[i] as i32;
            recon_u[i] = (recon_u[i] as i32 + ru).clamp(0, 255) as u8;
            recon_v[i] = (recon_v[i] as i32 + rv).clamp(0, 255) as u8;
        }

        // Convert recon YUV planes back to RGB24
        let recon_current_yuv = Yuv420Frame::from_planes(
            storage_w, storage_h, recon_y, recon_u, recon_v,
        )
        .map_err(|e| ResidualError::InvalidInput(format!("reconstructed planes invalid: {e}")))?;

        // Finish RANS encoding and get the encoded bytes (if we ever instantiated it).
        let residual_data = if let Some(mut enc) = rans_encoder {
            enc.finish()
        } else {
            Vec::new()
        };

        // Calculate blocks_kept using optimized skip mask (excludes blocks that became zero after quantization)
        let blocks_kept_optimized = optimized_skip_mask.iter().filter(|s| !**s).count();

        // Calculate compression statistics
        // raw_size: actual size of zigzag coefficients for blocks that are actually encoded (non-skipped blocks only)
        // Each block has Y (256) + U (64) + V (64) = 384 coefficients, each i16 = 2 bytes
        let raw_size = blocks_kept_optimized * (Y_COEFFS_PER_BLOCK + UV_COEFFS_PER_BLOCK * 2) * 2; // bytes (i16 = 2 bytes)
        let rans_size = residual_data.len();
        let rans_compression_ratio = if rans_size > 0 {
            raw_size as f64 / rans_size as f64
        } else {
            1.0
        };

        // Count non-zero coefficients for diagnostics
        let non_zero_coeffs: usize = dct_coefficients.iter().filter(|&&c| c != 0).count();
        let total_coeffs = dct_coefficients.len();
        let zero_percentage = if total_coeffs > 0 {
            ((total_coeffs - non_zero_coeffs) as f64 / total_coeffs as f64) * 100.0
        } else {
            0.0
        };

        let rle_stats = Some(RleCompressionStats {
            raw_size_bytes: raw_size,
            rle_size_bytes: rans_size,
            compression_ratio: rans_compression_ratio,
            total_pairs: 0,       // Not applicable for RANS
            pairs_per_block: 0.0, // Not applicable for RANS
            non_zero_coeffs,
            total_coeffs,
            zero_percentage,
        });

        Ok(InterResidualEncodeResult {
            residual_data,
            recon_current_yuv,
            blocks_total,
            blocks_kept: blocks_kept_optimized,
            rle_stats,
            optimized_skip_mask,
        })
    }
}

pub struct ResidualDecoder;

impl ResidualDecoder {
    pub fn decode_intra(
        residual_data: &[u8],
        storage_width: u32,
        storage_height: u32,
        quality: u8,
    ) -> Result<Yuv420Frame> {
        let storage_w = storage_width as usize;
        let storage_h = storage_height as usize;
        if storage_w % 16 != 0 || storage_h % 16 != 0 {
            return Err(ResidualError::InvalidInput(format!(
                "storage dims must be multiple of 16, got {}x{}",
                storage_w, storage_h
            )));
        }

        let y_len = storage_w * storage_h;
        let uv_w = storage_w / 2;
        let uv_h = storage_h / 2;
        let uv_len = uv_w * uv_h;
        let blocks_w = storage_w / 16;
        let blocks_h = storage_h / 16;
        let blocks_total = blocks_w * blocks_h;
        let quant_step = quant_step_from_quality(quality);

        const Y_COEFFS_PER_BLOCK: usize = 16 * 16;
        const UV_COEFFS_PER_BLOCK: usize = 8 * 8;

        let mut rans_decoder = DctRansDecoder::with_dc_prediction(true);
        if !residual_data.is_empty() {
            rans_decoder.consume(residual_data);
        }

        let mut y_coeffs_all = vec![0i16; blocks_total * Y_COEFFS_PER_BLOCK];
        let mut u_coeffs_all = vec![0i16; blocks_total * UV_COEFFS_PER_BLOCK];
        let mut v_coeffs_all = vec![0i16; blocks_total * UV_COEFFS_PER_BLOCK];

        // All blocks are present for intra (no skip).
        let mut y_off = 0usize;
        let mut uv_off = 0usize;
        for _ in 0..blocks_total {
            rans_decoder.decode_block_into(
                &mut y_coeffs_all[y_off..y_off + Y_COEFFS_PER_BLOCK],
                &mut u_coeffs_all[uv_off..uv_off + UV_COEFFS_PER_BLOCK],
                &mut v_coeffs_all[uv_off..uv_off + UV_COEFFS_PER_BLOCK],
            );
            y_off += Y_COEFFS_PER_BLOCK;
            uv_off += UV_COEFFS_PER_BLOCK;
        }

        let skip_mask = vec![false; blocks_total];
        let mut res_y = vec![0i16; y_len];
        let mut res_u = vec![0i16; uv_len];
        let mut res_v = vec![0i16; uv_len];
        let y_qtable = build_intra_y_quant_table(quant_step);
        let uv_qtable = build_intra_uv_quant_table(quant_step);
        reitero_dct::decode_plane_16x16_matrix(
            &y_coeffs_all, &mut res_y, storage_w, storage_w, storage_h,
            &y_qtable, &skip_mask,
        );
        reitero_dct::decode_plane_8x8_matrix(
            &u_coeffs_all, &mut res_u, uv_w, uv_w, uv_h, &uv_qtable, &skip_mask,
        );
        reitero_dct::decode_plane_8x8_matrix(
            &v_coeffs_all, &mut res_v, uv_w, uv_w, uv_h, &uv_qtable, &skip_mask,
        );

        let recon_y: Vec<u8> = res_y.iter().map(|&r| (r as i32 + 128).clamp(0, 255) as u8).collect();
        let recon_u: Vec<u8> = res_u.iter().map(|&r| (r as i32 + 128).clamp(0, 255) as u8).collect();
        let recon_v: Vec<u8> = res_v.iter().map(|&r| (r as i32 + 128).clamp(0, 255) as u8).collect();

        Yuv420Frame::from_planes(storage_w, storage_h, recon_y, recon_u, recon_v)
            .map_err(|e| ResidualError::InvalidInput(format!("intra recon planes invalid: {e}")))
    }

    pub fn decode_inter(p: InterResidualDecodeParams<'_>) -> Result<Yuv420Frame> {
        let storage_w = p.storage_width as usize;
        let storage_h = p.storage_height as usize;
        if p.predicted_yuv.width() != storage_w || p.predicted_yuv.height() != storage_h {
            return Err(ResidualError::InvalidInput(format!(
                "predicted_yuv dims mismatch: expected {}x{}, got {}x{}",
                storage_w,
                storage_h,
                p.predicted_yuv.width(),
                p.predicted_yuv.height()
            )));
        }
        if storage_w % 16 != 0 || storage_h % 16 != 0 {
            return Err(ResidualError::InvalidInput(format!(
                "storage dims must be multiple of 16, got {}x{}",
                storage_w, storage_h
            )));
        }
        let blocks_w = storage_w / 16;
        let blocks_h = storage_h / 16;
        let blocks_total = blocks_w * blocks_h;
        let base_quant_step = quant_step_from_quality(p.inter_quality);
        if p.skip_mask.len() != blocks_total {
            return Err(ResidualError::InvalidInput(format!(
                "skip_mask len mismatch: expected {blocks_total}, got {}",
                p.skip_mask.len()
            )));
        }

        // If skip_residuals is true, just return the predicted frame
        if p.skip_residuals {
            return Ok(p.predicted_yuv.clone());
        }

        // Create RANS decoder and consume the encoded bytes
        let mut rans_decoder = DctRansDecoder::new();
        if !p.residual_data.is_empty() {
            reitero_video_common::Instrument::start_measure("20_rans_consume");
            rans_decoder.consume(p.residual_data);
            reitero_video_common::Instrument::stop_measure("20_rans_consume");
        }

        // Decode RANS blocks back to DCT coefficients
        reitero_video_common::Instrument::start_measure("21_rans_decode_blocks");
        // Format: For each motion block: Y block (256 coeffs), U block (64 coeffs), V block (64 coeffs)
        const Y_BLOCK_SIZE: usize = 16;
        const UV_BLOCK_SIZE: usize = 8;
        const Y_COEFFS_PER_BLOCK: usize = Y_BLOCK_SIZE * Y_BLOCK_SIZE; // 256
        const UV_COEFFS_PER_BLOCK: usize = UV_BLOCK_SIZE * UV_BLOCK_SIZE; // 64

        // Preallocate per-plane coefficient arrays for the whole frame.
        // We keep them zero-initialized so skipped blocks remain all-zero.
        let mut y_coeffs_all = vec![0i16; blocks_total * Y_COEFFS_PER_BLOCK];
        let mut u_coeffs_all = vec![0i16; blocks_total * UV_COEFFS_PER_BLOCK];
        let mut v_coeffs_all = vec![0i16; blocks_total * UV_COEFFS_PER_BLOCK];

        let t_r0 = Utc::now();
        let mut y_off = 0usize;
        let mut uv_off = 0usize;
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                let bi = by * blocks_w + bx;

                // IMPORTANT: skip_mask is authoritative - skipped blocks have no RANS data.
                // For skipped blocks, we leave the coefficient slices at zero.
                if !p.skip_mask[bi] {
                    // Non-skipped block: decode RANS blocks for Y, U, V directly into the
                    // appropriate slices inside the frame-sized buffers.
                    let y_slice = &mut y_coeffs_all[y_off..y_off + Y_COEFFS_PER_BLOCK];
                    let u_slice = &mut u_coeffs_all[uv_off..uv_off + UV_COEFFS_PER_BLOCK];
                    let v_slice = &mut v_coeffs_all[uv_off..uv_off + UV_COEFFS_PER_BLOCK];
                    rans_decoder.decode_block_into(y_slice, u_slice, v_slice);
                }

                y_off += Y_COEFFS_PER_BLOCK;
                uv_off += UV_COEFFS_PER_BLOCK;
            }
        }
        RANS_DECODE_NS.fetch_add((Utc::now() - t_r0).num_nanoseconds().unwrap_or(0).max(0) as u64, Ordering::Relaxed);
        reitero_video_common::Instrument::stop_measure("21_rans_decode_blocks");

        // Compute adaptive per-block quant steps from predicted frame (matches encoder)
        let pred_planar = p.predicted_yuv.as_planar();
        let aq_y_steps = compute_aq_quant_steps(
            pred_planar.y_plane, pred_planar.y_stride as usize,
            storage_w, storage_h, base_quant_step,
        );

        // Decode DCT coefficients to residual planes
        reitero_video_common::Instrument::start_measure("22_dct_decode_y");
        let y_len = (p.storage_width * p.storage_height) as usize;
        let uv_len = ((p.storage_width / 2) * (p.storage_height / 2)) as usize;
        let mut res_y = vec![0i16; y_len];
        let mut res_u = vec![0i16; uv_len];
        let mut res_v = vec![0i16; uv_len];

        let t_dct0 = Utc::now();
        reitero_dct::decode_plane_16x16_aq(
            &y_coeffs_all,
            &mut res_y,
            storage_w,
            storage_w,
            storage_h,
            &aq_y_steps,
            p.skip_mask,
        );
        DCT_Y_NS.fetch_add((Utc::now() - t_dct0).num_nanoseconds().unwrap_or(0).max(0) as u64, Ordering::Relaxed);
        reitero_video_common::Instrument::stop_measure("22_dct_decode_y");
        reitero_video_common::Instrument::start_measure("23_dct_decode_uv");
        let t_dctuv0 = Utc::now();
        reitero_dct::decode_plane_8x8_aq(
            &u_coeffs_all,
            &mut res_u,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            &aq_y_steps,
            p.skip_mask,
        );
        reitero_dct::decode_plane_8x8_aq(
            &v_coeffs_all,
            &mut res_v,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            &aq_y_steps,
            p.skip_mask,
        );
        DCT_UV_NS.fetch_add((Utc::now() - t_dctuv0).num_nanoseconds().unwrap_or(0).max(0) as u64, Ordering::Relaxed);
        reitero_video_common::Instrument::stop_measure("23_dct_decode_uv");

        // Copy predicted planes to contiguous arrays (matching residual storage format)
        // We apply residuals in-place on these u8 planes using widened i32 arithmetic per pixel.
        let (mut recon_y, mut recon_u, mut recon_v) = p.predicted_yuv.clone_planes();

        // Apply residual in YUV, but force residual=0 on skipped blocks (even if payload isn't neutral).
        // This keeps skip semantics correct.
        // Y plane: 16x16 per block; U/V: 8x8 per block (YUV420 halves both dimensions).
        reitero_video_common::Instrument::start_measure("24_apply_residual");
        let t_a0 = Utc::now();
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                let bi = by * blocks_w + bx;
                if p.skip_mask[bi] {
                    continue;
                }
                let x0 = bx * 16;
                let y0 = by * 16;

                // Y plane: process each 16-pixel row using slice zipping to avoid per-pixel index checks
                for yy in 0..16 {
                    let y = y0 + yy;
                    let row = y * storage_w;
                    let base = row + x0;
                    let recon_slice = &mut recon_y[base..base + 16];
                    let res_slice = &res_y[base..base + 16];
                    for (r, &rv) in recon_slice.iter_mut().zip(res_slice.iter()) {
                        let sum = *r as i32 + rv as i32;
                        *r = sum.clamp(0, 255) as u8;
                    }
                }

                // U/V planes: 8x8 per 16x16 block. Use slices similarly to speed inner loop.
                for yy in 0..8 {
                    let y_uv = (y0 / 2) + yy;
                    let row = y_uv * (storage_w / 2);
                    let base = row + (x0 / 2);
                    let recon_u_slice = &mut recon_u[base..base + 8];
                    let recon_v_slice = &mut recon_v[base..base + 8];
                    let res_u_slice = &res_u[base..base + 8];
                    let res_v_slice = &res_v[base..base + 8];
                    for ((ru, rv), (&su, &sv)) in recon_u_slice
                        .iter_mut()
                        .zip(recon_v_slice.iter_mut())
                        .zip(res_u_slice.iter().zip(res_v_slice.iter()))
                    {
                        let new_u = *ru as i32 + su as i32;
                        let new_v = *rv as i32 + sv as i32;
                        *ru = new_u.clamp(0, 255) as u8;
                        *rv = new_v.clamp(0, 255) as u8;
                    }
                }
            }
        }
        APPLY_NS.fetch_add((Utc::now() - t_a0).num_nanoseconds().unwrap_or(0).max(0) as u64, Ordering::Relaxed);
        reitero_video_common::Instrument::stop_measure("24_apply_residual");

        // Convert recon YUV planes back to RGB24
        Yuv420Frame::from_planes(storage_w, storage_h, recon_y, recon_u, recon_v)
            .map_err(|e| ResidualError::InvalidInput(format!("reconstructed planes invalid: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that optimized skip mask is authoritative:
    /// 1. Blocks that become zero after quantization are marked as skipped
    /// 2. No RLE data is generated for skipped blocks
    /// 3. Decoder correctly handles optimized skip mask
    #[test]
    fn test_optimized_skip_mask_authoritative() {
        // Create a simple test case: 32x32 frame (2x2 blocks of 16x16)
        let storage_w = 32u32;
        let storage_h = 32u32;
        let blocks_w = storage_w as usize / 16;
        let blocks_h = storage_h as usize / 16;
        let blocks_total = blocks_w * blocks_h;

        // Create current frame (all zeros - will quantize to zeros)
        let curr_rgb = vec![0u8; (storage_w * storage_h * 3) as usize];
        let curr_yuv = Yuv420Frame::from_rgb(&curr_rgb, storage_w as usize, storage_h as usize)
            .expect("curr rgb -> yuv");

        // Create predicted frame (also all zeros)
        let predicted_rgb = vec![0u8; (storage_w * storage_h * 3) as usize];
        let predicted_yuv =
            Yuv420Frame::from_rgb(&predicted_rgb, storage_w as usize, storage_h as usize)
                .expect("predicted rgb -> yuv");

        // Skip mask: first block skipped, others not
        let mut skip_mask = vec![false; blocks_total];
        skip_mask[0] = true; // First block skipped

        // Encode with high quality (so quantization doesn't zero out blocks)
        let result = ResidualEncoder::encode_inter(InterResidualEncodeParams {
            curr_yuv: &curr_yuv,
            predicted_yuv: &predicted_yuv,
            storage_width: storage_w,
            storage_height: storage_h,
            skip_mask: &skip_mask,
            inter_quality: 90, // High quality
            inter_dead_zone: 0.5,
        })
        .unwrap();

        // Verify optimized skip mask includes original skip
        assert_eq!(result.optimized_skip_mask.len(), blocks_total);
        assert!(
            result.optimized_skip_mask[0],
            "Original skipped block should be in optimized skip mask"
        );

        // Verify residual data is small for all-zero blocks
        // For all-zero residuals with high quality, blocks should quantize to zeros
        // So optimized skip mask should mark all blocks as skipped
        // This means minimal RANS data should be generated (just EOB markers for zero blocks)
        // With RANS, even zero blocks encode EOB, so we expect some data but it should be small
        assert!(
            result.residual_data.len() < 1000,
            "All-zero blocks should produce minimal RANS data, got {} bytes",
            result.residual_data.len()
        );

        // Verify that blocks_kept reflects optimized skip mask
        let blocks_kept_optimized = result.optimized_skip_mask.iter().filter(|s| !**s).count();
        assert_eq!(
            result.blocks_kept, blocks_kept_optimized,
            "blocks_kept should match optimized skip mask"
        );
    }

    /// Test that blocks quantizing to zero are included in optimized skip mask
    #[test]
    fn test_optimized_skip_mask_quantization_zeros() {
        let storage_w = 32u32;
        let storage_h = 32u32;
        let blocks_w = storage_w as usize / 16;
        let blocks_h = storage_h as usize / 16;
        let blocks_total = blocks_w * blocks_h;

        // Create frames with very small differences (will quantize to zero with low quality)
        let curr_rgb = vec![5u8; (storage_w * storage_h * 3) as usize];
        let curr_yuv = Yuv420Frame::from_rgb(&curr_rgb, storage_w as usize, storage_h as usize)
            .expect("curr rgb -> yuv");
        let predicted_rgb = vec![3u8; (storage_w * storage_h * 3) as usize]; // Small difference
        let predicted_yuv =
            Yuv420Frame::from_rgb(&predicted_rgb, storage_w as usize, storage_h as usize)
                .expect("predicted rgb -> yuv");

        // No blocks initially skipped
        let skip_mask = vec![false; blocks_total];

        // Encode with very low quality (will quantize small residuals to zero)
        let result = ResidualEncoder::encode_inter(InterResidualEncodeParams {
            curr_yuv: &curr_yuv,
            predicted_yuv: &predicted_yuv,
            storage_width: storage_w,
            storage_height: storage_h,
            skip_mask: &skip_mask,
            inter_quality: 1, // Very low quality - will quantize small residuals to zero
            inter_dead_zone: 0.5,
        })
        .unwrap();

        // With very low quality, small residuals should quantize to zero
        // So optimized skip mask should mark these blocks as skipped
        // Verify at least some blocks are in optimized skip mask
        let optimized_skipped = result.optimized_skip_mask.iter().filter(|s| **s).count();
        assert!(
            optimized_skipped > 0,
            "Some blocks should be marked as skipped after quantization"
        );

        // Verify blocks_kept is correct
        let blocks_kept_expected = blocks_total - optimized_skipped;
        assert_eq!(
            result.blocks_kept, blocks_kept_expected,
            "blocks_kept should match optimized skip mask"
        );
    }

    /// Test roundtrip: encode with optimized skip mask, then decode
    #[test]
    fn test_optimized_skip_mask_roundtrip() {
        let storage_w = 32u32;
        let storage_h = 32u32;
        let blocks_w = storage_w as usize / 16;
        let blocks_h = storage_h as usize / 16;
        let blocks_total = blocks_w * blocks_h;

        // Create test frames
        let mut curr_rgb = vec![128u8; (storage_w * storage_h * 3) as usize];
        // Add some variation to first block
        for i in 0..(16 * 16 * 3) {
            curr_rgb[i] = (128 + (i % 10) as u8).min(255);
        }
        let curr_yuv = Yuv420Frame::from_rgb(&curr_rgb, storage_w as usize, storage_h as usize)
            .expect("curr rgb -> yuv");

        let predicted_rgb = vec![128u8; (storage_w * storage_h * 3) as usize];
        let predicted_yuv =
            Yuv420Frame::from_rgb(&predicted_rgb, storage_w as usize, storage_h as usize)
                .expect("predicted rgb -> yuv");

        // Skip first block
        let mut skip_mask = vec![false; blocks_total];
        skip_mask[0] = true;

        // Encode
        let encode_result = ResidualEncoder::encode_inter(InterResidualEncodeParams {
            curr_yuv: &curr_yuv,
            predicted_yuv: &predicted_yuv,
            storage_width: storage_w,
            storage_height: storage_h,
            skip_mask: &skip_mask,
            inter_quality: 50,
            inter_dead_zone: 0.5,
        })
        .unwrap();

        // Decode using optimized skip mask (which should match what's in MV flags)
        let decoded_yuv = ResidualDecoder::decode_inter(InterResidualDecodeParams {
            predicted_yuv: &predicted_yuv,
            storage_width: storage_w,
            storage_height: storage_h,
            skip_mask: &encode_result.optimized_skip_mask, // Use optimized skip mask
            residual_data: &encode_result.residual_data,
            inter_quality: 50,
            skip_residuals: false,

        })
        .unwrap();

        let decoded = decoded_yuv
            .to_rgb24()
            .expect("decoded yuv -> rgb conversion failed");

        // Verify reconstruction is reasonable (exact match not expected due to quantization)
        assert_eq!(
            decoded.len(),
            curr_rgb.len(),
            "Decoded frame should have correct size"
        );

        // Check a few pixels to ensure they are not all zero or completely wrong
        let mut diff_sum = 0u64;
        for i in 0..decoded.len() {
            diff_sum += (decoded[i] as i32 - curr_rgb[i] as i32).abs() as u64;
        }
        let avg_diff = diff_sum as f64 / decoded.len() as f64;
        println!("Average reconstruction error: {}", avg_diff);
        assert!(
            avg_diff < 10.0,
            "Average reconstruction error too high: {}",
            avg_diff
        );
    }

    #[test]
    fn test_intra_encode_decode_roundtrip() {
        let storage_w = 32u32;
        let storage_h = 32u32;

        // Build a YUV frame with some variation.
        let mut y = vec![128u8; (storage_w * storage_h) as usize];
        let u = vec![100u8; ((storage_w / 2) * (storage_h / 2)) as usize];
        let v = vec![150u8; ((storage_w / 2) * (storage_h / 2)) as usize];
        for (i, val) in y.iter_mut().enumerate() {
            *val = (50 + (i % 200) as u8).min(255);
        }
        let curr_yuv = Yuv420Frame::from_planes(
            storage_w as usize, storage_h as usize, y.clone(), u.clone(), v.clone(),
        )
        .expect("build yuv frame");

        let result = ResidualEncoder::encode_intra(
            &curr_yuv, storage_w, storage_h, 80,
        )
        .expect("encode_intra");

        let decoded = ResidualDecoder::decode_intra(
            &result.residual_data, storage_w, storage_h, 80,
        )
        .expect("decode_intra");

        // Encoder recon and decoder output must match exactly.
        assert_eq!(result.recon_yuv.y_plane(), decoded.y_plane());
        assert_eq!(result.recon_yuv.u_plane(), decoded.u_plane());
        assert_eq!(result.recon_yuv.v_plane(), decoded.v_plane());

        // Reconstruction should be close to the original.
        let avg_err: f64 = y.iter().zip(decoded.y_plane().iter())
            .map(|(&a, &b)| (a as i32 - b as i32).abs() as f64)
            .sum::<f64>() / y.len() as f64;
        assert!(avg_err < 15.0, "avg Y reconstruction error too high: {avg_err}");
    }
}
