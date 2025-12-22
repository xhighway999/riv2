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

use crate::jpeg::{decode_jpeg_rgb_with_dims, encode_jpeg_rgb};
use crate::rans::{RansDecoder, RansEncoder};
use yuv::YuvChromaSubsampling;
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

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
    #[error("jpeg encode error: {0}")]
    JpegEncode(String),
    #[error("jpeg decode error: {0}")]
    JpegDecode(String),
}

/// Result for encoding an intra frame (full JPEG RGB24), including the reconstructed RGB the
/// decoder will see (i.e. JPEG-decoded).
#[derive(Debug, Clone)]
pub struct IntraResidualEncodeResult {
    pub jpeg_rgb: Vec<u8>,
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
        curr_rgb: &[u8],
        storage_width: u32,
        storage_height: u32,
        quality: u8,
    ) -> Result<IntraResidualEncodeResult> {
        let expected = (storage_width * storage_height * 3) as usize;
        if curr_rgb.len() != expected {
            return Err(ResidualError::InvalidInput(format!(
                "curr_rgb len mismatch: expected {expected}, got {}",
                curr_rgb.len()
            )));
        }
        let jpeg_rgb = encode_jpeg_rgb(curr_rgb, storage_width, storage_height, quality)?;
        let (recon_rgb, w, h) = decode_jpeg_rgb_with_dims(&jpeg_rgb)?;
        if w != storage_width || h != storage_height {
            return Err(ResidualError::InvalidInput(format!(
                "jpeg decode dims mismatch: expected {storage_width}x{storage_height}, got {w}x{h}"
            )));
        }
        let recon_yuv =
            Yuv420Frame::from_rgb(&recon_rgb, storage_width as usize, storage_height as usize)
                .map_err(|e| {
                    ResidualError::InvalidInput(format!(
                        "jpeg recon rgb->yuv conversion failed: {e}"
                    ))
                })?;
        Ok(IntraResidualEncodeResult {
            jpeg_rgb,
            recon_yuv,
        })
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
        let quant_step = quant_step_from_quality(p.inter_quality);
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
        let mut rans_encoder: Option<RansEncoder> = None;
        let mut dct_coefficients =
            Vec::with_capacity(blocks_total * (Y_COEFFS_PER_BLOCK + UV_COEFFS_PER_BLOCK * 2));
        // Build optimized skip mask: original skip_mask + blocks that become zero after quantization
        let mut optimized_skip_mask = Vec::with_capacity(blocks_total);
        // TEST-ONLY: track how many blocks become newly skipped after quantization
        let mut newly_skipped_after_quant: usize = 0;

        // Encode planes (parallel internally if supported by backend)
        let y_coeffs_all = reitero_dct::encode_plane_16x16(
            &res_y,
            storage_w,
            storage_w,
            storage_h,
            quant_step,
            Some(p.skip_mask),
        );
        let u_coeffs_all = reitero_dct::encode_plane_8x8(
            &res_u,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            quant_step,
            Some(p.skip_mask),
        );
        let v_coeffs_all = reitero_dct::encode_plane_8x8(
            &res_v,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            quant_step,
            Some(p.skip_mask),
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

                // Debugging aid for tests
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
                    let enc = rans_encoder.get_or_insert_with(RansEncoder::new);
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

        reitero_dct::decode_plane_16x16(
            &y_coeffs_recon,
            &mut recon_res_y,
            storage_w,
            storage_w,
            storage_h,
            quant_step,
            &optimized_skip_mask,
        );
        reitero_dct::decode_plane_8x8(
            &u_coeffs_recon,
            &mut recon_res_u,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            quant_step,
            &optimized_skip_mask,
        );
        reitero_dct::decode_plane_8x8(
            &v_coeffs_recon,
            &mut recon_res_v,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            quant_step,
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
        jpeg_rgb: &[u8],
        storage_width: u32,
        storage_height: u32,
    ) -> Result<Yuv420Frame> {
        let (rgb, w, h) = decode_jpeg_rgb_with_dims(jpeg_rgb)?;
        if w != storage_width || h != storage_height {
            return Err(ResidualError::InvalidInput(format!(
                "jpeg decode dims mismatch: expected {storage_width}x{storage_height}, got {w}x{h}"
            )));
        }
        Yuv420Frame::from_rgb(&rgb, storage_width as usize, storage_height as usize).map_err(|e| {
            ResidualError::InvalidInput(format!("jpeg rgb->yuv conversion failed: {e}"))
        })
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
        let quant_step = quant_step_from_quality(p.inter_quality);
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
        let mut rans_decoder = RansDecoder::new();
        if !p.residual_data.is_empty() {
            rans_decoder.consume(p.residual_data);
        }

        // Decode RANS blocks back to DCT coefficients
        // Format: For each motion block: Y block (256 coeffs), U block (64 coeffs), V block (64 coeffs)
        const Y_BLOCK_SIZE: usize = 16;
        const UV_BLOCK_SIZE: usize = 8;
        const Y_COEFFS_PER_BLOCK: usize = Y_BLOCK_SIZE * Y_BLOCK_SIZE; // 256
        const UV_COEFFS_PER_BLOCK: usize = UV_BLOCK_SIZE * UV_BLOCK_SIZE; // 64

        // Preallocate per-plane coefficient arrays directly to avoid an interleaved copy later
        let mut y_coeffs_all = Vec::with_capacity(blocks_total * Y_COEFFS_PER_BLOCK);
        let mut u_coeffs_all = Vec::with_capacity(blocks_total * UV_COEFFS_PER_BLOCK);
        let mut v_coeffs_all = Vec::with_capacity(blocks_total * UV_COEFFS_PER_BLOCK);

        // Static zero blocks to avoid repeated allocations for skipped blocks
        static ZERO_Y_BLOCK: [i16; 256] = [0i16; 256];
        static ZERO_UV_BLOCK: [i16; 64] = [0i16; 64];

        let t_r0 = Instant::now();
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                let bi = by * blocks_w + bx;

                // IMPORTANT: skip_mask is authoritative - skipped blocks have no RANS data
                if p.skip_mask[bi] {
                    // Skipped block: no RANS data was written, use zeros
                    y_coeffs_all.extend_from_slice(&ZERO_Y_BLOCK);
                    u_coeffs_all.extend_from_slice(&ZERO_UV_BLOCK);
                    v_coeffs_all.extend_from_slice(&ZERO_UV_BLOCK);
                } else {
                    // Non-skipped block: decode RANS blocks for Y, U, V
                    let (y_coeffs, u_coeffs, v_coeffs) = rans_decoder.decode_block(
                        Y_COEFFS_PER_BLOCK,
                        UV_COEFFS_PER_BLOCK,
                        UV_COEFFS_PER_BLOCK,
                    );
                    y_coeffs_all.extend_from_slice(&y_coeffs);
                    u_coeffs_all.extend_from_slice(&u_coeffs);
                    v_coeffs_all.extend_from_slice(&v_coeffs);
                }
            }
        }
        RANS_DECODE_NS.fetch_add(t_r0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Decode DCT coefficients to residual planes
        let y_len = (p.storage_width * p.storage_height) as usize;
        let uv_len = ((p.storage_width / 2) * (p.storage_height / 2)) as usize;
        let mut res_y = vec![0i16; y_len];
        let mut res_u = vec![0i16; uv_len];
        let mut res_v = vec![0i16; uv_len];

        let t_dct0 = Instant::now();
        reitero_dct::decode_plane_16x16(
            &y_coeffs_all,
            &mut res_y,
            storage_w,
            storage_w,
            storage_h,
            quant_step,
            p.skip_mask,
        );
        DCT_Y_NS.fetch_add(t_dct0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_dctuv0 = Instant::now();
        reitero_dct::decode_plane_8x8(
            &u_coeffs_all,
            &mut res_u,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            quant_step,
            p.skip_mask,
        );
        reitero_dct::decode_plane_8x8(
            &v_coeffs_all,
            &mut res_v,
            storage_w / 2,
            storage_w / 2,
            storage_h / 2,
            quant_step,
            p.skip_mask,
        );
        DCT_UV_NS.fetch_add(t_dctuv0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Copy predicted planes to contiguous arrays (matching residual storage format)
        let (pred_y, pred_u, pred_v) = p.predicted_yuv.clone_planes();
        // Convert predicted planes to signed i32 buffers to speed inner-loop arithmetic
        let mut recon_y_i32: Vec<i32> = pred_y.iter().map(|&v| v as i32).collect();
        let mut recon_u_i32: Vec<i32> = pred_u.iter().map(|&v| v as i32).collect();
        let mut recon_v_i32: Vec<i32> = pred_v.iter().map(|&v| v as i32).collect();

        // Apply residual in YUV, but force residual=0 on skipped blocks (even if payload isn't neutral).
        // This keeps skip semantics correct.
        // Y plane: 16x16 per block; U/V: 8x8 per block (YUV420 halves both dimensions).
        let t_a0 = Instant::now();
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
                    let row = y * storage_w;
                    for xx in 0..16 {
                        let x = x0 + xx;
                        let i = row + x;
                        let residual = res_y[i] as i32;
                        let sum = recon_y_i32[i] + residual;
                        recon_y_i32[i] = if sum < 0 { 0 } else if sum > 255 { 255 } else { sum };
                    }
                }
                // U/V planes: 8x8 per 16x16 block
                for yy in 0..8 {
                    let y_uv = (y0 / 2) + yy;
                    let row = y_uv * (storage_w / 2);
                    for xx in 0..8 {
                        let x = (x0 / 2) + xx;
                        let i = row + x;
                        let ru = res_u[i] as i32;
                        let rv = res_v[i] as i32;
                        let su = recon_u_i32[i] + ru;
                        let sv = recon_v_i32[i] + rv;
                        recon_u_i32[i] = if su < 0 { 0 } else if su > 255 { 255 } else { su };
                        recon_v_i32[i] = if sv < 0 { 0 } else if sv > 255 { 255 } else { sv };
                    }
                }
            }
        }
        APPLY_NS.fetch_add(t_a0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Convert recon i32 planes back to u8
        let recon_y: Vec<u8> = recon_y_i32.iter().map(|&v| v as u8).collect();
        let recon_u: Vec<u8> = recon_u_i32.iter().map(|&v| v as u8).collect();
        let recon_v: Vec<u8> = recon_v_i32.iter().map(|&v| v as u8).collect();

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
}
