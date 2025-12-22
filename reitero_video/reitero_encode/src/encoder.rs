use crate::config::{EncoderConfig, KeyframeInterval};
use crate::error::{EncodeError, Result};
use crate::motion::{MotionVector, hex_search_yuv_sad_with_scores};
use crate::rdo::{RdoContext, RdoTelemetry};
use crate::writer::VideoWriter;
use reitero_residual::{
    InterResidualEncodeParams, MvCodedBlock, MvMode, MvRansEncoder, ResidualEncoder,
    derive_mv_predictors_with_stats, gather_mv_neighbor_set, mv_class_from_magnitude,
};
use reitero_video_common::{FrameType, PackedFrame, VideoHeader, Yuv420Frame, build_predicted};

/// Raw video frame data (input)
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
}

impl Frame {
    pub fn new(data: Vec<u8>, width: u32, height: u32, timestamp: u64) -> Self {
        Self {
            data,
            width,
            height,
            timestamp,
        }
    }
}

/// Video encoder for custom ReItero format
pub struct Encoder<W: VideoWriter> {
    config: EncoderConfig,
    header: VideoHeader,
    writer: W,
    frame_count: u64,
    header_position: u64,
    /// Previous reconstructed frame (YUV420). This matches the decoder's reference,
    /// so inter residuals must be computed against this (not the original source frame).
    prev_recon_yuv: Option<Yuv420Frame>,
    /// RANS encoder for motion vectors (lives for entire video, maintains probabilities across frames)
    mv_rans_encoder: MvRansEncoder,
    /// Previous frame's motion vectors (for temporal MV prediction). None for first frame
    /// or immediately after an intra frame.
    prev_mvs: Option<Vec<MotionVector>>,
}

/// Per-frame encoding stats useful for debugging/compression tuning.
#[derive(Debug, Clone)]
pub struct EncodeFrameStats {
    pub frame_index: u64,
    pub timestamp_ms: u64,
    pub frame_type: FrameType,
    pub total_bytes: usize,
    pub mv_bytes: usize,
    pub mv_raw_bytes: usize,
    pub residual_jpeg_bytes: usize,
    pub storage_width: u32,
    pub storage_height: u32,
    pub blocks_total: usize,
    pub blocks_skipped: usize,
    /// Number of motion-vector blocks where the predicted MV matched perfectly
    /// (ddx == 0 && ddy == 0 after delta coding).
    pub mv_zero_delta_blocks: usize,
    pub mv_mode_counts: [usize; 7],
    pub mv_new_zero_pre_bias: usize,
    pub mv_new_zero_post_bias: usize,
    pub mv_new_zero_axes_x: usize,
    pub mv_new_zero_axes_y: usize,
    /// For NEW-coded blocks, which base was actually selected for delta coding.
    /// Index order: nearest, near, top-right, top-left, temporal.
    pub mv_new_base_counts: [usize; 5],
    /// For NEW-coded blocks, which (integer) predictor base would have minimized L1 distance.
    /// Index order: nearest, near, top-right, top-left, temporal.
    pub mv_new_best_ref_counts: [usize; 5],
    /// For NEW-coded blocks: total L1(px) saved if we could always pick the best ref above
    /// compared to the current 1-bit selection between nearest/near.
    pub mv_new_best_ref_l1_saved_sum: u64,
    pub mv_new_blocks: usize,
    pub mv_new_delta_count: usize,
    pub mv_new_delta_mag_sum: f64,
    pub mv_new_delta_mag_sq_sum: f64,
    pub mv_class_histogram: [u64; 11],
    pub mv_bias_dx: i8,
    pub mv_bias_dy: i8,
    pub mv_candidate_unique_total: usize,
    pub mv_candidate_unique_samples: usize,
    pub mv_candidate_unique_min: usize,
    pub mv_candidate_unique_max: usize,
    /// For non-zero MV blocks, counts of which raw neighbor exactly matched the chosen MV.
    /// Index order: left, top, top-right, top-left, temporal, none.
    pub mv_match_source_counts: [usize; 6],
    /// Number of blocks considered in `mv_match_source_counts` (i.e., non-zero MV mode blocks).
    pub mv_match_nonzero_blocks: usize,
    /// Among non-zero MV blocks: how often MV matched any spatial neighbor (left/top/top-right/top-left).
    pub mv_match_any_spatial: usize,
    /// Among non-zero MV blocks: how often MV matched temporal (same block previous frame).
    pub mv_match_any_temporal: usize,
    pub rle_stats: Option<reitero_residual::RleCompressionStats>,
    /// Size of residuals after zigzag (before RANS), in bytes
    pub resi_raw: Option<usize>,
    /// Size of residuals after RANS encoding, in bytes
    pub resi_rans: Option<usize>,
    /// Lambda used for this frame's skip RDO.
    pub rdo_lambda: f64,
    /// Number of blocks evaluated by RDO before residual quantization may add more skips.
    pub rdo_blocks_evaluated: usize,
    /// How many blocks the RDO pass elected to skip (pre-quantization).
    pub rdo_blocks_selected_skip: usize,
    pub rdo_forced_fractional: usize,
    pub rdo_forced_threshold: usize,
    pub rdo_forced_disabled: usize,
    pub rdo_avg_skip_cost: f64,
    pub rdo_avg_coded_cost: f64,
}

impl<W: VideoWriter> Encoder<W> {
    #[inline]
    fn est_new_base_selector_bits(base: u8) -> u32 {
        // Mirrors the NEW-base decision tree in `reitero_residual::mv_rans`.
        // 0=nearest, 1=near, 2=top-right, 3=top-left, 4=temporal.
        match base {
            0 => 1,
            1 => 2,
            2 => 3,
            3 => 4,
            4 => 4,
            _ => 4,
        }
    }

    #[inline]
    fn est_new_component_bits(delta: i8) -> u32 {
        // Roughly approximate the number of binary decisions emitted by the MV component coder.
        // - class symbol: fixed 4-way binary search over 0..=MV_MAX_CLASS
        // - if class>0: sign bit + (class-1) magnitude bits (class=1 emits sign only)
        let mag = (i16::from(delta).abs()) as u16;
        let class = mv_class_from_magnitude(mag) as u32;
        if class == 0 {
            4
        } else {
            // 4 (class tree) + 1 (sign) + (class-1) (magnitude bits) = 4 + class
            4 + class
        }
    }

    #[inline]
    fn est_new_delta_bits(dx: i8, dy: i8) -> u32 {
        Self::est_new_component_bits(dx) + Self::est_new_component_bits(dy)
    }

    /// Create a new encoder with the given configuration and writer
    pub fn new(config: EncoderConfig, writer: W) -> Result<Self> {
        config.validate()?;

        let header = VideoHeader::new(
            config.display_width,
            config.display_height,
            config.storage_width,
            config.storage_height,
            config.fps,
        );

        let mut encoder = Self {
            config,
            header,
            writer,
            frame_count: 0,
            header_position: 0,
            prev_recon_yuv: None,
            mv_rans_encoder: MvRansEncoder::new(),
            prev_mvs: None,
        };

        // Write header immediately (with placeholder frame_count = 0)
        encoder.write_header()?;

        Ok(encoder)
    }

    /// Write the video header
    fn write_header(&mut self) -> Result<()> {
        // Store position where header is written
        self.header_position = self.writer.position();

        // TODO: Serialize header into bytes
        let header_bytes = self.serialize_header()?;

        self.writer.write(&header_bytes)?;

        Ok(())
    }

    /// Update the header with the final frame count
    fn update_header(&mut self) -> Result<()> {
        // Save current position (end of file)
        let end_position = self.writer.position();

        // Seek back to header position
        self.writer.seek(self.header_position)?;

        // Serialize updated header
        let header_bytes = self.serialize_header()?;

        // Write updated header
        self.writer.write(&header_bytes)?;

        // Seek back to end position
        self.writer.seek(end_position)?;

        Ok(())
    }

    /// Serialize header to bytes (internal implementation)
    fn serialize_header(&self) -> Result<Vec<u8>> {
        Ok(self.header.to_bytes())
    }

    /// Encode a single frame into the custom format
    pub fn encode_frame(&mut self, frame: Frame) -> Result<()> {
        let _ = self.encode_frame_with_stats(frame)?;
        Ok(())
    }

    /// Encode a single frame and return per-frame stats (sizes, skip rate, etc).
    pub fn encode_frame_with_stats(&mut self, frame: Frame) -> Result<EncodeFrameStats> {
        // Validate frame dimensions match config
        if frame.width != self.config.display_width || frame.height != self.config.display_height {
            return Err(EncodeError::FrameError(format!(
                "Frame dimensions ({}, {}) do not match config ({}, {})",
                frame.width, frame.height, self.config.display_width, self.config.display_height
            )));
        }

        let frame_index = self.frame_count;
        let (encoded_data, stats) =
            self.encode_frame_data_with_stats(&frame.data, frame.timestamp, frame_index)?;

        // Write encoded frame data
        self.writer.write(&encoded_data)?;

        // Update counters/header (critical: CLI uses this method directly).
        self.frame_count += 1;
        self.header.frame_count = self.frame_count;

        Ok(stats)
    }

    fn encode_frame_data_with_stats(
        &mut self,
        data: &[u8],
        timestamp: u64,
        frame_index: u64,
    ) -> Result<(Vec<u8>, EncodeFrameStats)> {
        // Determine frame type based on keyframe interval configuration
        let mut frame_type = match self.config.keyframe_interval {
            KeyframeInterval::AllIntra => {
                // Every frame is an I-frame
                FrameType::Intra
            }
            KeyframeInterval::Automatic => {
                // TODO: Implement automatic keyframe detection based on content
                // For now, use a sensible default interval
                if self.frame_count % 30 == 0 {
                    FrameType::Intra
                } else {
                    FrameType::Inter
                }
            }
            KeyframeInterval::Fixed(interval) => {
                // I-frame every N frames
                if self.frame_count % interval == 0 {
                    FrameType::Intra
                } else {
                    FrameType::Inter
                }
            }
        };

        // If we don't yet have a reference frame, we must emit an intra frame.
        if self.prev_recon_yuv.is_none() {
            frame_type = FrameType::Intra;
        }

        let curr_padded = pad_rgb24(
            data,
            self.config.display_width as usize,
            self.config.display_height as usize,
            self.config.storage_width as usize,
            self.config.storage_height as usize,
        )?;
        let curr_yuv = Yuv420Frame::from_rgb(
            &curr_padded,
            self.config.storage_width as usize,
            self.config.storage_height as usize,
        )
        .map_err(|e| EncodeError::FrameError(format!("curr rgb->yuv conversion failed: {e}")))?;

        match frame_type {
            FrameType::Intra => {
                // Encode full frame as JPEG RGB24 and reconstruct reference by decoding our own JPEG
                // (accounts for compression loss). This is shared with the decoder via reitero_residual.
                let intra = ResidualEncoder::encode_intra(
                    &curr_padded,
                    self.config.storage_width,
                    self.config.storage_height,
                    self.config.intra_quality,
                )
                .map_err(|e| {
                    EncodeError::FrameError(format!("Residual intra encode error: {e}"))
                })?;
                self.prev_recon_yuv = Some(intra.recon_yuv.clone());
                let bytes = PackedFrame::new_intra(intra.jpeg_rgb, timestamp).to_bytes();
                let blocks_total = (self.config.storage_width as usize / 16)
                    * (self.config.storage_height as usize / 16);
                // After an intra frame there is no meaningful previous MV field.
                self.prev_mvs = None;
                // Reset MV entropy contexts so seekers can restart from this key frame.
                self.mv_rans_encoder.reset_contexts();
                let stats = EncodeFrameStats {
                    frame_index,
                    timestamp_ms: timestamp,
                    frame_type: FrameType::Intra,
                    total_bytes: bytes.len(),
                    mv_bytes: 0,
                    mv_raw_bytes: 0,
                    residual_jpeg_bytes: bytes.len().saturating_sub(8 + 1 + 4),
                    storage_width: self.config.storage_width,
                    storage_height: self.config.storage_height,
                    blocks_total,
                    blocks_skipped: 0,
                    mv_zero_delta_blocks: 0,
                    mv_mode_counts: [0; 7],
                    mv_new_zero_pre_bias: 0,
                    mv_new_zero_post_bias: 0,
                    mv_new_zero_axes_x: 0,
                    mv_new_zero_axes_y: 0,
                    mv_new_base_counts: [0; 5],
                    mv_new_best_ref_counts: [0; 5],
                    mv_new_best_ref_l1_saved_sum: 0,
                    mv_new_blocks: 0,
                    mv_new_delta_count: 0,
                    mv_new_delta_mag_sum: 0.0,
                    mv_new_delta_mag_sq_sum: 0.0,
                    mv_class_histogram: [0; 11],
                    mv_bias_dx: 0,
                    mv_bias_dy: 0,
                    mv_candidate_unique_total: 0,
                    mv_candidate_unique_samples: 0,
                    mv_candidate_unique_min: 0,
                    mv_candidate_unique_max: 0,
                    mv_match_source_counts: [0; 6],
                    mv_match_nonzero_blocks: 0,
                    mv_match_any_spatial: 0,
                    mv_match_any_temporal: 0,
                    rle_stats: None, // Intra frames use JPEG, not RANS
                    resi_raw: None,  // Intra frames use JPEG, not residuals
                    resi_rans: None,
                    rdo_lambda: 0.0,
                    rdo_blocks_evaluated: 0,
                    rdo_blocks_selected_skip: 0,
                    rdo_forced_fractional: 0,
                    rdo_forced_threshold: 0,
                    rdo_forced_disabled: 0,
                    rdo_avg_skip_cost: 0.0,
                    rdo_avg_coded_cost: 0.0,
                };
                Ok((bytes, stats))
            }
            FrameType::Inter => {
                let prev = self
                    .prev_recon_yuv
                    .as_ref()
                    .expect("prev_recon_yuv checked above");

                // Compute candidate per-block skip decisions (only for integer-aligned MVs with no fractional offset).
                let blocks_w = (self.config.storage_width as usize) / 16;
                let blocks_h = (self.config.storage_height as usize) / 16;
                let block_area_bytes = 16usize * 16usize * 3usize;
                let rdo_context = RdoContext::new(
                    self.config.inter_quality,
                    block_area_bytes,
                    self.config.skip_threshold,
                    self.config.rdo_lambda_mult,
                );

                // Motion vectors (hex RGB SAD + half-pel), and per-block best SAD scores.
                // We reuse the SAD scores for skip decisions to avoid an extra full SAD pass.
                let (mvs, best_sads) = hex_search_yuv_sad_with_scores(
                    prev,
                    &curr_yuv,
                    self.config.storage_width as usize,
                    self.config.storage_height as usize,
                    self.config.search_range,
                    self.prev_mvs.as_deref(),
                    self.config.me_zero_mv_threshold,
                    self.config.me_predictor_threshold,
                    rdo_context.lambda(),
                );
                let predicted = build_predicted(
                    prev,
                    self.config.storage_width as usize,
                    self.config.storage_height as usize,
                    &mvs,
                );

                let mut rdo_telemetry = RdoTelemetry::new(rdo_context.lambda());
                let mut skip_mask: Vec<bool> = Vec::with_capacity(mvs.len());
                let mut raw_deltas: Vec<(i8, i8)> = Vec::with_capacity(mvs.len());
                let mut block_flags: Vec<u8> = Vec::with_capacity(mvs.len());
                let mut block_modes: Vec<MvMode> = Vec::with_capacity(mvs.len());
                let mut new_base: Vec<u8> = Vec::with_capacity(mvs.len());
                let mut coded_prefix: Vec<MotionVector> = Vec::with_capacity(mvs.len());
                let mut mode_counts = [0usize; 7];
                let mut new_zero_pre_bias = 0usize;
                let mut new_blocks = 0usize;
                let mut new_base_counts = [0usize; 5];
                let mut new_best_ref_counts = [0usize; 5];
                let mut new_best_ref_l1_saved_sum: u64 = 0;
                let mut candidate_unique_total = 0usize;
                let mut candidate_unique_samples = 0usize;
                let mut candidate_unique_min = usize::MAX;
                let mut candidate_unique_max = 0usize;
                let mut mv_match_source_counts = [0usize; 6];
                let mut mv_match_nonzero_blocks = 0usize;
                let mut mv_match_any_spatial = 0usize;
                let mut mv_match_any_temporal = 0usize;

                for (i, mv) in mvs.iter().enumerate() {
                    let bx = i % blocks_w;
                    let by = i / blocks_w;

                    let frac_x_code = mv.subpixel_x();
                    let frac_y_code = mv.subpixel_y();
                    let integer_aligned = frac_x_code == 0 && frac_y_code == 0;
                    let rdo_decision = rdo_context.decide(best_sads[i], integer_aligned);
                    rdo_telemetry.record(&rdo_decision);
                    let skip_candidate = rdo_decision.skip;
                    skip_mask.push(skip_candidate);

                    let (predictors, unique_candidates) = derive_mv_predictors_with_stats(
                        &coded_prefix,
                        self.prev_mvs.as_deref(),
                        blocks_w,
                        blocks_h,
                        bx,
                        by,
                    );
                    candidate_unique_total += unique_candidates;
                    candidate_unique_samples += 1;
                    candidate_unique_min = candidate_unique_min.min(unique_candidates);
                    candidate_unique_max = candidate_unique_max.max(unique_candidates);

                    let zero_eligible =
                        mv.dx() == 0 && mv.dy() == 0 && frac_x_code == 0 && frac_y_code == 0;

                    let neigh = gather_mv_neighbor_set(
                        &coded_prefix,
                        self.prev_mvs.as_deref(),
                        blocks_w,
                        blocks_h,
                        bx,
                        by,
                    );

                    // Attribute exact matches to raw neighbor sources for non-zero blocks.
                    // This is intentionally independent of Nearest/Near predictor dedup/order.
                    if !zero_eligible {
                        mv_match_nonzero_blocks += 1;
                        let actual = (mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y());

                        let spatial_any = neigh.left == Some(actual)
                            || neigh.top == Some(actual)
                            || neigh.top_right == Some(actual)
                            || neigh.top_left == Some(actual);
                        if spatial_any {
                            mv_match_any_spatial += 1;
                        }
                        if neigh.temporal == Some(actual) {
                            mv_match_any_temporal += 1;
                        }

                        let src_idx = if neigh.left == Some(actual) {
                            0
                        } else if neigh.top == Some(actual) {
                            1
                        } else if neigh.top_right == Some(actual) {
                            2
                        } else if neigh.top_left == Some(actual) {
                            3
                        } else if neigh.temporal == Some(actual) {
                            4
                        } else {
                            5
                        };
                        mv_match_source_counts[src_idx] += 1;
                    }

                    let mode = if zero_eligible {
                        MvMode::Zero
                    } else {
                        let mv_x_hp = mv.dx_hp();
                        let mv_y_hp = mv.dy_hp();

                        let pred_nearest_x_hp =
                            (predictors.nearest.0 as i32) * 2 + (predictors.nearest.2 as i32);
                        let pred_nearest_y_hp =
                            (predictors.nearest.1 as i32) * 2 + (predictors.nearest.3 as i32);

                        let pred_near_x_hp =
                            (predictors.near.0 as i32) * 2 + (predictors.near.2 as i32);
                        let pred_near_y_hp =
                            (predictors.near.1 as i32) * 2 + (predictors.near.3 as i32);

                        let pred_temporal_x_hp =
                            (predictors.temporal.0 as i32) * 2 + (predictors.temporal.2 as i32);
                        let pred_temporal_y_hp =
                            (predictors.temporal.1 as i32) * 2 + (predictors.temporal.3 as i32);

                        let pred_top_right = neigh.top_right.map(|p| {
                            (
                                (p.0 as i32) * 2 + (p.2 as i32),
                                (p.1 as i32) * 2 + (p.3 as i32),
                            )
                        });
                        let pred_top_left = neigh.top_left.map(|p| {
                            (
                                (p.0 as i32) * 2 + (p.2 as i32),
                                (p.1 as i32) * 2 + (p.3 as i32),
                            )
                        });

                        if mv_x_hp == pred_nearest_x_hp && mv_y_hp == pred_nearest_y_hp {
                            MvMode::Nearest
                        } else if mv_x_hp == pred_near_x_hp && mv_y_hp == pred_near_y_hp {
                            MvMode::Near
                        } else if pred_top_right.is_some_and(|(x, y)| mv_x_hp == x && mv_y_hp == y)
                        {
                            MvMode::TopRight
                        } else if pred_top_left.is_some_and(|(x, y)| mv_x_hp == x && mv_y_hp == y) {
                            MvMode::TopLeft
                        } else if mv_x_hp == pred_temporal_x_hp && mv_y_hp == pred_temporal_y_hp {
                            MvMode::Temporal
                        } else {
                            MvMode::New
                        }
                    };

                    let mut chosen_new_base: u8 = 0;
                    let mut raw_delta = (0i8, 0i8);
                    if mode == MvMode::New {
                        let actual_dx = mv.dx() as i16;
                        let actual_dy = mv.dy() as i16;
                        let nearest_cost = (actual_dx - predictors.nearest.0 as i16).abs()
                            + (actual_dy - predictors.nearest.1 as i16).abs();
                        let near_cost = (actual_dx - predictors.near.0 as i16).abs()
                            + (actual_dy - predictors.near.1 as i16).abs();

                        // Candidates: {Nearest, Near, TopRight, TopLeft, Temporal}.
                        // Integer-only because NEW deltas code only dx/dy; fractional is stored in flags.
                        let candidates: [Option<(i16, i16, u32)>; 5] = [
                            Some((
                                predictors.nearest.0 as i16,
                                predictors.nearest.1 as i16,
                                nearest_cost as u32,
                            )),
                            Some((
                                predictors.near.0 as i16,
                                predictors.near.1 as i16,
                                near_cost as u32,
                            )),
                            neigh.top_right.map(|p| {
                                let c =
                                    (actual_dx - p.0 as i16).abs() + (actual_dy - p.1 as i16).abs();
                                (p.0 as i16, p.1 as i16, c as u32)
                            }),
                            neigh.top_left.map(|p| {
                                let c =
                                    (actual_dx - p.0 as i16).abs() + (actual_dy - p.1 as i16).abs();
                                (p.0 as i16, p.1 as i16, c as u32)
                            }),
                            Some((
                                predictors.temporal.0 as i16,
                                predictors.temporal.1 as i16,
                                {
                                    let c = (actual_dx - predictors.temporal.0 as i16).abs()
                                        + (actual_dy - predictors.temporal.1 as i16).abs();
                                    c as u32
                                },
                            )),
                        ];

                        // Telemetry: ideal L1 base (independent of the actual base chooser).
                        let mut best_l1_idx = 0usize;
                        let mut best_l1_cost = nearest_cost as u32;
                        for (idx, cand) in candidates.iter().enumerate() {
                            if let Some((_, _, c)) = cand {
                                if *c < best_l1_cost {
                                    best_l1_cost = *c;
                                    best_l1_idx = idx;
                                }
                            }
                        }
                        new_best_ref_counts[best_l1_idx] += 1;

                        // Savings vs the old 1-bit selector restricted to {Nearest, Near}.
                        let current_cost = (nearest_cost.min(near_cost)) as u32;
                        if best_l1_cost < current_cost {
                            new_best_ref_l1_saved_sum += (current_cost - best_l1_cost) as u64;
                        }

                        // Actual choice: approximate bit cost (selector + delta coding).
                        // This is a crude proxy for RANS cost, but it prevents picking a
                        // more-expensive base when the delta shrink doesn't pay for it.
                        let mut best_rate_idx = 0usize;
                        let mut best_rate_cost = u32::MAX;
                        let mut best_base =
                            (predictors.nearest.0 as i16, predictors.nearest.1 as i16);
                        for (idx, cand) in candidates.iter().enumerate() {
                            let Some((bx, by, _)) = cand else { continue };
                            let ddx = (actual_dx - *bx).clamp(-128, 127) as i8;
                            let ddy = (actual_dy - *by).clamp(-128, 127) as i8;
                            let selector_bits = Self::est_new_base_selector_bits(idx as u8);
                            let delta_bits = Self::est_new_delta_bits(ddx, ddy);
                            let cost = selector_bits + delta_bits;
                            if cost < best_rate_cost {
                                best_rate_cost = cost;
                                best_rate_idx = idx;
                                best_base = (*bx, *by);
                            }
                        }

                        chosen_new_base = best_rate_idx as u8;
                        new_base_counts[best_rate_idx] += 1;

                        let ddx = (actual_dx - best_base.0).clamp(-128, 127) as i8;
                        let ddy = (actual_dy - best_base.1).clamp(-128, 127) as i8;
                        raw_delta = (ddx, ddy);
                        new_blocks += 1;
                        if raw_delta.0 == 0 && raw_delta.1 == 0 {
                            new_zero_pre_bias += 1;
                        }
                    }

                    mode_counts[mode.as_u8() as usize] += 1;
                    raw_deltas.push(raw_delta);
                    block_modes.push(mode);
                    new_base.push(chosen_new_base);
                    let mut reconstructed_mv = *mv;
                    if mode == MvMode::Nearest {
                        reconstructed_mv = MotionVector::new(
                            predictors.nearest.0,
                            predictors.nearest.1,
                            predictors.nearest.2,
                            predictors.nearest.3,
                            mv.is_skip(),
                        );
                    } else if mode == MvMode::Near {
                        reconstructed_mv = MotionVector::new(
                            predictors.near.0,
                            predictors.near.1,
                            predictors.near.2,
                            predictors.near.3,
                            mv.is_skip(),
                        );
                    } else if mode == MvMode::TopRight {
                        if let Some((dx, dy, sx, sy)) = neigh.top_right {
                            reconstructed_mv = MotionVector::new(dx, dy, sx, sy, mv.is_skip());
                        }
                    } else if mode == MvMode::TopLeft {
                        if let Some((dx, dy, sx, sy)) = neigh.top_left {
                            reconstructed_mv = MotionVector::new(dx, dy, sx, sy, mv.is_skip());
                        }
                    } else if mode == MvMode::Temporal {
                        reconstructed_mv = MotionVector::new(
                            predictors.temporal.0,
                            predictors.temporal.1,
                            predictors.temporal.2,
                            predictors.temporal.3,
                            mv.is_skip(),
                        );
                    } else if mode == MvMode::New {
                        let (dx, dy) = raw_delta;
                        let base = match chosen_new_base {
                            0 => predictors.nearest,
                            1 => predictors.near,
                            2 => neigh.top_right.unwrap_or(predictors.nearest),
                            3 => neigh.top_left.unwrap_or(predictors.nearest),
                            4 => predictors.temporal,
                            _ => predictors.nearest,
                        };
                        // Reconstruct from the delta we will actually encode.
                        // Note: We use raw_delta which is (actual - base).clamp().
                        // The decoder will receive (raw_delta - bias) + bias = raw_delta (if no secondary clamping).
                        let recon_dx = (base.0 as i16 + dx as i16).clamp(-128, 127) as i8;
                        let recon_dy = (base.1 as i16 + dy as i16).clamp(-128, 127) as i8;
                        reconstructed_mv.set_dx(recon_dx);
                        reconstructed_mv.set_dy(recon_dy);
                    }
                    coded_prefix.push(reconstructed_mv);

                    let mut flags = if mode == MvMode::New {
                        mv.raw_flags() & 0x0F // bits 0-1: half x, bits 2-3: half y
                    } else {
                        0 // Implicit from predictor
                    };
                    if skip_candidate {
                        flags |= 0x40; // bit 6: skip flag
                    }
                    block_flags.push(flags);
                }
                let rdo_summary = rdo_telemetry.finalize();
                if candidate_unique_min == usize::MAX {
                    candidate_unique_min = 0;
                }
                // Residual atlas encode + self-reconstruction (shared with decoder via reitero_residual).
                // This generates optimized_skip_mask which is authoritative.
                let inter = ResidualEncoder::encode_inter(InterResidualEncodeParams {
                    curr_yuv: &curr_yuv,
                    predicted_yuv: &predicted,
                    storage_width: self.config.storage_width,
                    storage_height: self.config.storage_height,
                    skip_mask: &skip_mask,
                    inter_quality: self.config.inter_quality,
                })
                .map_err(|e| {
                    EncodeError::FrameError(format!("Residual inter encode error: {e}"))
                })?;

                // Update MV flags to reflect authoritative optimized_skip_mask
                // The optimized skip mask may include blocks that became zero after quantization
                for (i, &optimized_skip) in inter.optimized_skip_mask.iter().enumerate() {
                    if let Some(flags) = block_flags.get_mut(i) {
                        if optimized_skip {
                            *flags |= 0x40;
                        } else {
                            *flags &= !0x40;
                        }
                    }
                }

                let (bias_dx, bias_dy) = compute_delta_bias_for_new(&raw_deltas, &block_modes);
                let global_mv = MotionVector::new(bias_dx, bias_dy, 0, 0, false); // stored as per-frame delta bias

                let mut mv_class_histogram = [0u64; 11];
                let mut mv_new_zero_post_bias = 0usize;
                let mut mv_new_zero_axes_x = 0usize;
                let mut mv_new_zero_axes_y = 0usize;
                let mut mv_new_delta_mag_sum = 0.0f64;
                let mut mv_new_delta_mag_sq_sum = 0.0f64;
                let mut mv_blocks: Vec<MvCodedBlock> = Vec::with_capacity(mvs.len());
                for idx in 0..mvs.len() {
                    let (ddx, ddy) = raw_deltas[idx];
                    let (mut delta_x, mut delta_y) = (0i8, 0i8);
                    if block_modes[idx] == MvMode::New {
                        delta_x = (i16::from(ddx) - i16::from(bias_dx)).clamp(-128, 127) as i8;
                        delta_y = (i16::from(ddy) - i16::from(bias_dy)).clamp(-128, 127) as i8;
                        let dx = f64::from(delta_x);
                        let dy = f64::from(delta_y);
                        let mag = (dx * dx + dy * dy).sqrt();
                        mv_new_delta_mag_sum += mag;
                        mv_new_delta_mag_sq_sum += mag * mag;
                        let class_x = mv_class_from_magnitude(i16::from(delta_x).abs() as u16);
                        let class_y = mv_class_from_magnitude(i16::from(delta_y).abs() as u16);
                        mv_class_histogram[class_x as usize] += 1;
                        mv_class_histogram[class_y as usize] += 1;
                        if delta_x == 0 {
                            mv_new_zero_axes_x += 1;
                        }
                        if delta_y == 0 {
                            mv_new_zero_axes_y += 1;
                        }
                        if delta_x == 0 && delta_y == 0 {
                            mv_new_zero_post_bias += 1;
                        }
                    }
                    mv_blocks.push(MvCodedBlock {
                        mode: block_modes[idx],
                        new_base: new_base[idx],
                        delta_x,
                        delta_y,
                        flags: block_flags[idx],
                    });
                }

                // Encode motion vectors using RANS (maintains state across frames)
                // The encoder lives for the entire video, contexts persist across frames
                let mv_raw_bytes = mv_blocks.len() * 2 + new_blocks * 2; // mode+flags per block, +deltas for NEW
                let mv_rans = self
                    .mv_rans_encoder
                    .encode_frame_and_get_data(&mv_blocks, blocks_w, blocks_h);
                let mv_rans_bytes = mv_rans.len();

                self.prev_recon_yuv = Some(inter.recon_current_yuv.clone());
                // Store current MVs as temporal reference for next inter frame
                self.prev_mvs = Some(mvs.clone());
                let mv_bytes = mv_rans_bytes;
                // Residual data is already RANS-compressed, no need for additional DEFLATE
                let residual_bytes = inter.residual_data.len();
                let bytes = PackedFrame::new_inter_with_mv(
                    self.config.inter_quality,
                    global_mv,
                    mv_rans,
                    inter.residual_data,
                    timestamp,
                )
                .to_bytes();
                let blocks_total = blocks_w * blocks_h;
                // Use optimized skip mask for statistics (includes blocks that became zero after quantization)
                let blocks_skipped_optimized =
                    inter.optimized_skip_mask.iter().filter(|s| **s).count();

                // Count how many MV deltas are exactly zero (perfect prediction matches)
                let mv_zero_delta_blocks = block_modes
                    .iter()
                    .enumerate()
                    .filter(|(idx, mode)| match mode {
                        MvMode::New => {
                            let (dx, dy) = raw_deltas[*idx];
                            dx == 0 && dy == 0
                        }
                        _ => true,
                    })
                    .count();

                // Extract residual size metrics from RANS stats
                let (resi_raw, resi_rans_stats) = if let Some(ref rle_stats) = inter.rle_stats {
                    (
                        Some(rle_stats.raw_size_bytes),
                        Some(rle_stats.rle_size_bytes), // This is now RANS size, not RLE
                    )
                } else {
                    (None, None)
                };

                let stats = EncodeFrameStats {
                    frame_index,
                    timestamp_ms: timestamp,
                    frame_type: FrameType::Inter,
                    total_bytes: bytes.len(),
                    mv_bytes,
                    mv_raw_bytes,
                    residual_jpeg_bytes: residual_bytes,
                    storage_width: self.config.storage_width,
                    storage_height: self.config.storage_height,
                    blocks_total,
                    blocks_skipped: blocks_skipped_optimized,
                    mv_zero_delta_blocks,
                    mv_mode_counts: mode_counts,
                    mv_new_zero_pre_bias: new_zero_pre_bias,
                    mv_new_zero_post_bias,
                    mv_new_zero_axes_x,
                    mv_new_zero_axes_y,
                    mv_new_base_counts: new_base_counts,
                    mv_new_best_ref_counts: new_best_ref_counts,
                    mv_new_best_ref_l1_saved_sum: new_best_ref_l1_saved_sum,
                    mv_new_blocks: new_blocks,
                    mv_new_delta_count: new_blocks,
                    mv_new_delta_mag_sum,
                    mv_new_delta_mag_sq_sum,
                    mv_class_histogram,
                    mv_bias_dx: bias_dx,
                    mv_bias_dy: bias_dy,
                    mv_candidate_unique_total: candidate_unique_total,
                    mv_candidate_unique_samples: candidate_unique_samples,
                    mv_candidate_unique_min: candidate_unique_min,
                    mv_candidate_unique_max: candidate_unique_max,
                    mv_match_source_counts,
                    mv_match_nonzero_blocks,
                    mv_match_any_spatial,
                    mv_match_any_temporal,
                    rle_stats: inter.rle_stats,
                    resi_raw,
                    resi_rans: resi_rans_stats.or(Some(residual_bytes)),
                    rdo_lambda: rdo_summary.lambda,
                    rdo_blocks_evaluated: rdo_summary.evaluated_blocks,
                    rdo_blocks_selected_skip: rdo_summary.skip_chosen,
                    rdo_forced_fractional: rdo_summary.forced_fractional,
                    rdo_forced_threshold: rdo_summary.forced_threshold,
                    rdo_forced_disabled: rdo_summary.forced_disabled,
                    rdo_avg_skip_cost: rdo_summary.avg_skip_cost,
                    rdo_avg_coded_cost: rdo_summary.avg_coded_cost,
                };

                Ok((bytes, stats))
            }
        }
    }

    /// Finish encoding (finalize the video file)
    pub fn finish(&mut self) -> Result<()> {
        // Update header with final frame count
        self.update_header()?;

        // TODO: Write any final data (footer, index, etc.)
        let footer_bytes = self.serialize_footer()?;
        self.writer.write(&footer_bytes)?;

        self.writer.flush()?;

        Ok(())
    }

    /// Serialize footer to bytes (internal implementation)
    fn serialize_footer(&self) -> Result<Vec<u8>> {
        // TODO: Implement footer serialization for custom format
        Ok(Vec::new())
    }

    /// Get the current frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the encoder configuration
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Get the video header
    pub fn header(&self) -> &VideoHeader {
        &self.header
    }
}

fn compute_delta_bias_for_new(deltas: &[(i8, i8)], modes: &[MvMode]) -> (i8, i8) {
    let filtered: Vec<(i8, i8)> = deltas
        .iter()
        .zip(modes.iter())
        .filter_map(|(&(dx, dy), mode)| {
            if matches!(mode, MvMode::New) {
                Some((dx, dy))
            } else {
                None
            }
        })
        .collect();
    if filtered.is_empty() {
        return (0, 0);
    }
    compute_bias_inner(&filtered)
}

fn compute_bias_inner(deltas: &[(i8, i8)]) -> (i8, i8) {
    const MAX_BIAS: i32 = 24;
    if deltas.is_empty() {
        return (0, 0);
    }

    let mut best_dx = 0i8;
    let mut best_dy = 0i8;
    let mut best_zero_x = -1i64;
    let mut best_zero_y = -1i64;
    let mut best_cost_x = i64::MAX;
    let mut best_cost_y = i64::MAX;

    for bias in -MAX_BIAS..=MAX_BIAS {
        let bias = bias as i16;
        let mut zero_x = 0i64;
        let mut cost_x = 0i64;
        let mut zero_y = 0i64;
        let mut cost_y = 0i64;
        for &(dx, dy) in deltas {
            let adj_x = (i16::from(dx) - bias).clamp(-128, 127) as i8;
            if adj_x == 0 {
                zero_x += 1;
            }
            cost_x += i64::from(adj_x.abs() as i32);

            let adj_y = (i16::from(dy) - bias).clamp(-128, 127) as i8;
            if adj_y == 0 {
                zero_y += 1;
            }
            cost_y += i64::from(adj_y.abs() as i32);
        }

        if zero_x > best_zero_x || (zero_x == best_zero_x && cost_x < best_cost_x) {
            best_zero_x = zero_x;
            best_cost_x = cost_x;
            best_dx = bias as i8;
        }
        if zero_y > best_zero_y || (zero_y == best_zero_y && cost_y < best_cost_y) {
            best_zero_y = zero_y;
            best_cost_y = cost_y;
            best_dy = bias as i8;
        }
    }

    (best_dx, best_dy)
}
fn pad_rgb24(
    display: &[u8],
    display_width: usize,
    display_height: usize,
    storage_width: usize,
    storage_height: usize,
) -> Result<Vec<u8>> {
    let expected = display_width * display_height * 3;
    if display.len() != expected {
        return Err(EncodeError::FrameError(format!(
            "Display RGB24 size mismatch: expected {expected}, got {}",
            display.len()
        )));
    }
    if storage_width < display_width || storage_height < display_height {
        return Err(EncodeError::FrameError(
            "Storage dims must be >= display dims".to_string(),
        ));
    }
    let mut out = vec![0u8; storage_width * storage_height * 3];

    // Copy display area.
    for y in 0..display_height {
        let src_row = &display[(y * display_width * 3)..((y + 1) * display_width * 3)];
        let dst_off = y * storage_width * 3;
        out[dst_off..dst_off + display_width * 3].copy_from_slice(src_row);
        // Pad to the right by repeating the last pixel.
        if storage_width > display_width {
            let last_px = &src_row[(display_width - 1) * 3..display_width * 3];
            for x in display_width..storage_width {
                let di = dst_off + x * 3;
                out[di..di + 3].copy_from_slice(last_px);
            }
        }
    }

    // Pad bottom rows by repeating the last display row.
    if storage_height > display_height {
        let last_row_start = (display_height - 1) * storage_width * 3;
        for y in display_height..storage_height {
            let dst_off = y * storage_width * 3;
            let row = out[last_row_start..last_row_start + storage_width * 3].to_vec();
            out[dst_off..dst_off + storage_width * 3].copy_from_slice(&row);
        }
    }

    Ok(out)
}

// Old pack_mv_delta removed - using new 3-byte format instead
