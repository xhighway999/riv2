use crate::error::{DecodeError, Result};
use crate::reader::VideoReader;
use reitero_residual::{
    InterResidualDecodeParams, MvMode, MvRansDecoder, ResidualDecoder, derive_mv_predictors,
    gather_mv_neighbor_set,
};
use reitero_video_common::FrameType;
use reitero_video_common::PackedFrame;
use reitero_video_common::VideoHeader;
use reitero_video_common::{MotionVector, Yuv420Frame, build_predicted};
use reitero_video_common::{PackedFrameData, RIV_VERSION};
use std::time::Instant;

fn crop_rgb24(
    storage: &[u8],
    storage_width: usize,
    display_width: usize,
    display_height: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; display_width * display_height * 3];
    for y in 0..display_height {
        let src_off = y * storage_width * 3;
        let dst_off = y * display_width * 3;
        out[dst_off..dst_off + display_width * 3]
            .copy_from_slice(&storage[src_off..src_off + display_width * 3]);
    }
    out
}

/// Decoded frame data (output)
#[derive(Debug)]
pub struct DecodedFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
    pub frame_index: u64,
    pub frame_type: FrameType,
}

impl DecodedFrame {
    pub fn new(
        data: Vec<u8>,
        width: u32,
        height: u32,
        timestamp: u64,
        frame_index: u64,
        frame_type: FrameType,
    ) -> Self {
        Self {
            data,
            width,
            height,
            timestamp,
            frame_index,
            frame_type,
        }
    }
#[derive(Default, Debug, Clone, Copy)]
pub struct DecodePhaseTimings {
    pub read_bits_ns: u64,
    pub parse_frame_ns: u64,
    pub mv_decode_ns: u64,
    pub build_pred_ns: u64,
    pub residual_ns: u64,
    pub yuv_to_rgb_ns: u64,
}

}

/// Video decoder for custom ReItero format
pub struct Decoder<R: VideoReader> {
    // cumulative timings (ns) for coarse profiling
    timings: DecodePhaseTimings,
    // reusable scratch buffers to reduce per-frame allocations
    mvs_scratch: Vec<MotionVector>,
    skip_mask_scratch: Vec<bool>,
    _reader: R,
    header: VideoHeader,
    current_frame: u64,
    /// Previous reconstructed frame (YUV420). Required to decode inter residuals.
    prev_recon_yuv: Option<Yuv420Frame>,
    /// If true, skip residual decoding and return motion-predicted frames only
    skip_residuals: bool,
    /// Previous frame's motion vectors for temporal MV prediction. None before the
    /// first inter frame or immediately after an intra.
    prev_mvs: Option<Vec<MotionVector>>,
    /// RANS decoder for motion vectors (lives for entire video, maintains probabilities across frames)
    mv_rans_decoder: MvRansDecoder,
}

impl<R: VideoReader> Decoder<R> {
    /// Create a new decoder with the given reader
    pub fn new(mut reader: R) -> Result<Self> {
        // Read and parse header
        let header = Self::read_header(&mut reader)?;

        Ok(Self {
            timings: DecodePhaseTimings::default(),
            mvs_scratch: Vec::new(),
            skip_mask_scratch: Vec::new(),
            _reader: reader,
            header,
            current_frame: 0,
            prev_recon_yuv: None,
            skip_residuals: false,
            mv_rans_decoder: MvRansDecoder::new(),
            prev_mvs: None,
        })
    }

    /// Read and parse the video header
    fn read_header(reader: &mut R) -> Result<VideoHeader> {
        use reitero_video_common::{RIV_MAGIC, VideoHeader};
        let mut buf = [0u8; VideoHeader::header_size()];
        read_exact(reader, &mut buf)?;

        // Magic
        if &buf[0..4] != RIV_MAGIC {
            return Err(DecodeError::InvalidHeader(
                "Invalid magic bytes".to_string(),
            ));
        }
        // Version
        let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        if version != RIV_VERSION {
            return Err(DecodeError::InvalidHeader(format!(
                "Unsupported version: {}",
                version
            )));
        }
        // Display width/height
        let display_width = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let display_height = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
        // Storage width/height
        let storage_width = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
        let storage_height = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
        // FPS
        let fps = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
        // Frame count
        let frame_count = u64::from_le_bytes([
            buf[28], buf[29], buf[30], buf[31], buf[32], buf[33], buf[34], buf[35],
        ]);

        Ok(VideoHeader {
            display_width,
            display_height,
            storage_width,
            storage_height,
            fps,
            frame_count,
        })
    }

    /// Decode the next frame
    pub fn decode_frame(&mut self) -> Result<DecodedFrame> {
        if self.current_frame >= self.header.frame_count {
            return Err(DecodeError::EndOfStream);
        }

        let (frame_type, timestamp_ms, frame_data) = self.decode_next()?;

        let frame = DecodedFrame::new(
            frame_data,
            self.header.display_width,
            self.header.display_height,
            timestamp_ms,
            self.current_frame,
            frame_type,
        );

        self.current_frame += 1;
        use std::time::Instant;
        let t0 = Instant::now();


        Ok(frame)
    }

    fn decode_next(&mut self) -> Result<(FrameType, u64, Vec<u8>)> {
        // Single format: decoded via reitero_video_common::PackedFrame::from_bytes,
        // but we still need to read enough bytes from the stream first.
        // Read timestamp + type first.
        let mut head9 = [0u8; 9];
        read_exact(&mut self._reader, &mut head9)?;
        let timestamp_ms = u64::from_le_bytes([
            head9[0], head9[1], head9[2], head9[3], head9[4], head9[5], head9[6], head9[7],
        ]);
        let frame_type = FrameType::from_u8(head9[8])
            .ok_or_else(|| DecodeError::InvalidFrame("Bad frame_type".into()))?;

        // Read remaining header pieces + payload, then parse.
        let mut buf = Vec::new();
        buf.extend_from_slice(&head9);

        match frame_type {
            FrameType::Intra => {
                let mut size_buf = [0u8; 4];
                read_exact(&mut self._reader, &mut size_buf)?;
                buf.extend_from_slice(&size_buf);
                let size = u32::from_le_bytes(size_buf) as usize;
                let mut payload = vec![0u8; size];
                read_exact(&mut self._reader, &mut payload)?;
                buf.extend_from_slice(&payload);
            }
            FrameType::Inter => {
                // quality(u8)
                let mut quality_buf = [0u8; 1];
                read_exact(&mut self._reader, &mut quality_buf)?;
                buf.extend_from_slice(&quality_buf);

                // global motion vector (dx, dy, flags)
                let mut global_mv_buf = [0u8; 3];
                read_exact(&mut self._reader, &mut global_mv_buf)?;
                buf.extend_from_slice(&global_mv_buf);

                // mv_size(u32)
                let mut mv_size_buf = [0u8; 4];
                read_exact(&mut self._reader, &mut mv_size_buf)?;
                buf.extend_from_slice(&mv_size_buf);
                let mv_size = u32::from_le_bytes(mv_size_buf) as usize;
                let mut mv = vec![0u8; mv_size];
                read_exact(&mut self._reader, &mut mv)?;
                buf.extend_from_slice(&mv);

                let mut res_size_buf = [0u8; 4];
                read_exact(&mut self._reader, &mut res_size_buf)?;
                buf.extend_from_slice(&res_size_buf);
                let res_size = u32::from_le_bytes(res_size_buf) as usize;
                let mut payload = vec![0u8; res_size];
                read_exact(&mut self._reader, &mut payload)?;
                buf.extend_from_slice(&payload);
            }
        }

        let (packed, _) = PackedFrame::from_bytes(&buf).ok_or_else(|| {
            DecodeError::DecodingFailed("Failed to parse packed frame (v2)".into())
        })?;

        let out_frame_type = packed.frame_type();
        let storage_w = self.header.storage_width as usize;
        let storage_h = self.header.storage_height as usize;
        let storage_yuv = match packed.data {
            PackedFrameData::Intra { jpeg_rgb } => {
                let yuv = ResidualDecoder::decode_intra(
                    &jpeg_rgb,
                    self.header.storage_width,
                    self.header.storage_height,
                )
                .map_err(|e| DecodeError::InvalidFrame(format!("Intra JPEG decode error: {e}")))?;
                self.prev_recon_yuv = Some(yuv.clone());
                // No MVs for intra frames; clear temporal MV history.
                self.prev_mvs = None;
                // Reset MV entropy contexts so future inter frames can decode after a seek.
                self.mv_rans_decoder.reset_contexts();
                yuv
            }
            PackedFrameData::Inter {
                quality,
                global_mv,
                mv_deflate,
                residual_yuv420,
            } => {
                let prev = self.prev_recon_yuv.as_ref().ok_or_else(|| {
                    DecodeError::InvalidFrame("Inter frame before first intra frame".into())
                })?;

                // Decode motion vectors using RANS (decoder lives across frames, contexts persist)
                let blocks_w = storage_w / 16;
                let blocks_h = storage_h / 16;
                let num_blocks = blocks_w * blocks_h;

                // Consume per-frame data (decoder contexts persist)
                let t_mv0 = Instant::now();
                self.mv_rans_decoder.consume_frame(&mv_deflate);
                let mv_blocks = self.mv_rans_decoder.decode_frame(blocks_w, blocks_h);

                // Reconstruct motion vectors from structured blocks
                if self.mvs_scratch.len() < num_blocks { self.mvs_scratch.resize(num_blocks, MotionVector::from_raw(0,0,0)); }
                if self.skip_mask_scratch.len() < num_blocks { self.skip_mask_scratch.resize(num_blocks, false); }
                let mvs = &mut self.mvs_scratch;
                let skip_mask = &mut self.skip_mask_scratch;
                mvs.clear(); mvs.reserve_exact(num_blocks);
                skip_mask.clear(); skip_mask.reserve_exact(num_blocks);
                let bias_dx = global_mv.dx() as i16;
                let bias_dy = global_mv.dy() as i16;

                for (block_idx, block) in mv_blocks.iter().enumerate() {
                    let bx = block_idx % blocks_w;
                    let by = block_idx / blocks_w;
                    let flags = block.flags;

                    let delta_x = if block.mode == MvMode::New {
                        (i16::from(block.delta_x) + bias_dx).clamp(-128, 127) as i8
                    } else {
                        0
                    };
                    let delta_y = if block.mode == MvMode::New {
                        (i16::from(block.delta_y) + bias_dy).clamp(-128, 127) as i8
                    } else {
                        0
                    };

                    let predictors = derive_mv_predictors(
                        &mvs,
                        self.prev_mvs.as_deref(),
                        blocks_w,
                        blocks_h,
                        bx,
                        by,
                    );

                    let neigh = gather_mv_neighbor_set(
                        &mvs,
                        self.prev_mvs.as_deref(),
                        blocks_w,
                        blocks_h,
                        bx,
                        by,
                    );

                    let (base_dx, base_dy, base_sub_x, base_sub_y) = match block.mode {
                        MvMode::Zero => (0, 0, 0, 0),
                        MvMode::Nearest => predictors.nearest,
                        MvMode::Near => predictors.near,
                        MvMode::TopRight => neigh.top_right.unwrap_or(predictors.nearest),
                        MvMode::TopLeft => neigh.top_left.unwrap_or(predictors.nearest),
                        MvMode::Temporal => predictors.temporal,
                        MvMode::New => match block.new_base {
                            0 => predictors.nearest,
                            1 => predictors.near,
                            2 => neigh.top_right.unwrap_or(predictors.nearest),
                            3 => neigh.top_left.unwrap_or(predictors.nearest),
                            4 => predictors.temporal,
                            _ => predictors.nearest,
                        },
                    };

                    let dx = (base_dx as i16 + delta_x as i16).clamp(-128, 127) as i8;
                    let dy = (base_dy as i16 + delta_y as i16).clamp(-128, 127) as i8;

                    let mark_skip = (flags & 0x40) != 0;

                    let mv = if block.mode == MvMode::New {
                        MotionVector::from_raw(dx, dy, flags)
                    } else {
                        MotionVector::new(dx, dy, base_sub_x, base_sub_y, mark_skip)
                    };

                    mvs.push(mv);
                    skip_mask.push(mark_skip);
                }
                self.timings.mv_decode_ns += t_mv0.elapsed().as_nanos() as u64;

                // Residual data is RANS-compressed directly (no DEFLATE decompression needed)
                let residual_data = &residual_yuv420;

                let t_pred0 = Instant::now();
                let predicted = build_predicted(prev, storage_w, storage_h, &mvs[..]);
                self.timings.build_pred_ns += t_pred0.elapsed().as_nanos() as u64;
                let curr = ResidualDecoder::decode_inter(InterResidualDecodeParams {
                    predicted_yuv: &predicted,
                    storage_width: self.header.storage_width,
                    storage_height: self.header.storage_height,
                    skip_mask: &skip_mask[..],
                    residual_data: &residual_data,
                    inter_quality: quality,
                    skip_residuals: self.skip_residuals,
                })
                .map_err(|e| {
                    DecodeError::InvalidFrame(format!("Inter residual decode error: {e}"))
                })?;
                self.prev_recon_yuv = Some(curr.clone());
                // Store current MVs as temporal reference for next inter frame
                self.prev_mvs = Some(mvs.clone());
                curr
            }
        };

        let t_rgb0 = Instant::now();
        let storage_rgb = storage_yuv.to_rgb24().map_err(|e| {
            DecodeError::InvalidFrame(format!("storage yuv→rgb conversion failed: {e}"))
        })?;
        self.timings.yuv_to_rgb_ns += t_rgb0.elapsed().as_nanos() as u64;

        let t_crop0 = Instant::now();
        let cropped = crop_rgb24(
            &storage_rgb,
            storage_w,
            self.header.display_width as usize,
            self.header.display_height as usize,
        );
        self.timings.parse_frame_ns += t_crop0.elapsed().as_nanos() as u64; // attribute crop time to parse
        Ok((out_frame_type, timestamp_ms, cropped))
    }

    /// Decode the next frame but stop at storage YUV to avoid RGB conversion/crop (benchmark fast path)
    pub fn decode_frame_null(&mut self) -> Result<()> {
        // reset per-frame timings
        self.timings = DecodePhaseTimings::default();
        let (_ft, _ts, _yuv) = self.decode_next_yuv()?;
        Ok(())
    }

    /// Internal: decode next frame producing storage-sized YUV420 frame without RGB conversion
    fn decode_next_yuv(&mut self) -> Result<(FrameType, u64, Yuv420Frame)> {
        if self.current_frame >= self.header.frame_count {
            return Err(DecodeError::EndOfStream);
        }

        // Read timestamp + type first.
        let t_read0 = Instant::now();
        let mut head9 = [0u8; 9];
        read_exact(&mut self._reader, &mut head9)?;
        let timestamp_ms = u64::from_le_bytes([
            head9[0], head9[1], head9[2], head9[3], head9[4], head9[5], head9[6], head9[7],
        ]);
        let frame_type = FrameType::from_u8(head9[8])
            .ok_or_else(|| DecodeError::InvalidFrame("Bad frame_type".into()))?;

        // Read remaining header pieces + payload, then parse.
        let mut buf = Vec::new();
        buf.extend_from_slice(&head9);

        match frame_type {
            FrameType::Intra => {
                let mut size_buf = [0u8; 4];
                read_exact(&mut self._reader, &mut size_buf)?;
                buf.extend_from_slice(&size_buf);
                let size = u32::from_le_bytes(size_buf) as usize;
                let mut payload = vec![0u8; size];
                read_exact(&mut self._reader, &mut payload)?;
                buf.extend_from_slice(&payload);
            }
            FrameType::Inter => {
                let mut quality_buf = [0u8; 1];
                read_exact(&mut self._reader, &mut quality_buf)?;
                buf.extend_from_slice(&quality_buf);

                let mut global_mv_buf = [0u8; 3];
                read_exact(&mut self._reader, &mut global_mv_buf)?;
                buf.extend_from_slice(&global_mv_buf);

                let mut mv_size_buf = [0u8; 4];
                read_exact(&mut self._reader, &mut mv_size_buf)?;
                buf.extend_from_slice(&mv_size_buf);
                let mv_size = u32::from_le_bytes(mv_size_buf) as usize;
                let mut mv = vec![0u8; mv_size];
                read_exact(&mut self._reader, &mut mv)?;
                buf.extend_from_slice(&mv);

                let mut res_size_buf = [0u8; 4];
                read_exact(&mut self._reader, &mut res_size_buf)?;
                buf.extend_from_slice(&res_size_buf);
                let res_size = u32::from_le_bytes(res_size_buf) as usize;
                let mut payload = vec![0u8; res_size];
                read_exact(&mut self._reader, &mut payload)?;
                buf.extend_from_slice(&payload);
            }
        }
        self.timings.read_bits_ns += t_read0.elapsed().as_nanos() as u64;

        let t_parse0 = Instant::now();
        let (packed, _) = PackedFrame::from_bytes(&buf)
            .ok_or_else(|| DecodeError::DecodingFailed("Failed to parse packed frame (v2)".into()))?;
        self.timings.parse_frame_ns += t_parse0.elapsed().as_nanos() as u64;

        let storage_w = self.header.storage_width as usize;
        let storage_h = self.header.storage_height as usize;
        let storage_yuv = match packed.data {
            PackedFrameData::Intra { jpeg_rgb } => {
                let yuv = ResidualDecoder::decode_intra(
                    &jpeg_rgb,
                    self.header.storage_width,
                    self.header.storage_height,
                )
                .map_err(|e| DecodeError::InvalidFrame(format!("Intra JPEG decode error: {e}")))?;
                self.prev_recon_yuv = Some(yuv.clone());
                self.prev_mvs = None;
                self.mv_rans_decoder.reset_contexts();
                yuv
            }
            PackedFrameData::Inter {
                quality,
                global_mv,
                mv_deflate,
                residual_yuv420,
            } => {
                let prev = self.prev_recon_yuv.as_ref().ok_or_else(|| {
                    DecodeError::InvalidFrame("Inter frame before first intra frame".into())
                })?;

                let blocks_w = storage_w / 16;
                let blocks_h = storage_h / 16;
                let num_blocks = blocks_w * blocks_h;

                self.mv_rans_decoder.consume_frame(&mv_deflate);
                let mv_blocks = self.mv_rans_decoder.decode_frame(blocks_w, blocks_h);

                let mut mvs = Vec::with_capacity(num_blocks);
                let mut skip_mask = Vec::with_capacity(num_blocks);
                let bias_dx = global_mv.dx() as i16;
                let bias_dy = global_mv.dy() as i16;

                for (block_idx, block) in mv_blocks.iter().enumerate() {
                    let bx = block_idx % blocks_w;
                    let by = block_idx / blocks_w;
                    let flags = block.flags;

                    let delta_x = if block.mode == MvMode::New {
                        (i16::from(block.delta_x) + bias_dx).clamp(-128, 127) as i8
                    } else {
                        0
                    };
                    let delta_y = if block.mode == MvMode::New {
                        (i16::from(block.delta_y) + bias_dy).clamp(-128, 127) as i8
                    } else {
                        0
                    };

                    let predictors = derive_mv_predictors(
                        &mvs,
                        self.prev_mvs.as_deref(),
                        blocks_w,
                        blocks_h,
                        bx,
                        by,
                    );

                    let neigh = gather_mv_neighbor_set(
                        &mvs,
                        self.prev_mvs.as_deref(),
                        blocks_w,
                        blocks_h,
                        bx,
                        by,
                    );

                    let (base_dx, base_dy, base_sub_x, base_sub_y) = match block.mode {
                        MvMode::Zero => (0, 0, 0, 0),
                        MvMode::Nearest => predictors.nearest,
                        MvMode::Near => predictors.near,
                        MvMode::TopRight => neigh.top_right.unwrap_or(predictors.nearest),
                        MvMode::TopLeft => neigh.top_left.unwrap_or(predictors.nearest),
                        MvMode::Temporal => predictors.temporal,
                        MvMode::New => match block.new_base {
                            0 => predictors.nearest,
                            1 => predictors.near,
                            2 => neigh.top_right.unwrap_or(predictors.nearest),
                            3 => neigh.top_left.unwrap_or(predictors.nearest),
                            4 => predictors.temporal,
                            _ => predictors.nearest,
                        },
                    };

                    let dx = (base_dx as i16 + delta_x as i16).clamp(-128, 127) as i8;
                    let dy = (base_dy as i16 + delta_y as i16).clamp(-128, 127) as i8;

                    let mark_skip = (flags & 0x40) != 0;

                    let mv = if block.mode == MvMode::New {
                        MotionVector::from_raw(dx, dy, flags)
                    } else {
                        MotionVector::new(dx, dy, base_sub_x, base_sub_y, mark_skip)
                    };

                    mvs.push(mv);
                    skip_mask.push(mark_skip);
                }

                let predicted = build_predicted(prev, storage_w, storage_h, &mvs);
                let curr = ResidualDecoder::decode_inter(InterResidualDecodeParams {
                    predicted_yuv: &predicted,
                    storage_width: self.header.storage_width,
                    storage_height: self.header.storage_height,
                    skip_mask: &skip_mask,
                    residual_data: &residual_yuv420,
                    inter_quality: quality,
                    skip_residuals: self.skip_residuals,
                })
                .map_err(|e| {
                    DecodeError::InvalidFrame(format!("Inter residual decode error: {e}"))
                })?;
                self.prev_recon_yuv = Some(curr.clone());
                self.prev_mvs = Some(mvs.clone());
                curr
            }
        };

        // increment index and return without RGB conversion
        let frame_type = FrameType::from_u8(head9[8])
            .ok_or_else(|| DecodeError::InvalidFrame("Bad frame_type".into()))?;
        self.current_frame += 1;
        Ok((frame_type, timestamp_ms, storage_yuv))
    }


    /// Get the video header
    pub fn header(&self) -> &VideoHeader {
        &self.header
    }

    /// Get the current frame index
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Check if there are more frames to decode
    pub fn has_more_frames(&self) -> bool {
        self.current_frame < self.header.frame_count
    }

    /// Drain and reset per-frame timings (used by null-mode benchmarking)
    pub fn drain_timings(&mut self) -> DecodePhaseTimings {
        let t = self.timings;
        self.timings = DecodePhaseTimings::default();
        t
    }

    /// Set whether to skip residual decoding (return motion-predicted frames only)
    pub fn set_skip_residuals(&mut self, skip: bool) {
        self.skip_residuals = skip;
    }

    /// Get whether residual decoding is skipped
    pub fn skip_residuals(&self) -> bool {
        self.skip_residuals
    }
}

fn read_exact<R: VideoReader>(reader: &mut R, mut buf: &mut [u8]) -> Result<()> {
    while !buf.is_empty() {
        let n = reader.read(buf)?;
        if n == 0 {
            return Err(DecodeError::EndOfStream);
        }
        buf = &mut buf[n..];
    }
    Ok(())
}
