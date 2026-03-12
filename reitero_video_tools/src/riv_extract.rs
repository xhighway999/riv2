use anyhow::{Context, Result};
use reitero_residual::{
    InterResidualDecodeParams, MvMode, MvRansDecoder, ResidualDecoder, derive_mv_predictors,
    gather_mv_neighbor_set,
};
use reitero_video_common::{
    FrameType, PackedFrame, PackedFrameData, RIV_MAGIC, RIV_VERSION, VideoHeader,
};
use reitero_video_common::{MotionVector, Yuv420Frame, build_predicted};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

fn read_exact<R: Read>(r: &mut R, mut buf: &mut [u8]) -> Result<()> {
    while !buf.is_empty() {
        let n = r.read(buf).context("read failed")?;
        if n == 0 {
            anyhow::bail!("unexpected EOF");
        }
        buf = &mut buf[n..];
    }
    Ok(())
}

fn parse_header(mut r: impl Read) -> Result<VideoHeader> {
    let mut buf = vec![0u8; VideoHeader::header_size()];
    read_exact(&mut r, &mut buf)?;

    if &buf[0..4] != RIV_MAGIC {
        anyhow::bail!("bad magic");
    }
    let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if version != RIV_VERSION {
        anyhow::bail!("unsupported riv version: {version}");
    }
    let display_width = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let display_height = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
    let storage_width = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    let storage_height = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    let fps = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
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

fn crop_rgb24(storage: &[u8], storage_w: usize, display_w: usize, display_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; display_w * display_h * 3];
    for y in 0..display_h {
        let src_off = y * storage_w * 3;
        let dst_off = y * display_w * 3;
        out[dst_off..dst_off + display_w * 3]
            .copy_from_slice(&storage[src_off..src_off + display_w * 3]);
    }
    out
}

fn write_ppm(path: impl AsRef<Path>, w: usize, h: usize, rgb: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let mut f = fs::File::create(path).with_context(|| format!("create {path:?}"))?;
    write!(f, "P6\n{} {}\n255\n", w, h)?;
    f.write_all(rgb)?;
    Ok(())
}

fn read_next_frame_record(r: &mut impl Read) -> Result<PackedFrame> {
    let mut head9 = [0u8; 9];
    read_exact(r, &mut head9)?;
    let frame_type =
        FrameType::from_u8(head9[8]).ok_or_else(|| anyhow::anyhow!("bad frame_type"))?;

    let mut buf = Vec::new();
    buf.extend_from_slice(&head9);

    match frame_type {
        FrameType::Intra => {
            // V4 layout: quality(u8) + size(u32) + residual_data
            let mut quality_buf = [0u8; 1];
            read_exact(r, &mut quality_buf)?;
            buf.extend_from_slice(&quality_buf);
            let mut size_buf = [0u8; 4];
            read_exact(r, &mut size_buf)?;
            buf.extend_from_slice(&size_buf);
            let size = u32::from_le_bytes(size_buf) as usize;
            let mut payload = vec![0u8; size];
            read_exact(r, &mut payload)?;
            buf.extend_from_slice(&payload);
        }
        FrameType::Inter => {
            let mut quality_buf = [0u8; 1];
            read_exact(r, &mut quality_buf)?;
            buf.extend_from_slice(&quality_buf);

            let mut global_mv_buf = [0u8; 3];
            read_exact(r, &mut global_mv_buf)?;
            buf.extend_from_slice(&global_mv_buf);

            let mut mv_size_buf = [0u8; 4];
            read_exact(r, &mut mv_size_buf)?;
            buf.extend_from_slice(&mv_size_buf);
            let mv_size = u32::from_le_bytes(mv_size_buf) as usize;
            let mut mv = vec![0u8; mv_size];
            read_exact(r, &mut mv)?;
            buf.extend_from_slice(&mv);

            let mut res_size_buf = [0u8; 4];
            read_exact(r, &mut res_size_buf)?;
            buf.extend_from_slice(&res_size_buf);
            let res_size = u32::from_le_bytes(res_size_buf) as usize;
            let mut payload = vec![0u8; res_size];
            read_exact(r, &mut payload)?;
            buf.extend_from_slice(&payload);
        }
    }

    let (pf, _) =
        PackedFrame::from_bytes(&buf).ok_or_else(|| anyhow::anyhow!("parse PackedFrame failed"))?;
    Ok(pf)
}

pub fn extract_frame_to_pwd(input: &Path, frame_index: u64) -> Result<PathBuf> {
    let mut f = fs::File::open(input).with_context(|| format!("open {input:?}"))?;
    let header = parse_header(&mut f)?;

    let storage_w = header.storage_width as usize;
    let storage_h = header.storage_height as usize;
    let display_w = header.display_width as usize;
    let display_h = header.display_height as usize;
    let out_dir = PathBuf::from("reconstructed").join(format!("frame_{frame_index}"));
    fs::create_dir_all(&out_dir).with_context(|| format!("create dir {out_dir:?}"))?;

    let mut prev_recon: Option<Yuv420Frame> = None;
    // Previous frame's motion vectors for temporal MV prediction
    let mut prev_mvs: Option<Vec<MotionVector>> = None;

    for i in 0..=frame_index {
        let pf =
            read_next_frame_record(&mut f).with_context(|| format!("read frame record {i}"))?;

        match pf.data {
            PackedFrameData::Intra { quality, residual_data } => {
                let recon_yuv = ResidualDecoder::decode_intra(
                    &residual_data,
                    header.storage_width,
                    header.storage_height,
                    quality,
                )
                .map_err(|e| anyhow::anyhow!("intra decode failed: {e}"))?;
                if i == frame_index {
                    let recon = recon_yuv.to_rgb24()
                        .map_err(|e| anyhow::anyhow!("intra yuv->rgb failed: {e}"))?;
                    let cropped = crop_rgb24(&recon, storage_w, display_w, display_h);
                    write_ppm(
                        out_dir.join("reconstructed.ppm"),
                        display_w,
                        display_h,
                        &cropped,
                    )?;
                    // For intra, predicted == reconstructed (no MV).
                    write_ppm(
                        out_dir.join("predicted.ppm"),
                        display_w,
                        display_h,
                        &cropped,
                    )?;
                    fs::write(out_dir.join("intra_residual.bin"), &residual_data)
                        .context("write intra_residual.bin")?;
                }
                prev_recon = Some(recon_yuv);
                prev_mvs = None;
            }
            PackedFrameData::Inter {
                quality,
                global_mv,
                mv_deflate,
                residual_yuv420,
            } => {
                let prev = prev_recon
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("inter before first intra"))?;
                // Decode motion vectors using RANS (same as main decoder)
                let blocks_w = storage_w / 16;
                let blocks_h = storage_h / 16;
                let num_blocks = blocks_w * blocks_h;

                let mut mv_decoder = MvRansDecoder::new();
                mv_decoder.consume_frame(&mv_deflate);
                let mv_blocks = mv_decoder.decode_frame(blocks_w, blocks_h);

                // Reconstruct motion vectors and skip mask from structured blocks
                let mut mvs = Vec::with_capacity(num_blocks);
                let mut skip_mask = Vec::with_capacity(num_blocks);
                let bias_dx = global_mv.dx() as i16;
                let bias_dy = global_mv.dy() as i16;

                for (block_idx, block) in mv_blocks.iter().enumerate() {
                    let bx = block_idx % blocks_w;
                    let by = block_idx / blocks_w;
                    let skip_flag = block.skip;

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

                    let predictors =
                        derive_mv_predictors(&mvs, prev_mvs.as_deref(), blocks_w, blocks_h, bx, by);
                    let neigh = gather_mv_neighbor_set(
                        &mvs,
                        prev_mvs.as_deref(),
                        blocks_w,
                        blocks_h,
                        bx,
                        by,
                    );

                    let base = match block.mode {
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

                    let dx = (base.0 as i16 + delta_x as i16).clamp(-128, 127) as i8;
                    let dy = (base.1 as i16 + delta_y as i16).clamp(-128, 127) as i8;

                    let mark_skip = skip_flag;

                    let spx = match block.subpel_x { reitero_residual::Subpel::PlusHalf => 1, reitero_residual::Subpel::MinusHalf => -1, _ => 0 };
                    let spy = match block.subpel_y { reitero_residual::Subpel::PlusHalf => 1, reitero_residual::Subpel::MinusHalf => -1, _ => 0 };
                    mvs.push(MotionVector::new(dx, dy, spx, spy, mark_skip));
                    skip_mask.push(mark_skip);
                }

                // Residual data is RANS-compressed directly (no DEFLATE decompression needed)
                let residual_data = &residual_yuv420;

                let predicted = build_predicted(prev, storage_w, storage_h, &mvs);

                let recon = ResidualDecoder::decode_inter(InterResidualDecodeParams {
                    predicted_yuv: &predicted,
                    storage_width: header.storage_width,
                    storage_height: header.storage_height,
                    skip_mask: &skip_mask,
                    residual_data,
                    inter_quality: quality,
                    skip_residuals: false, // riv_extract always decodes residuals
                })
                .map_err(|e| anyhow::anyhow!("inter residual decode error: {e}"))?;

                if i == frame_index {
                    let predicted_rgb = predicted
                        .to_rgb24()
                        .map_err(|e| anyhow::anyhow!("predicted yuv→rgb failed: {e}"))?;
                    let pred_crop = crop_rgb24(&predicted_rgb, storage_w, display_w, display_h);
                    let recon_rgb = recon
                        .to_rgb24()
                        .map_err(|e| anyhow::anyhow!("recon yuv→rgb failed: {e}"))?;
                    let recon_crop = crop_rgb24(&recon_rgb, storage_w, display_w, display_h);
                    write_ppm(
                        out_dir.join("predicted.ppm"),
                        display_w,
                        display_h,
                        &pred_crop,
                    )?;
                    write_ppm(
                        out_dir.join("reconstructed.ppm"),
                        display_w,
                        display_h,
                        &recon_crop,
                    )?;
                    // Write residual as raw i16 bytes (little-endian)
                    fs::write(out_dir.join("delta.yuv420"), residual_yuv420)
                        .context("write delta.yuv420 (residual yuv)")?;
                }

                prev_recon = Some(recon);
                prev_mvs = Some(mvs);
            }
        }
    }

    Ok(out_dir)
}
