use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ffmpeg_next as ffmpeg;
use reitero_encode::{Encoder, EncoderConfig, Frame};
use std::fmt::Write as _;
use std::io::{Seek, Write};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "ri-quality")]
#[command(about = "ReItero video quality/perf tester", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Encode->decode roundtrip and compute metrics (PSNR/DSSIM), plus timings
    Run {
        /// Input video file path (e.g., peng.mp4)
        #[arg(short, long)]
        input: String,
        /// Max frames to process (0 = all)
        #[arg(long, default_value_t = 60)]
        max_frames: u64,
        /// JPEG intra frame quality (1-100)
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100), default_value_t = 85)]
        intra_quality: u8,
        /// JPEG inter residual quality (1-100)
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100), default_value_t = 80)]
        inter_quality: u8,
        /// Motion search range (0..=31)
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=31), default_value_t = 12)]
        search_range: u8,
        /// Skip SAD threshold per byte (0..=255)
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=255), default_value_t = 12)]
        skip_threshold: u8,
        /// Early termination threshold for zero-MV search (SAD <= threshold). Default: 0 (disabled)
        #[arg(long, default_value_t = 0)]
        me_zero_mv_threshold: i64,
        /// Early termination threshold for predictor-based search (SAD <= threshold). Default: 0 (disabled)
        #[arg(long, default_value_t = 0)]
        me_predictor_threshold: i64,
        /// Include per-frame metrics in the human-readable report
        #[arg(long, default_value_t = false)]
        per_frame: bool,
        /// Compute luminance-only metrics too (BT.601)
        #[arg(long, default_value_t = true)]
        y_only: bool,
        /// RDO Lambda multiplier. Default: 0.49
        #[arg(long, default_value_t = 0.49)]
        rdo_lambda_mult: f64,
        /// Optional output report path
        #[arg(long)]
        report_path: Option<String>,
    },
}

// Simple file writer for .riv
struct FileWriter {
    writer: std::io::BufWriter<std::fs::File>,
    pos: u64,
}
impl FileWriter {
    fn new(path: &str) -> Result<Self> {
        let f = std::fs::File::create(path)?;
        Ok(Self {
            writer: std::io::BufWriter::new(f),
            pos: 0,
        })
    }
}
impl reitero_encode::VideoWriter for FileWriter {
    fn write(&mut self, data: &[u8]) -> reitero_encode::Result<()> {
        self.writer
            .write_all(data)
            .map_err(reitero_encode::EncodeError::Io)?;
        self.pos += data.len() as u64;
        Ok(())
    }
    fn position(&self) -> u64 {
        self.pos
    }
    fn seek(&mut self, pos: u64) -> reitero_encode::Result<()> {
        use std::io::Seek;
        self.writer
            .seek(std::io::SeekFrom::Start(pos))
            .map_err(reitero_encode::EncodeError::Io)?;
        self.pos = pos;
        Ok(())
    }
    fn flush(&mut self) -> reitero_encode::Result<()> {
        self.writer.flush().map_err(reitero_encode::EncodeError::Io)
    }
}

// File reader for decoder
struct FileVideoReader {
    file: std::fs::File,
}
impl FileVideoReader {
    fn new(path: &str) -> reitero_decode::Result<Self> {
        Ok(Self {
            file: std::fs::File::open(path)?,
        })
    }
}
impl reitero_decode::VideoReader for FileVideoReader {
    fn read(&mut self, buf: &mut [u8]) -> reitero_decode::Result<usize> {
        Ok(std::io::Read::read(&mut self.file, buf)?)
    }
    fn position(&mut self) -> u64 {
        self.file.stream_position().unwrap_or(0)
    }
    fn seek(&mut self, pos: u64) -> reitero_decode::Result<()> {
        use std::io::Seek;
        self.file.seek(std::io::SeekFrom::Start(pos))?;
        Ok(())
    }
}

fn rgb_to_luma_601(r: u8, g: u8, b: u8) -> f64 {
    0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64
}

#[derive(Default, Clone)]
struct FrameMetrics {
    index: u64,
    psnr_y: Option<f64>,
    psnr_rgb: Option<f64>,
    mse_y: Option<f64>,
    mse_rgb: Option<f64>,
    dssim: Option<f64>,
    encode_ms: Option<f64>,
    decode_ms: Option<f64>,
    frame_type: Option<String>,
    total_bytes: Option<usize>,
    mv_bytes: Option<usize>,
    mv_raw_bytes: Option<usize>,
    residual_bytes: Option<usize>,
    residual_raw_bytes: Option<usize>,
    residual_rans_bytes: Option<usize>,
    blocks_total: Option<usize>,
    blocks_skipped: Option<usize>,
    mv_zero_delta_blocks: Option<usize>,
    mv_mode_counts: Option<[usize; 7]>,
    mv_new_zero_pre_bias: Option<usize>,
    mv_new_zero_post_bias: Option<usize>,
    mv_new_zero_axes_x: Option<usize>,
    mv_new_zero_axes_y: Option<usize>,
    mv_new_base_counts: Option<[usize; 5]>,
    mv_new_best_ref_counts: Option<[usize; 5]>,
    mv_new_best_ref_l1_saved_sum: Option<u64>,
    mv_new_blocks: Option<usize>,
    mv_new_delta_count: Option<usize>,
    mv_new_delta_mag_sum: Option<f64>,
    mv_new_delta_mag_sq_sum: Option<f64>,
    mv_class_histogram: Option<[u64; 11]>,
    mv_bias_dx: Option<i8>,
    mv_bias_dy: Option<i8>,
    mv_candidate_unique_total: Option<usize>,
    mv_candidate_unique_samples: Option<usize>,
    mv_candidate_unique_min: Option<usize>,
    mv_candidate_unique_max: Option<usize>,
    mv_match_source_counts: Option<[usize; 6]>,
    mv_match_nonzero_blocks: Option<usize>,
    mv_match_any_spatial: Option<usize>,
    mv_match_any_temporal: Option<usize>,
}

struct PredecodedFrame {
    data: Vec<u8>,
    timestamp_ms: u64,
}

struct PredecodeResult {
    frames: Vec<PredecodedFrame>,
    elapsed_ms: f64,
}

#[derive(Clone)]
struct SummaryTotals {
    frames: u64,
    frames_i: u64,
    frames_p: u64,
    mv_bytes: u64,
    mv_raw_bytes: u64,
    residual_i_bytes: u64,
    residual_p_bytes: u64,
    residual_raw_bytes: u64,
    residual_rans_bytes: u64,
    blocks_total: u64,
    blocks_skipped: u64,
    mv_zero_blocks: u64,
    mv_mode_counts: [u64; 7],
    mv_new_zero_pre_bias: u64,
    mv_new_zero_post_bias: u64,
    mv_new_zero_axes_x: u64,
    mv_new_zero_axes_y: u64,
    mv_new_base_counts: [u64; 5],
    mv_new_best_ref_counts: [u64; 5],
    mv_new_best_ref_l1_saved_sum: u64,
    mv_new_blocks: u64,
    mv_new_delta_count: u64,
    mv_new_delta_mag_sum: f64,
    mv_new_delta_mag_sq_sum: f64,
    mv_class_histogram: [u64; 11],
    mv_bias_dx_sum: i64,
    mv_bias_dy_sum: i64,
    mv_bias_frames: u64,
    mv_bias_dx_min: i32,
    mv_bias_dx_max: i32,
    mv_bias_dy_min: i32,
    mv_bias_dy_max: i32,
    mv_candidate_unique_total: u64,
    mv_candidate_unique_samples: u64,
    mv_candidate_unique_min: u64,
    mv_candidate_unique_max: u64,
    mv_match_source_counts: [u64; 6],
    mv_match_nonzero_blocks: u64,
    mv_match_any_spatial: u64,
    mv_match_any_temporal: u64,
}

impl Default for SummaryTotals {
    fn default() -> Self {
        Self {
            frames: 0,
            frames_i: 0,
            frames_p: 0,
            mv_bytes: 0,
            mv_raw_bytes: 0,
            residual_i_bytes: 0,
            residual_p_bytes: 0,
            residual_raw_bytes: 0,
            residual_rans_bytes: 0,
            blocks_total: 0,
            blocks_skipped: 0,
            mv_zero_blocks: 0,
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
            mv_bias_dx_sum: 0,
            mv_bias_dy_sum: 0,
            mv_bias_frames: 0,
            mv_bias_dx_min: i32::MAX,
            mv_bias_dx_max: i32::MIN,
            mv_bias_dy_min: i32::MAX,
            mv_bias_dy_max: i32::MIN,
            mv_candidate_unique_total: 0,
            mv_candidate_unique_samples: 0,
            mv_candidate_unique_min: u64::MAX,
            mv_candidate_unique_max: 0,
            mv_match_source_counts: [0; 6],
            mv_match_nonzero_blocks: 0,
            mv_match_any_spatial: 0,
            mv_match_any_temporal: 0,
        }
    }
}

impl SummaryTotals {
    fn record(&mut self, fm: &FrameMetrics) {
        self.frames += 1;
        match fm.frame_type.as_deref() {
            Some("I") => self.frames_i += 1,
            Some("P") => self.frames_p += 1,
            _ => {}
        }
        if let Some(bytes) = fm.mv_bytes {
            self.mv_bytes += bytes as u64;
        }
        if let Some(bytes) = fm.mv_raw_bytes {
            self.mv_raw_bytes += bytes as u64;
        }
        if let Some(bytes) = fm.residual_bytes {
            match fm.frame_type.as_deref() {
                Some("I") => self.residual_i_bytes += bytes as u64,
                Some("P") => self.residual_p_bytes += bytes as u64,
                _ => self.residual_p_bytes += bytes as u64,
            }
        }
        if let Some(bytes) = fm.residual_raw_bytes {
            self.residual_raw_bytes += bytes as u64;
        }
        if let Some(bytes) = fm.residual_rans_bytes {
            self.residual_rans_bytes += bytes as u64;
        }
        if let Some(total) = fm.blocks_total {
            self.blocks_total += total as u64;
        }
        if let Some(skipped) = fm.blocks_skipped {
            self.blocks_skipped += skipped as u64;
        }
        if let Some(zero) = fm.mv_zero_delta_blocks {
            self.mv_zero_blocks += zero as u64;
        }

        if let Some(counts) = fm.mv_mode_counts {
            for (idx, count) in counts.iter().enumerate() {
                self.mv_mode_counts[idx] += *count as u64;
            }
        }
        if let Some(v) = fm.mv_new_zero_pre_bias {
            self.mv_new_zero_pre_bias += v as u64;
        }
        if let Some(v) = fm.mv_new_zero_post_bias {
            self.mv_new_zero_post_bias += v as u64;
        }
        if let Some(v) = fm.mv_new_zero_axes_x {
            self.mv_new_zero_axes_x += v as u64;
        }
        if let Some(v) = fm.mv_new_zero_axes_y {
            self.mv_new_zero_axes_y += v as u64;
        }
        if let Some(counts) = fm.mv_new_base_counts {
            for (idx, count) in counts.iter().enumerate() {
                self.mv_new_base_counts[idx] += *count as u64;
            }
        }
        if let Some(counts) = fm.mv_new_best_ref_counts {
            for (idx, count) in counts.iter().enumerate() {
                self.mv_new_best_ref_counts[idx] += *count as u64;
            }
        }
        if let Some(v) = fm.mv_new_best_ref_l1_saved_sum {
            self.mv_new_best_ref_l1_saved_sum += v;
        }
        if let Some(v) = fm.mv_new_blocks {
            self.mv_new_blocks += v as u64;
        }
        if let Some(v) = fm.mv_new_delta_count {
            self.mv_new_delta_count += v as u64;
        }
        if let Some(sum) = fm.mv_new_delta_mag_sum {
            self.mv_new_delta_mag_sum += sum;
        }
        if let Some(sum_sq) = fm.mv_new_delta_mag_sq_sum {
            self.mv_new_delta_mag_sq_sum += sum_sq;
        }
        if let Some(hist) = fm.mv_class_histogram {
            for (idx, count) in hist.iter().enumerate() {
                self.mv_class_histogram[idx] += *count;
            }
        }
        if let (Some(dx), Some(dy), Some(new_blocks)) =
            (fm.mv_bias_dx, fm.mv_bias_dy, fm.mv_new_blocks)
        {
            if new_blocks > 0 {
                self.mv_bias_frames += 1;
                let dx_i32 = dx as i32;
                let dy_i32 = dy as i32;
                self.mv_bias_dx_sum += dx_i32 as i64;
                self.mv_bias_dy_sum += dy_i32 as i64;
                self.mv_bias_dx_min = self.mv_bias_dx_min.min(dx_i32);
                self.mv_bias_dx_max = self.mv_bias_dx_max.max(dx_i32);
                self.mv_bias_dy_min = self.mv_bias_dy_min.min(dy_i32);
                self.mv_bias_dy_max = self.mv_bias_dy_max.max(dy_i32);
            }
        }
        if let (Some(total), Some(samples)) =
            (fm.mv_candidate_unique_total, fm.mv_candidate_unique_samples)
        {
            if samples > 0 {
                self.mv_candidate_unique_total += total as u64;
                self.mv_candidate_unique_samples += samples as u64;
                if let Some(min) = fm.mv_candidate_unique_min {
                    self.mv_candidate_unique_min = self.mv_candidate_unique_min.min(min as u64);
                }
                if let Some(max) = fm.mv_candidate_unique_max {
                    self.mv_candidate_unique_max = self.mv_candidate_unique_max.max(max as u64);
                }
            }
        }

        if let Some(v) = fm.mv_match_nonzero_blocks {
            self.mv_match_nonzero_blocks += v as u64;
        }
        if let Some(v) = fm.mv_match_any_spatial {
            self.mv_match_any_spatial += v as u64;
        }
        if let Some(v) = fm.mv_match_any_temporal {
            self.mv_match_any_temporal += v as u64;
        }
        if let Some(counts) = fm.mv_match_source_counts {
            for (idx, c) in counts.iter().enumerate() {
                self.mv_match_source_counts[idx] += *c as u64;
            }
        }
    }

    fn residual_total_bytes(&self) -> u64 {
        self.residual_i_bytes + self.residual_p_bytes
    }
    fn skip_pct(&self) -> f64 {
        pct(self.blocks_skipped, self.blocks_total)
    }
    fn mv_zero_pct(&self) -> f64 {
        pct(self.mv_zero_blocks, self.blocks_total)
    }
    fn mv_ratio(&self) -> Option<f64> {
        if self.mv_bytes > 0 && self.mv_raw_bytes > 0 {
            Some(self.mv_raw_bytes as f64 / self.mv_bytes as f64)
        } else {
            None
        }
    }
    fn residual_ratio(&self) -> Option<f64> {
        if self.residual_rans_bytes > 0 && self.residual_raw_bytes > 0 {
            Some(self.residual_raw_bytes as f64 / self.residual_rans_bytes as f64)
        } else {
            None
        }
    }
}

fn pct(part: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64 * 100.0) / total as f64
    }
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn human_bytes_opt(value: Option<usize>) -> String {
    value
        .map(|v| human_bytes(v as u64))
        .unwrap_or_else(|| "-".into())
}

fn fmt_opt(value: Option<f64>, precision: usize) -> String {
    value
        .map(|v| format!("{:.*}", precision, v))
        .unwrap_or_else(|| "n/a".into())
}

fn fmt_ms(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.2} ms"))
        .unwrap_or_else(|| "-".into())
}

fn fmt_pct_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.1}%"))
        .unwrap_or_else(|| "-".into())
}

fn format_frame_line(fm: &FrameMetrics) -> String {
    let skip_pct = match (fm.blocks_total, fm.blocks_skipped) {
        (Some(total), Some(skipped)) if total > 0 => Some((skipped as f64 * 100.0) / total as f64),
        _ => None,
    };
    let mv_zero_pct = match (fm.blocks_total, fm.mv_zero_delta_blocks) {
        (Some(total), Some(zero)) if total > 0 => Some((zero as f64 * 100.0) / total as f64),
        _ => None,
    };
    let cand_avg = match (fm.mv_candidate_unique_total, fm.mv_candidate_unique_samples) {
        (Some(total), Some(samples)) if samples > 0 => Some(total as f64 / samples as f64),
        _ => None,
    };
    let cand_summary = cand_avg
        .map(|v| format!("cand={v:.2}"))
        .unwrap_or_else(|| "cand=-".into());
    let new_zero_summary = match (
        fm.mv_new_zero_pre_bias,
        fm.mv_new_zero_post_bias,
        fm.mv_new_blocks,
    ) {
        (Some(pre), Some(post), Some(blocks)) if blocks > 0 => {
            let pre_pct = (pre as f64 * 100.0) / blocks as f64;
            let post_pct = (post as f64 * 100.0) / blocks as f64;
            format!("new0={pre_pct:.1}%→{post_pct:.1}%")
        }
        _ => "new0=-".into(),
    };
    let bias_summary = match (fm.mv_bias_dx, fm.mv_bias_dy) {
        (Some(dx), Some(dy)) => format!("bias=({dx},{dy})"),
        _ => "bias=-".into(),
    };
    let mode_summary = fm
        .mv_mode_counts
        .map(|counts| {
            let total: usize = counts.iter().sum();
            if total == 0 {
                return "modes=-".into();
            }
            let labels = ["Z", "Nr", "Na", "TR", "TL", "Nw", "T"];
            let mut parts: Vec<String> = Vec::with_capacity(labels.len());
            for (label, count) in labels.iter().zip(counts.iter()) {
                let pct = (*count as f64 * 100.0) / total as f64;
                parts.push(format!("{label}{pct:.0}%"));
            }
            format!("modes={}", parts.join("/"))
        })
        .unwrap_or_else(|| "modes=-".into());
    let ty = fm.frame_type.as_deref().unwrap_or("?");
    format!(
        "#{:04} [{ty}] psnrY={} psnrRGB={} dssim={} enc={} dec={} size={} mv={} (raw {}) res={} (raw {}) skip={} mv0={} {} {} {} {}",
        fm.index,
        fmt_opt(fm.psnr_y, 2),
        fmt_opt(fm.psnr_rgb, 2),
        fm.dssim
            .map(|d| format!("{d:.5}"))
            .unwrap_or_else(|| "n/a".into()),
        fmt_ms(fm.encode_ms),
        fmt_ms(fm.decode_ms),
        human_bytes_opt(fm.total_bytes),
        human_bytes_opt(fm.mv_bytes),
        human_bytes_opt(fm.mv_raw_bytes),
        human_bytes_opt(fm.residual_bytes),
        human_bytes_opt(fm.residual_raw_bytes),
        fmt_pct_opt(skip_pct),
        fmt_pct_opt(mv_zero_pct),
        cand_summary,
        new_zero_summary,
        bias_summary,
        mode_summary,
    )
}

fn report_progress(label: &str, current: u64, total: Option<u64>) {
    use std::io::{self, Write};
    if let Some(total) = total.filter(|t| *t > 0) {
        let pct = ((current as f64 / total as f64) * 100.0).clamp(0.0, 100.0);
        eprint!(
            "\r{label}: {:>6.2}% ({}/{})",
            pct,
            current.min(total),
            total
        );
    } else {
        eprint!("\r{label}: frame {}", current);
    }
    let _ = io::stderr().flush();
}

struct ReportInput<'a> {
    input_path: &'a str,
    width: u32,
    height: u32,
    fps: u32,
    predecode_ms: f64,
    frames: u64,
    encoded_bytes: u64,
    bits_per_pixel: f64,
    bytes_per_frame: f64,
    intra_quality: u8,
    inter_quality: u8,
    search_range: u8,
    skip_threshold: u8,
    max_frames: u64,
    y_only: bool,
    rdo_lambda_mult: f64,
    psnr_y_mean: Option<f64>,
    psnr_rgb_mean: Option<f64>,
    dssim_mean: Option<f64>,
    enc_total_ms: f64,
    enc_i_total_ms: f64,
    enc_p_total_ms: f64,
    dec_total_ms: f64,
    dec_i_total_ms: f64,
    dec_p_total_ms: f64,
    totals: SummaryTotals,
    per_frame_enabled: bool,
    per_frame_lines: &'a [String],
}

fn build_report(ctx: &ReportInput<'_>) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "ri-quality report");
    let _ = writeln!(out, "Input: {}", ctx.input_path);
    let _ = writeln!(
        out,
        "Frames processed: {} (limit: {}), resolution: {}x{} @ {} fps",
        ctx.frames,
        if ctx.max_frames == 0 {
            "all".into()
        } else {
            ctx.max_frames.to_string()
        },
        ctx.width,
        ctx.height,
        ctx.fps
    );
    let _ = writeln!(
        out,
        "Settings: intra={} inter={} search={} skip={} y-only={} per-frame={} lambda_mult={:.2}",
        ctx.intra_quality,
        ctx.inter_quality,
        ctx.search_range,
        ctx.skip_threshold,
        ctx.y_only,
        ctx.per_frame_enabled,
        ctx.rdo_lambda_mult
    );
    let _ = writeln!(
        out,
        "Encoded size: {} ({} bytes, {:.2} bytes/frame, {:.3} bpp)",
        human_bytes(ctx.encoded_bytes),
        ctx.encoded_bytes,
        ctx.bytes_per_frame,
        ctx.bits_per_pixel
    );
    if ctx.totals.blocks_total > 0 {
        let skip_pct = ctx.totals.skip_pct();
        let _ = writeln!(
            out,
            "File skip rate: {:.2}% blocks ({} / {}) — these motion-predicted blocks quantized to zero residual, so no residual bytes were stored for them",
            skip_pct, ctx.totals.blocks_skipped, ctx.totals.blocks_total
        );
    }

    let _ = writeln!(out, "\n=== Quality Metrics ===");
    let _ = writeln!(out, "  PSNR-Y mean: {} dB", fmt_opt(ctx.psnr_y_mean, 2));
    let _ = writeln!(out, "  PSNR-RGB mean: {} dB", fmt_opt(ctx.psnr_rgb_mean, 2));
    let _ = writeln!(
        out,
        "  DSSIM mean: {}",
        ctx.dssim_mean
            .map(|d| format!("{d:.5}"))
            .unwrap_or_else(|| "n/a".into())
    );

    let enc_fps = if ctx.enc_total_ms > 0.0 {
        (ctx.frames as f64) / (ctx.enc_total_ms / 1000.0)
    } else {
        0.0
    };
    let dec_fps = if ctx.dec_total_ms > 0.0 {
        (ctx.frames as f64) / (ctx.dec_total_ms / 1000.0)
    } else {
        0.0
    };

    let _ = writeln!(out, "\n=== Performance ===");
    let _ = writeln!(
        out,
        "  Predecode (ffmpeg): total {:.1} ms",
        ctx.predecode_ms
    );
    let _ = writeln!(
        out,
        "  Encode: total {:.1} ms (I {:.1} ms / P {:.1} ms) | {:.2} fps",
        ctx.enc_total_ms, ctx.enc_i_total_ms, ctx.enc_p_total_ms, enc_fps
    );
    let _ = writeln!(
        out,
        "  Decode: total {:.1} ms (I {:.1} ms / P {:.1} ms) | {:.2} fps",
        ctx.dec_total_ms, ctx.dec_i_total_ms, ctx.dec_p_total_ms, dec_fps
    );

    let bytes_times_dssim = ctx.dssim_mean.map(|d| (ctx.encoded_bytes as f64) * d);
    let bytes_per_psnr_y = ctx
        .psnr_y_mean
        .filter(|psnr| *psnr > 0.0)
        .map(|psnr| (ctx.encoded_bytes as f64) / psnr);
    let bytes_per_psnr_rgb = ctx
        .psnr_rgb_mean
        .filter(|psnr| *psnr > 0.0)
        .map(|psnr| (ctx.encoded_bytes as f64) / psnr);
    let encode_quality_throughput = if enc_fps > 0.0 {
        ctx.psnr_y_mean.map(|psnr| psnr * enc_fps)
    } else {
        None
    };
    let decode_quality_throughput = if dec_fps > 0.0 {
        ctx.psnr_y_mean.map(|psnr| psnr * dec_fps)
    } else {
        None
    };

    let _ = writeln!(out, "\n=== Efficiency Metrics ===");
    let _ = writeln!(
        out,
        "  Bytes × DSSIM: {}",
        bytes_times_dssim
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|| "n/a".into())
    );
    let _ = writeln!(
        out,
        "  Bytes per PSNR-Y dB: {}",
        bytes_per_psnr_y
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "n/a".into())
    );
    let _ = writeln!(
        out,
        "  Bytes per PSNR-RGB dB: {}",
        bytes_per_psnr_rgb
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "n/a".into())
    );
    let _ = writeln!(
        out,
        "  Encode quality throughput (dB·fps): {}",
        encode_quality_throughput
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "n/a".into())
    );
    let _ = writeln!(
        out,
        "  Decode quality throughput (dB·fps): {}",
        decode_quality_throughput
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "n/a".into())
    );

    let totals = &ctx.totals;
    let total_residual = totals.residual_total_bytes();
    let mv_share = if ctx.encoded_bytes > 0 {
        (totals.mv_bytes as f64 * 100.0) / ctx.encoded_bytes as f64
    } else {
        0.0
    };
    let resi_share = if ctx.encoded_bytes > 0 {
        (total_residual as f64 * 100.0) / ctx.encoded_bytes as f64
    } else {
        0.0
    };
    let i_share = if ctx.encoded_bytes > 0 {
        (totals.residual_i_bytes as f64 * 100.0) / ctx.encoded_bytes as f64
    } else {
        0.0
    };
    let p_share = if ctx.encoded_bytes > 0 {
        (totals.residual_p_bytes as f64 * 100.0) / ctx.encoded_bytes as f64
    } else {
        0.0
    };

    let _ = writeln!(out, "\n=== Data Breakdown ===");
    let _ = writeln!(
        out,
        "  Motion vectors: raw {} -> compressed {} (ratio {}), {:.1}% of file",
        human_bytes(totals.mv_raw_bytes),
        human_bytes(totals.mv_bytes),
        totals
            .mv_ratio()
            .map(|r| format!("{r:.2}:1"))
            .unwrap_or_else(|| "n/a".into()),
        mv_share
    );
    let _ = writeln!(
        out,
        "  Residuals total: {} ({:.1}% of file)",
        human_bytes(total_residual),
        resi_share
    );
    let _ = writeln!(
        out,
        "    I-frame JPEG: {} ({:.1}% of file)",
        human_bytes(totals.residual_i_bytes),
        i_share
    );
    let _ = writeln!(
        out,
        "    P-frame residual: {} ({:.1}% of file)",
        human_bytes(totals.residual_p_bytes),
        p_share
    );
    let _ = writeln!(
        out,
        "    Residual raw {} -> compressed {} (ratio {})",
        human_bytes(totals.residual_raw_bytes),
        human_bytes(totals.residual_rans_bytes),
        totals
            .residual_ratio()
            .map(|r| format!("{r:.2}:1"))
            .unwrap_or_else(|| "n/a".into())
    );

    if totals.blocks_total > 0 {
        let _ = writeln!(out, "\n=== Block Stats ===");
        let _ = writeln!(
            out,
            "  Skip rate: {:.2}% ({} / {})",
            totals.skip_pct(),
            totals.blocks_skipped,
            totals.blocks_total
        );
        let _ = writeln!(
            out,
            "  MV zero-delta: {:.2}% ({} / {})",
            totals.mv_zero_pct(),
            totals.mv_zero_blocks,
            totals.blocks_total
        );
    }

    let mv_components: u64 = totals.mv_mode_counts.iter().sum();
    if mv_components > 0 {
        let _ = writeln!(out, "\n=== Motion-Vector Diagnostics ===");
        let mode_labels = ["Zero", "Nearest", "Near", "TopR", "TopL", "New", "Temporal"];
        let mut mode_parts: Vec<String> = Vec::new();
        for (label, count) in mode_labels.iter().zip(totals.mv_mode_counts.iter()) {
            let part = format!(
                "{label} {:.1}% ({} blocks)",
                pct(*count, mv_components),
                count
            );
            mode_parts.push(part);
        }
        let _ = writeln!(out, "  Mode split: {}", mode_parts.join(" | "));

        if totals.mv_candidate_unique_samples > 0 {
            let avg =
                totals.mv_candidate_unique_total as f64 / totals.mv_candidate_unique_samples as f64;
            let cand_min = if totals.mv_candidate_unique_min == u64::MAX {
                "-".into()
            } else {
                totals.mv_candidate_unique_min.to_string()
            };
            let cand_max = totals.mv_candidate_unique_max.to_string();
            let _ = writeln!(
                out,
                "  Predictor richness: avg {:.2} unique/block (min {} max {})",
                avg, cand_min, cand_max
            );
        }

        if totals.mv_match_nonzero_blocks > 0 {
            let labels = ["Left", "Top", "TopR", "TopL", "Temporal", "None"];
            let mut parts: Vec<String> = Vec::new();
            for (label, count) in labels.iter().zip(totals.mv_match_source_counts.iter()) {
                if *count == 0 {
                    continue;
                }
                parts.push(format!(
                    "{label} {:.1}% ({count})",
                    pct(*count, totals.mv_match_nonzero_blocks)
                ));
            }
            if parts.is_empty() {
                parts.push("no samples".into());
            }
            let spatial_any = pct(totals.mv_match_any_spatial, totals.mv_match_nonzero_blocks);
            let temporal_any = pct(totals.mv_match_any_temporal, totals.mv_match_nonzero_blocks);
            let _ = writeln!(
                out,
                "  MV exact-match source (non-zero): {}",
                parts.join(" | ")
            );
            let _ = writeln!(
                out,
                "  MV exact-match (non-zero): spatial-any {:.1}% | temporal-any {:.1}%",
                spatial_any, temporal_any
            );
        }

        if totals.mv_new_blocks > 0 {
            let pre = pct(totals.mv_new_zero_pre_bias, totals.mv_new_blocks);
            let post = pct(totals.mv_new_zero_post_bias, totals.mv_new_blocks);
            let axis_den = totals.mv_new_blocks * 2;
            let zero_x = pct(totals.mv_new_zero_axes_x, axis_den);
            let zero_y = pct(totals.mv_new_zero_axes_y, axis_den);
            let _ = writeln!(
                out,
                "  NEW zero share: pre {:.1}% → post {:.1}% | axis-zero X {:.1}% Y {:.1}%",
                pre, post, zero_x, zero_y
            );

            let base_labels = ["Nearest", "Near", "TopR", "TopL", "Temporal"];
            let mut base_parts: Vec<String> = Vec::new();
            for (label, count) in base_labels.iter().zip(totals.mv_new_base_counts.iter()) {
                if *count == 0 {
                    continue;
                }
                base_parts.push(format!(
                    "{label} {:.1}% ({count})",
                    pct(*count, totals.mv_new_blocks)
                ));
            }
            if base_parts.is_empty() {
                base_parts.push("no samples".into());
            }
            let _ = writeln!(out, "  NEW base selector: {}", base_parts.join(" | "));

            let best_labels = ["Nearest", "Near", "TopR", "TopL", "Temporal"];
            let mut parts: Vec<String> = Vec::new();
            for (label, count) in best_labels.iter().zip(totals.mv_new_best_ref_counts.iter()) {
                if *count == 0 {
                    continue;
                }
                parts.push(format!(
                    "{label} {:.1}% ({count})",
                    pct(*count, totals.mv_new_blocks)
                ));
            }
            if parts.is_empty() {
                parts.push("no samples".into());
            }
            let avg_saved =
                (totals.mv_new_best_ref_l1_saved_sum as f64) / (totals.mv_new_blocks.max(1) as f64);
            let _ = writeln!(out, "  NEW best ref (ideal int-L1): {}", parts.join(" | "));
            let _ = writeln!(
                out,
                "  NEW best-ref L1 saved: avg {:.3}px per NEW block",
                avg_saved
            );
        }

        if totals.mv_new_delta_count > 0 {
            let mean = totals.mv_new_delta_mag_sum / totals.mv_new_delta_count as f64;
            let var =
                (totals.mv_new_delta_mag_sq_sum / totals.mv_new_delta_count as f64) - mean * mean;
            let var = var.max(0.0);
            let stddev = var.sqrt();
            let _ = writeln!(
                out,
                "  NEW delta magnitude: mean {:.2}px, stddev {:.2}px ({} blocks)",
                mean, stddev, totals.mv_new_delta_count
            );
        }

        let axis_total = totals.mv_new_delta_count * 2;
        if axis_total > 0 {
            let mut hist_parts: Vec<String> = Vec::new();
            for (class, count) in totals.mv_class_histogram.iter().enumerate() {
                if *count == 0 {
                    continue;
                }
                hist_parts.push(format!("C{class}:{:.1}%", pct(*count, axis_total)));
            }
            if hist_parts.is_empty() {
                hist_parts.push("no NEW components".into());
            }
            let _ = writeln!(
                out,
                "  Magnitude classes (per-axis): {}",
                hist_parts.join(" | ")
            );
        }

        if totals.mv_bias_frames > 0 {
            let avg_dx = totals.mv_bias_dx_sum as f64 / totals.mv_bias_frames as f64;
            let avg_dy = totals.mv_bias_dy_sum as f64 / totals.mv_bias_frames as f64;
            let dx_min = if totals.mv_bias_dx_min == i32::MAX {
                "n/a".into()
            } else {
                totals.mv_bias_dx_min.to_string()
            };
            let dx_max = if totals.mv_bias_dx_max == i32::MIN {
                "n/a".into()
            } else {
                totals.mv_bias_dx_max.to_string()
            };
            let dy_min = if totals.mv_bias_dy_min == i32::MAX {
                "n/a".into()
            } else {
                totals.mv_bias_dy_min.to_string()
            };
            let dy_max = if totals.mv_bias_dy_max == i32::MIN {
                "n/a".into()
            } else {
                totals.mv_bias_dy_max.to_string()
            };
            let _ = writeln!(
                out,
                "  Bias (Δx, Δy): avg ({avg_dx:.1}, {avg_dy:.1}) | dx range {}..{} | dy range {}..{}",
                dx_min, dx_max, dy_min, dy_max
            );
        }
    }

    if ctx.per_frame_enabled && !ctx.per_frame_lines.is_empty() {
        let _ = writeln!(out, "\n=== Per-frame breakdown ===");
        for line in ctx.per_frame_lines {
            let _ = writeln!(out, "{line}");
        }
    }

    out
}

#[derive(Default)]
struct Accum {
    count: u64,
    psnr_y_sum: f64,
    psnr_rgb_sum: f64,
    have_y: u64,
    have_rgb: u64,
    dssim_sum: f64,
    have_dssim: u64,
}
impl Accum {
    fn add(&mut self, f: &FrameMetrics) {
        self.count += 1;
        if let Some(p) = f.psnr_y {
            self.have_y += 1;
            self.psnr_y_sum += p;
        }
        if let Some(p) = f.psnr_rgb {
            self.have_rgb += 1;
            self.psnr_rgb_sum += p;
        }
        if let Some(d) = f.dssim {
            self.have_dssim += 1;
            self.dssim_sum += d;
        }
    }
}

fn compute_psnr(idx: u64, a: &[u8], b: &[u8], y_only: bool) -> FrameMetrics {
    let n = a.len() / 3;
    let mut fm = FrameMetrics {
        index: idx,
        ..Default::default()
    };
    // Y-only
    {
        let mut se = 0.0;
        for i in 0..n {
            let y1 = rgb_to_luma_601(a[3 * i], a[3 * i + 1], a[3 * i + 2]);
            let y2 = rgb_to_luma_601(b[3 * i], b[3 * i + 1], b[3 * i + 2]);
            let d = y1 - y2;
            se += d * d;
        }
        let mse = se / (n as f64);
        fm.mse_y = Some(mse);
        fm.psnr_y = if mse == 0.0 {
            None
        } else {
            Some(10.0 * ((255.0 * 255.0) / mse).log10())
        };
    }
    if !y_only {
        let mut se = 0.0;
        for i in 0..n {
            let dr = a[3 * i] as f64 - b[3 * i] as f64;
            let dg = a[3 * i + 1] as f64 - b[3 * i + 1] as f64;
            let db = a[3 * i + 2] as f64 - b[3 * i + 2] as f64;
            se += dr * dr + dg * dg + db * db;
        }
        let mse = se / ((n as f64) * 3.0);
        fm.mse_rgb = Some(mse);
        fm.psnr_rgb = if mse == 0.0 {
            None
        } else {
            Some(10.0 * ((255.0 * 255.0) / mse).log10())
        };
    }
    fm
}

fn compute_dssim(dssim: &mut dssim_core::Dssim, w: u32, h: u32, a: &[u8], b: &[u8]) -> Option<f64> {
    use rgb::Rgb;
    let (w, h) = (w as usize, h as usize);
    if a.len() != b.len() || a.len() != w * h * 3 {
        return None;
    }
    let mut va: Vec<Rgb<u8>> = Vec::with_capacity(w * h);
    let mut vb: Vec<Rgb<u8>> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        va.push(Rgb {
            r: a[3 * i],
            g: a[3 * i + 1],
            b: a[3 * i + 2],
        });
        vb.push(Rgb {
            r: b[3 * i],
            g: b[3 * i + 1],
            b: b[3 * i + 2],
        });
    }
    let ia = match dssim.create_image_rgb(&va, w, h) {
        Some(v) => v,
        None => return None,
    };
    let ib = match dssim.create_image_rgb(&vb, w, h) {
        Some(v) => v,
        None => return None,
    };
    let (val, _map) = dssim.compare(&ia, &ib);
    Some(val.into())
}

fn predecode_frames(
    ictx: &mut ffmpeg::format::context::Input,
    video_stream_index: usize,
    decoder: &mut ffmpeg::codec::decoder::Video,
    scaler: &mut ffmpeg::software::scaling::context::Context,
    width: u32,
    height: u32,
    fps_val: u32,
    max_frames: u64,
) -> Result<PredecodeResult> {
    fn timestamp_ms(frame_count: u64, fps_val: u32) -> u64 {
        if fps_val == 0 {
            0
        } else {
            ((frame_count as f64 / fps_val as f64) * 1000.0) as u64
        }
    }

    let mut frames = Vec::new();
    let mut frame_count = 0u64;
    let progress_total = if max_frames > 0 {
        Some(max_frames)
    } else {
        None
    };
    let mut progress_printed = false;
    let start = Instant::now();

    let mut push_frame = |fr: &ffmpeg::util::frame::video::Video| -> Result<bool> {
        let mut rgb = ffmpeg::util::frame::video::Video::new(
            ffmpeg::util::format::Pixel::RGB24,
            width,
            height,
        );
        scaler.run(fr, &mut rgb)?;
        let data = rgb.data(0).to_vec();
        frames.push(PredecodedFrame {
            data,
            timestamp_ms: timestamp_ms(frame_count, fps_val),
        });
        frame_count += 1;
        report_progress("Predecoding", frame_count, progress_total);
        progress_printed = true;
        Ok(max_frames > 0 && frame_count >= max_frames)
    };

    'outer: for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            let mut fr = ffmpeg::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut fr).is_ok() {
                if push_frame(&fr)? {
                    break 'outer;
                }
            }
        }
    }

    decoder.send_eof().ok();
    let mut fr = ffmpeg::util::frame::video::Video::empty();
    while decoder.receive_frame(&mut fr).is_ok() {
        if push_frame(&fr)? {
            break;
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    if progress_printed {
        eprintln!();
    }

    Ok(PredecodeResult { frames, elapsed_ms })
}

fn run(
    input: &str,
    max_frames: u64,
    intra: u8,
    inter: u8,
    search: u8,
    skip: u8,
    me_zero_mv_threshold: i64,
    me_predictor_threshold: i64,
    per_frame: bool,
    y_only: bool,
    rdo_lambda_mult: f64,
    report_path: Option<String>,
) -> Result<()> {
    ffmpeg::init()?;
    let mut ictx = ffmpeg::format::input(&input)?;
    let vs = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .context("No video stream")?;
    let v_idx = vs.index();
    let dec_ctx = ffmpeg::codec::context::Context::from_parameters(vs.parameters())?;
    let mut vdec = dec_ctx.decoder().video()?;
    let w = vdec.width();
    let h = vdec.height();
    let fps = vs.avg_frame_rate();
    let fps_val = if fps.1 != 0 {
        (fps.0 / fps.1) as u32
    } else {
        30
    };

    let cfg = EncoderConfig::new(w, h, fps_val)
        .with_intra_quality(intra)
        .with_inter_quality(inter)
        .with_search_range(search)
        .with_skip_threshold(skip)
        .with_me_zero_mv_threshold(me_zero_mv_threshold)
        .with_me_predictor_threshold(me_predictor_threshold)
        .with_rdo_lambda_mult(rdo_lambda_mult);

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        vdec.format(),
        vdec.width(),
        vdec.height(),
        ffmpeg::util::format::Pixel::RGB24,
        vdec.width(),
        vdec.height(),
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;

    let predecode = predecode_frames(
        &mut ictx,
        v_idx,
        &mut vdec,
        &mut scaler,
        w,
        h,
        fps_val,
        max_frames,
    )?;

    if predecode.frames.is_empty() {
        return Err(anyhow::anyhow!("No frames decoded from input"));
    }

    let predecode_ms = predecode.elapsed_ms;
    let predecoded_frames = predecode.frames;
    let predecoded_count = predecoded_frames.len() as u64;

    // temp riv file
    let tmp_path = {
        let mut p = std::env::temp_dir();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        p.push(format!("reitero_quality_{now}.riv"));
        p
    };
    let tmp_path_str = tmp_path.to_string_lossy().to_string();
    let writer = FileWriter::new(&tmp_path_str)?;
    let mut enc = Encoder::new(cfg, writer)?;

    // Encode pass
    let mut frame_count = 0u64;
    let encode_progress_total = if predecoded_count > 0 {
        Some(predecoded_count)
    } else {
        None
    };
    let mut encode_progress_printed = false;
    let mut enc_total_ms = 0.0f64;
    let mut enc_i_total_ms = 0.0f64;
    let mut enc_p_total_ms = 0.0f64;
    let mut enc_frames: Vec<FrameMetrics> = Vec::new();
    for pre_frame in predecoded_frames.into_iter() {
        let data = pre_frame.data;
        let ts = pre_frame.timestamp_ms;
        let t0 = Instant::now();
        let st = enc.encode_frame_with_stats(Frame::new(data, w, h, ts))?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        enc_total_ms += ms;
        match st.frame_type {
            reitero_video_common::FrameType::Intra => enc_i_total_ms += ms,
            reitero_video_common::FrameType::Inter => enc_p_total_ms += ms,
        }
        let mut fm = FrameMetrics {
            index: st.frame_index,
            ..Default::default()
        };
        fm.frame_type = Some(match st.frame_type {
            reitero_video_common::FrameType::Intra => "I".into(),
            _ => "P".into(),
        });
        fm.total_bytes = Some(st.total_bytes);
        fm.mv_bytes = Some(st.mv_bytes);
        fm.mv_raw_bytes = Some(st.mv_raw_bytes);
        fm.residual_bytes = Some(st.residual_jpeg_bytes);
        fm.residual_raw_bytes = st.resi_raw;
        fm.residual_rans_bytes = st.resi_rans;
        fm.blocks_total = Some(st.blocks_total);
        fm.blocks_skipped = Some(st.blocks_skipped);
        fm.mv_zero_delta_blocks = Some(st.mv_zero_delta_blocks);
        fm.mv_mode_counts = Some(st.mv_mode_counts);
        fm.mv_new_zero_pre_bias = Some(st.mv_new_zero_pre_bias);
        fm.mv_new_zero_post_bias = Some(st.mv_new_zero_post_bias);
        fm.mv_new_zero_axes_x = Some(st.mv_new_zero_axes_x);
        fm.mv_new_zero_axes_y = Some(st.mv_new_zero_axes_y);
        fm.mv_new_base_counts = Some(st.mv_new_base_counts);
        fm.mv_new_best_ref_counts = Some(st.mv_new_best_ref_counts);
        fm.mv_new_best_ref_l1_saved_sum = Some(st.mv_new_best_ref_l1_saved_sum);
        fm.mv_new_blocks = Some(st.mv_new_blocks);
        fm.mv_new_delta_count = Some(st.mv_new_delta_count);
        fm.mv_new_delta_mag_sum = Some(st.mv_new_delta_mag_sum);
        fm.mv_new_delta_mag_sq_sum = Some(st.mv_new_delta_mag_sq_sum);
        fm.mv_class_histogram = Some(st.mv_class_histogram);
        fm.mv_bias_dx = Some(st.mv_bias_dx);
        fm.mv_bias_dy = Some(st.mv_bias_dy);
        fm.mv_candidate_unique_total = Some(st.mv_candidate_unique_total);
        fm.mv_candidate_unique_samples = Some(st.mv_candidate_unique_samples);
        fm.mv_candidate_unique_min = Some(st.mv_candidate_unique_min);
        fm.mv_candidate_unique_max = Some(st.mv_candidate_unique_max);
        fm.mv_match_source_counts = Some(st.mv_match_source_counts);
        fm.mv_match_nonzero_blocks = Some(st.mv_match_nonzero_blocks);
        fm.mv_match_any_spatial = Some(st.mv_match_any_spatial);
        fm.mv_match_any_temporal = Some(st.mv_match_any_temporal);
        fm.encode_ms = Some(ms);
        enc_frames.push(fm);
        frame_count += 1;
        report_progress("Encoding", frame_count, encode_progress_total);
        encode_progress_printed = true;
    }

    if encode_progress_printed {
        eprintln!();
    }

    enc.finish()?;
    // After finish() flushes/rewrites header, this on-disk length is the exact riv payload size
    let encoded_bytes = std::fs::metadata(&tmp_path_str)
        .map(|m| m.len())
        .unwrap_or(0);

    // Prepare source reader again for metrics pass
    let mut src_ctx = ffmpeg::format::input(&input)?;
    let vstream = src_ctx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .context("No video stream (metrics)")?;
    let vs_idx = vstream.index();
    let dec_ctx2 = ffmpeg::codec::context::Context::from_parameters(vstream.parameters())?;
    let mut src_dec = dec_ctx2.decoder().video()?;

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        src_dec.format(),
        src_dec.width(),
        src_dec.height(),
        ffmpeg::util::format::Pixel::RGB24,
        w,
        h,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;

    // Decode RIV
    let reader = FileVideoReader::new(&tmp_path_str).context("open temp riv")?;
    let mut riv_dec = reitero_decode::Decoder::new(reader).context("decoder init")?;

    let mut acc = Accum::default();
    let mut totals = SummaryTotals::default();
    let mut per_frame_lines: Vec<String> = Vec::new();
    let mut dssim = dssim_core::Dssim::new();

    let mut next_src_rgb: Option<Vec<u8>> = None;
    let mut fetch_src = |next_src_rgb: &mut Option<Vec<u8>>| -> Result<bool> {
        for (s, p) in src_ctx.packets() {
            if s.index() == vs_idx {
                src_dec.send_packet(&p)?;
                let mut f = ffmpeg::util::frame::video::Video::empty();
                while src_dec.receive_frame(&mut f).is_ok() {
                    let mut rgb = ffmpeg::util::frame::video::Video::new(
                        ffmpeg::util::format::Pixel::RGB24,
                        w,
                        h,
                    );
                    scaler.run(&f, &mut rgb)?;
                    *next_src_rgb = Some(rgb.data(0).to_vec());
                    return Ok(true);
                }
            }
        }
        src_dec.send_eof().ok();
        let mut f = ffmpeg::util::frame::video::Video::empty();
        while src_dec.receive_frame(&mut f).is_ok() {
            let mut rgb =
                ffmpeg::util::frame::video::Video::new(ffmpeg::util::format::Pixel::RGB24, w, h);
            scaler.run(&f, &mut rgb)?;
            *next_src_rgb = Some(rgb.data(0).to_vec());
            return Ok(true);
        }
        Ok(false)
    };

    let mut idx = 0u64;
    let mut dec_total_ms = 0.0f64;
    let mut dec_i_total_ms = 0.0f64;
    let mut dec_p_total_ms = 0.0f64;
    let decode_progress_total = if frame_count > 0 {
        Some(frame_count)
    } else {
        None
    };
    let mut decode_progress_printed = false;
    while riv_dec.has_more_frames() {
        let t0 = Instant::now();
        let decoded = match riv_dec.decode_frame() {
            Ok(df) => df,
            Err(e) => {
                eprintln!("decode error at {idx}: {e}");
                break;
            }
        };
        let dec_ms = t0.elapsed().as_secs_f64() * 1000.0;
        dec_total_ms += dec_ms;
        match decoded.frame_type {
            reitero_video_common::FrameType::Intra => dec_i_total_ms += dec_ms,
            _ => dec_p_total_ms += dec_ms,
        };

        if next_src_rgb.is_none() {
            if !fetch_src(&mut next_src_rgb)? {
                break;
            }
        }
        let src_rgb = next_src_rgb.take().unwrap();
        if src_rgb.len() != decoded.data.len() {
            eprintln!("size mismatch at {idx}");
            break;
        }
        let mut fm = compute_psnr(idx, &src_rgb, &decoded.data, y_only);
        fm.decode_ms = Some(dec_ms);
        // Try DSSIM
        fm.dssim = compute_dssim(&mut dssim, w, h, &src_rgb, &decoded.data);
        // Merge encode stats
        if let Some(encf) = enc_frames.get(idx as usize) {
            fm.frame_type = encf.frame_type.clone();
            fm.total_bytes = encf.total_bytes;
            fm.mv_bytes = encf.mv_bytes;
            fm.mv_raw_bytes = encf.mv_raw_bytes;
            fm.residual_bytes = encf.residual_bytes;
            fm.residual_raw_bytes = encf.residual_raw_bytes;
            fm.residual_rans_bytes = encf.residual_rans_bytes;
            fm.blocks_total = encf.blocks_total;
            fm.blocks_skipped = encf.blocks_skipped;
            fm.mv_zero_delta_blocks = encf.mv_zero_delta_blocks;
            fm.mv_mode_counts = encf.mv_mode_counts;
            fm.mv_new_zero_pre_bias = encf.mv_new_zero_pre_bias;
            fm.mv_new_zero_post_bias = encf.mv_new_zero_post_bias;
            fm.mv_new_zero_axes_x = encf.mv_new_zero_axes_x;
            fm.mv_new_zero_axes_y = encf.mv_new_zero_axes_y;
            fm.mv_new_base_counts = encf.mv_new_base_counts;
            fm.mv_new_best_ref_counts = encf.mv_new_best_ref_counts;
            fm.mv_new_best_ref_l1_saved_sum = encf.mv_new_best_ref_l1_saved_sum;
            fm.mv_new_blocks = encf.mv_new_blocks;
            fm.mv_new_delta_count = encf.mv_new_delta_count;
            fm.mv_new_delta_mag_sum = encf.mv_new_delta_mag_sum;
            fm.mv_new_delta_mag_sq_sum = encf.mv_new_delta_mag_sq_sum;
            fm.mv_class_histogram = encf.mv_class_histogram;
            fm.mv_bias_dx = encf.mv_bias_dx;
            fm.mv_bias_dy = encf.mv_bias_dy;
            fm.mv_candidate_unique_total = encf.mv_candidate_unique_total;
            fm.mv_candidate_unique_samples = encf.mv_candidate_unique_samples;
            fm.mv_candidate_unique_min = encf.mv_candidate_unique_min;
            fm.mv_candidate_unique_max = encf.mv_candidate_unique_max;
            fm.mv_match_source_counts = encf.mv_match_source_counts;
            fm.mv_match_nonzero_blocks = encf.mv_match_nonzero_blocks;
            fm.mv_match_any_spatial = encf.mv_match_any_spatial;
            fm.mv_match_any_temporal = encf.mv_match_any_temporal;
            fm.encode_ms = encf.encode_ms;
        }
        acc.add(&fm);
        totals.record(&fm);
        if per_frame {
            per_frame_lines.push(format_frame_line(&fm));
        }
        idx += 1;
        report_progress("Decoding", idx, decode_progress_total);
        decode_progress_printed = true;
        if max_frames > 0 && idx >= max_frames {
            break;
        }
    }

    if decode_progress_printed {
        eprintln!();
    }

    let frames_done = idx;
    let bpp = if frames_done > 0 {
        (encoded_bytes as f64 * 8.0) / (frames_done as f64 * (w as f64) * (h as f64))
    } else {
        0.0
    };
    let bytes_per_frame = if frames_done > 0 {
        (encoded_bytes as f64) / (frames_done as f64)
    } else {
        0.0
    };
    let psnr_y_mean = if acc.have_y > 0 {
        Some(acc.psnr_y_sum / (acc.have_y as f64))
    } else {
        None
    };
    let psnr_rgb_mean = if acc.have_rgb > 0 {
        Some(acc.psnr_rgb_sum / (acc.have_rgb as f64))
    } else {
        None
    };
    let dssim_mean = if acc.have_dssim > 0 {
        Some(acc.dssim_sum / (acc.have_dssim as f64))
    } else {
        None
    };

    let report_input = ReportInput {
        input_path: input,
        width: w,
        height: h,
        fps: fps_val,
        predecode_ms,
        frames: frames_done,
        encoded_bytes,
        bits_per_pixel: bpp,
        bytes_per_frame,
        intra_quality: intra,
        inter_quality: inter,
        search_range: search,
        skip_threshold: skip,
        max_frames,
        y_only,
        rdo_lambda_mult,
        psnr_y_mean,
        psnr_rgb_mean,
        dssim_mean,
        enc_total_ms,
        enc_i_total_ms,
        enc_p_total_ms,
        dec_total_ms,
        dec_i_total_ms,
        dec_p_total_ms,
        totals,
        per_frame_enabled: per_frame,
        per_frame_lines: &per_frame_lines,
    };
    let report = build_report(&report_input);

    let out_path = if let Some(p) = report_path {
        std::path::PathBuf::from(p)
    } else {
        let out_dir = std::path::Path::new("ri-quality-results");
        std::fs::create_dir_all(out_dir)?;
        let ts_ms: u128 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        out_dir.join(format!("{}.log", ts_ms))
    };

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, &report)?;

    print!("{}", report);
    println!("Saved report to {}", out_path.display());

    // Clean up temporary file
    if let Err(e) = std::fs::remove_file(&tmp_path_str) {
        eprintln!(
            "Warning: failed to remove temp file {}: {}",
            tmp_path_str, e
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Run {
            input,
            max_frames,
            intra_quality,
            inter_quality,
            search_range,
            skip_threshold,
            me_zero_mv_threshold,
            me_predictor_threshold,
            per_frame,
            y_only,
            rdo_lambda_mult,
            report_path,
        } => {
            run(
                &input,
                max_frames,
                intra_quality,
                inter_quality,
                search_range,
                skip_threshold,
                me_zero_mv_threshold,
                me_predictor_threshold,
                per_frame,
                y_only,
                rdo_lambda_mult,
                report_path,
            )?;
        }
    }
    Ok(())
}
