use std::time::{Duration, Instant};

use reitero_encode::EncodeFrameStats;
use reitero_video_common::FrameType;

pub struct AccumulatedStats {
    pub total_frames: usize,
    pub intra_frames: usize,
    pub inter_frames: usize,
    pub total_bytes: usize,
    pub total_mv_bytes: usize,
    pub total_residual_bytes: usize,
    pub total_blocks: usize,
    pub total_skipped_blocks: usize,
    pub rans_frames_with_stats: usize,
    pub total_rans_compression_ratio: f64,
    pub total_rans_zero_percentage: f64,
    pub total_resi_raw: usize,
    pub total_resi_rans: usize,
    pub resi_frames_with_stats: usize,
}

impl AccumulatedStats {
    pub fn new() -> Self {
        Self {
            total_frames: 0,
            intra_frames: 0,
            inter_frames: 0,
            total_bytes: 0,
            total_mv_bytes: 0,
            total_residual_bytes: 0,
            total_blocks: 0,
            total_skipped_blocks: 0,
            rans_frames_with_stats: 0,
            total_rans_compression_ratio: 0.0,
            total_rans_zero_percentage: 0.0,
            total_resi_raw: 0,
            total_resi_rans: 0,
            resi_frames_with_stats: 0,
        }
    }

    pub fn update(&mut self, stats: &EncodeFrameStats) {
        self.total_frames += 1;
        match stats.frame_type {
            FrameType::Intra => self.intra_frames += 1,
            FrameType::Inter => self.inter_frames += 1,
        }
        self.total_bytes += stats.total_bytes;
        self.total_mv_bytes += stats.mv_bytes;
        self.total_residual_bytes += stats.residual_bytes;
        self.total_blocks += stats.blocks_total;
        self.total_skipped_blocks += stats.blocks_skipped;
        if let Some(ref rle_stats) = stats.rle_stats {
            self.rans_frames_with_stats += 1;
            self.total_rans_compression_ratio += rle_stats.compression_ratio;
            self.total_rans_zero_percentage += rle_stats.zero_percentage;
        }
        if let (Some(raw), Some(rans)) = (stats.resi_raw, stats.resi_rans) {
            self.resi_frames_with_stats += 1;
            self.total_resi_raw += raw;
            self.total_resi_rans += rans;
        }
    }
}

pub fn human_bytes(n: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0usize;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} {}", UNITS[u])
    } else {
        format!("{:.1} {}", v, UNITS[u])
    }
}

fn fmt_eta(d: Duration) -> String {
    let secs = d.as_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

pub fn print_stats(s: &EncodeFrameStats, done: u64, total_frames: Option<u64>, started: Instant) {
    let kind = match s.frame_type {
        FrameType::Intra => "I",
        FrameType::Inter => "P",
    };

    let (skip_pct, skip_cnt) = if s.blocks_total == 0 {
        (0.0, 0usize)
    } else {
        (
            (s.blocks_skipped as f64 * 100.0) / (s.blocks_total as f64),
            s.blocks_skipped,
        )
    };

    let (mv_zero_pct, mv_zero_cnt) = if s.blocks_total == 0 {
        (0.0, 0usize)
    } else {
        (
            (s.mv_zero_delta_blocks as f64 * 100.0) / (s.blocks_total as f64),
            s.mv_zero_delta_blocks,
        )
    };

    let prog = match total_frames {
        Some(t) if t > 0 => format!("{done}/{t}"),
        _ => format!("{done}/?"),
    };
    let elapsed = started.elapsed().as_secs_f64().max(1e-6);
    let fps_enc = (done as f64) / elapsed;
    let eta = match total_frames {
        Some(t) if t > 0 && done > 0 && done <= t => {
            let rem = (t - done) as f64;
            if fps_enc > 1e-6 {
                fmt_eta(Duration::from_secs_f64(rem / fps_enc))
            } else {
                "--:--".to_string()
            }
        }
        _ => "--:--".to_string(),
    };
    let rans_info = if let Some(ref rle_stats) = s.rle_stats {
        format!(
            " rans={:.1}:1({:.1}%z)",
            rle_stats.compression_ratio, rle_stats.zero_percentage
        )
    } else {
        String::new()
    };

    let resi_info = if let (Some(raw), Some(rans)) = (s.resi_raw, s.resi_rans) {
        format!(
            " resi_raw={} resi_rans={}",
            human_bytes(raw),
            human_bytes(rans)
        )
    } else {
        String::new()
    };

    println!(
        "{prog} ETA={eta} FPS={fps_enc:5.1} frame {:>5} [{kind}] size={} mv={} resi={} skip={:>5.1}%({}/{}) mv0={:>5.1}%({}/{}){}{}",
        s.frame_index,
        human_bytes(s.total_bytes),
        human_bytes(s.mv_bytes),
        human_bytes(s.residual_bytes),
        skip_pct,
        skip_cnt,
        s.blocks_total,
        mv_zero_pct,
        mv_zero_cnt,
        s.blocks_total,
        rans_info,
        resi_info
    );
}
