use std::io::Write;
use std::time::{Duration, Instant};

pub struct DecodeStats {
    start_time: Instant,
    last_frame_time: Instant,
    frame_count: u64,
    total_duration: Duration,
    worst_frame_time: Duration,
    worst_frame_idx: u64,
    target_fps: f64,
    width: u32,
    height: u32,
    // cumulative timings for attribution (ns)
    total_read_ns: u64,
    total_parse_ns: u64,
    total_mv_ns: u64,
    total_pred_ns: u64,
    total_resid_ns: u64,
    total_rgb_ns: u64,
}

impl DecodeStats {
    pub fn new(width: u32, height: u32, target_fps: u32) -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_frame_time: now,
            frame_count: 0,
            total_duration: Duration::from_secs(0),
            worst_frame_time: Duration::from_secs(0),
            worst_frame_idx: 0,
            target_fps: target_fps as f64,
            width,
            height,
            total_read_ns: 0,
            total_parse_ns: 0,
            total_mv_ns: 0,
            total_pred_ns: 0,
            total_resid_ns: 0,
            total_rgb_ns: 0,
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let frame_duration = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;
        self.frame_count += 1;
        self.total_duration = now.duration_since(self.start_time);

        if frame_duration > self.worst_frame_time {
            self.worst_frame_time = frame_duration;
            self.worst_frame_idx = self.frame_count;
        }

        let fps = 1.0 / frame_duration.as_secs_f64();
        let avg_fps = self.frame_count as f64 / self.total_duration.as_secs_f64();
        let realtime_factor = avg_fps / self.target_fps;

        print!(
            "\rFrame: {:5} | FPS: {:7.2} | Avg: {:7.2} | {:.2}x realtime",
            self.frame_count, fps, avg_fps, realtime_factor
        );
        std::io::stdout().flush().ok();
    }

    pub fn add_timings(&mut self, t: reitero_decode::DecodePhaseTimings) {
        self.total_read_ns += t.read_bits_ns;
        self.total_parse_ns += t.parse_frame_ns;
        self.total_mv_ns += t.mv_decode_ns;
        self.total_pred_ns += t.build_pred_ns;
        self.total_resid_ns += t.residual_ns;
        self.total_rgb_ns += t.yuv_to_rgb_ns;
    }

    pub fn print_summary(&self) {
        let total_secs = self.total_duration.as_secs_f64();
        let avg_fps = if total_secs > 0.0 { self.frame_count as f64 / total_secs } else { 0.0 };
        let realtime_factor = if self.target_fps > 0.0 { avg_fps / self.target_fps } else { 0.0 };

        println!("\n\n=== Decoding Summary ===");
        println!("Resolution:      {}x{}", self.width, self.height);
        println!("Total Frames:    {}", self.frame_count);
        println!("Total Time:      {:.2}s", total_secs);
        println!(
            "Average Speed:   {:.2} FPS ({:.2}x realtime)",
            avg_fps, realtime_factor
        );
        println!(
            "Worst Frame:     #{} ({:.2}ms, {:.2} FPS)",
            self.worst_frame_idx,
            self.worst_frame_time.as_secs_f64() * 1000.0,
            1.0 / self.worst_frame_time.as_secs_f64()
        );
        println!("\nPhase attribution (cumulative over all frames):");
        let total_ns = (total_secs * 1e9) as u64;
        let pct = |ns: u64| -> f64 { if total_ns > 0 { (ns as f64) * 100.0 / (total_ns as f64) } else { 0.0 } };
        println!("  read_bits:   {:>10} ns ({:>5.1}%)", self.total_read_ns, pct(self.total_read_ns));
        println!("  parse_frame: {:>10} ns ({:>5.1}%)", self.total_parse_ns, pct(self.total_parse_ns));
        println!("  mv_decode:   {:>10} ns ({:>5.1}%)", self.total_mv_ns, pct(self.total_mv_ns));
        println!("  build_pred:  {:>10} ns ({:>5.1}%)", self.total_pred_ns, pct(self.total_pred_ns));
        println!("  residual:    {:>10} ns ({:>5.1}%)", self.total_resid_ns, pct(self.total_resid_ns));
        println!("  yuv->rgb:    {:>10} ns ({:>5.1}%)", self.total_rgb_ns, pct(self.total_rgb_ns));
        println!("========================\n");
    }
}
