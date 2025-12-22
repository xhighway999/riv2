use anyhow::Result;
use reitero_decode::{Decoder, VideoReader};
use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::{Duration, Instant};

struct FileReader { f: File }
impl FileReader { fn new(path: &str) -> Result<Self> { Ok(Self { f: File::open(path)? }) } }
impl VideoReader for FileReader {
    fn read(&mut self, buf: &mut [u8]) -> reitero_decode::Result<usize> { Ok(self.f.read(buf)?) }
    fn position(&mut self) -> u64 { self.f.stream_position().unwrap_or(0) }
    fn seek(&mut self, pos: u64) -> reitero_decode::Result<()> { self.f.seek(SeekFrom::Start(pos))?; Ok(()) }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 { eprintln!("Usage: {} <input.riv>", args[0]); std::process::exit(1); }
    let input = &args[1];

    let reader = FileReader::new(input)?;
    let mut dec = Decoder::new(reader)?;
    let hdr = dec.header().clone();
    println!("Video: {}x{} @ {} fps, {} frames", hdr.display_width, hdr.display_height, hdr.fps, hdr.frame_count);

    let start = Instant::now();
    let mut last = start;
    let mut frames = 0u64;
    let mut read_ns = 0u64; let mut parse_ns = 0u64; let mut mv_ns = 0u64; let mut pred_ns = 0u64; let mut resid_ns = 0u64; let mut rgb_ns = 0u64;
    let mut rans_ns = 0u64; let mut deint_ns = 0u64; let mut dct_y_ns = 0u64; let mut dct_uv_ns = 0u64; let mut apply_ns = 0u64;

    while dec.has_more_frames() {
        dec.decode_frame_null()?;
        let t = dec.drain_timings();
        read_ns += t.read_bits_ns; parse_ns += t.parse_frame_ns; mv_ns += t.mv_decode_ns; pred_ns += t.build_pred_ns; resid_ns += t.residual_ns; rgb_ns += t.yuv_to_rgb_ns;
        // drain residual-phase counters as well
        let (r, d, y, uv, a) = reitero_residual::drain_residual_phase_counters();
        rans_ns += r; deint_ns += d; dct_y_ns += y; dct_uv_ns += uv; apply_ns += a;
        frames += 1;
        let now = Instant::now();
        if frames % 25 == 0 { let fps = 25.0 / now.duration_since(last).as_secs_f64(); last = now; println!(".. {} frames, inst {:.2} FPS", frames, fps); }
    }

    let total = start.elapsed();
    let avg_fps = frames as f64 / total.as_secs_f64();
    println!("\n=== Null Decode Summary ===");
    println!("Frames: {} time: {:.2}s avg: {:.2} FPS", frames, total.as_secs_f64(), avg_fps);
    let total_ns = (total.as_secs_f64() * 1e9) as u64;
    let pct = |ns: u64| -> f64 { if total_ns > 0 { (ns as f64) * 100.0 / total_ns as f64 } else { 0.0 } };
    println!("read_bits:   {:>12} ns ({:>5.1}%)", read_ns, pct(read_ns));
    println!("parse_frame: {:>12} ns ({:>5.1}%)", parse_ns, pct(parse_ns));
    println!("mv_decode:   {:>12} ns ({:>5.1}%)", mv_ns, pct(mv_ns));
    println!("build_pred:  {:>12} ns ({:>5.1}%)", pred_ns, pct(pred_ns));
    println!("residual:    {:>12} ns ({:>5.1}%)", resid_ns, pct(resid_ns));
    println!("yuv->rgb:    {:>12} ns ({:>5.1}%)", rgb_ns, pct(rgb_ns));
    println!("  residual sub-phases (ns):");
    println!("    rans_decode: {:>12} ns ({:>5.1}%)", rans_ns, pct(rans_ns));
    println!("    deinterleave: {:>12} ns ({:>5.1}%)", deint_ns, pct(deint_ns));
    println!("    dct_y:        {:>12} ns ({:>5.1}%)", dct_y_ns, pct(dct_y_ns));
    println!("    dct_uv:       {:>12} ns ({:>5.1}%)", dct_uv_ns, pct(dct_uv_ns));
    println!("    apply:        {:>12} ns ({:>5.1}%)", apply_ns, pct(apply_ns));
    Ok(())
}
