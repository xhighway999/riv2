use crate::ffmpeg_utils::{self, VideoOutput};
use crate::file_video_reader::FileVideoReader;
use crate::ri_cli_modules::decode_stats::DecodeStats;
use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use reitero_decode::Decoder;
use std::io::Write;
use std::process::{Child, Command, Stdio};

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum DecodeOutputMode {
    /// Encode to an MP4 file using FFmpeg/libx264
    File,
    /// Decode only (benchmark mode, no output)
    Null,
    /// Pipe raw YUV420P to mpv
    Mpv,
}

trait FrameSink {
    fn process_frame(&mut self, rgb_data: &[u8], frame_idx: u64) -> Result<()>;
    fn finish(&mut self) -> Result<()>;
}

struct FileSink {
    video_output: VideoOutput,
    scaler: ffmpeg::software::scaling::Context,
    width: u32,
    height: u32,
    fps: u32,
}

impl FileSink {
    fn new(output_path: &str, width: u32, height: u32, fps: u32) -> Result<Self> {
        let video_output = VideoOutput::create(
            output_path,
            width,
            height,
            fps,
            ffmpeg::util::format::Pixel::YUV420P,
        )?;
        let scaler = ffmpeg_utils::create_scaler(
            ffmpeg::util::format::Pixel::RGB24,
            width,
            height,
            ffmpeg::util::format::Pixel::YUV420P,
            width,
            height,
        )?;
        Ok(Self {
            video_output,
            scaler,
            width,
            height,
            fps,
        })
    }

    fn receive_and_write_packets(&mut self) -> Result<()> {
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match self.video_output.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(self.video_output.stream_index);
                    packet.rescale_ts(
                        self.video_output.encoder_time_base,
                        self.video_output.output_time_base,
                    );
                    packet.write_interleaved(&mut self.video_output.format_context)?;
                }
                Err(ffmpeg::Error::Eof) => break,
                Err(ffmpeg::Error::Other { errno }) if errno == -11 || errno == 11 => break, // EAGAIN
                Err(ffmpeg::Error::Other { errno })
                    if errno == -541478725 || errno == 541478725 =>
                {
                    break;
                } // EOF
                Err(e) => return Err(anyhow::anyhow!(e)),
            }
        }
        Ok(())
    }
}

impl FrameSink for FileSink {
    fn process_frame(&mut self, rgb_data: &[u8], frame_idx: u64) -> Result<()> {
        // Create RGB24 frame from raw data
        let mut rgb_frame = ffmpeg::util::frame::video::Video::new(
            ffmpeg::util::format::Pixel::RGB24,
            self.width,
            self.height,
        );
        rgb_frame.data_mut(0)[..rgb_data.len()].copy_from_slice(rgb_data);

        // Convert to YUV420P
        let mut yuv_frame = ffmpeg::util::frame::video::Video::new(
            ffmpeg::util::format::Pixel::YUV420P,
            self.width,
            self.height,
        );
        self.scaler.run(&rgb_frame, &mut yuv_frame)?;

        // Set PTS
        let tb_num = self.video_output.encoder_time_base.0 as i128;
        let tb_den = self.video_output.encoder_time_base.1 as i128;
        let pts = (frame_idx as i128) * tb_den / (tb_num * self.fps as i128);
        yuv_frame.set_pts(Some(pts as i64));

        if let Err(e) = self.video_output.encoder.send_frame(&yuv_frame) {
            match e {
                ffmpeg::Error::Other { errno } if errno == -11 || errno == 11 => {
                    self.receive_and_write_packets()?;
                    self.video_output.encoder.send_frame(&yuv_frame)?;
                }
                _ => return Err(anyhow::anyhow!(e)),
            }
        }
        self.receive_and_write_packets()?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.video_output.encoder.send_eof()?;
        self.receive_and_write_packets()?;
        self.video_output.format_context.write_trailer()?;
        Ok(())
    }
}

struct MpvSink {
    child: Option<Child>,
    stdin: Option<std::process::ChildStdin>,
}

impl MpvSink {
    fn new(width: u32, height: u32, fps: u32) -> Result<Self> {
        let mut child = Command::new("mpv")
            .arg("-")
            .arg("--demuxer=rawvideo")
            .arg(format!("--demuxer-rawvideo-w={}", width))
            .arg(format!("--demuxer-rawvideo-h={}", height))
            .arg("--demuxer-rawvideo-mp-format=rgb24")
            .arg(format!("--demuxer-rawvideo-fps={}", fps))
            .stdin(Stdio::piped())
            .spawn()
            .context("Failed to spawn mpv")?;

        let stdin = child.stdin.take().context("Failed to open mpv stdin")?;
        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
        })
    }
}

impl FrameSink for MpvSink {
    fn process_frame(&mut self, rgb_data: &[u8], _frame_idx: u64) -> Result<()> {
        let stdin = self.stdin.as_mut().context("Stdin already closed")?;
        stdin.write_all(rgb_data)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.stdin.take();
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
        Ok(())
    }
}

pub fn decode_video(
    input: &str,
    output: Option<&str>,
    mode: DecodeOutputMode,
    skip_residuals: bool,
) -> Result<()> {
    match mode {
        DecodeOutputMode::File => println!(
            "Decoding {} to {} as MP4",
            input,
            output.unwrap_or("output.mp4")
        ),
        DecodeOutputMode::Null => println!("Decoding {} (null output)", input),
        DecodeOutputMode::Mpv => println!("Decoding {} and piping to mpv", input),
    }

    if skip_residuals {
        println!("Note: Residual decoding is disabled - outputting motion-predicted frames only");
    }

    let reader = FileVideoReader::new(input).context("Failed to open input file for decoding")?;
    let mut decoder = Decoder::new(reader).context("Failed to create decoder")?;
    decoder.set_skip_residuals(skip_residuals);

    let header = decoder.header().clone();
    let width = header.display_width;
    let height = header.display_height;
    let fps = header.fps;

    println!(
        "Video info: {}x{} @ {} fps, {} frames",
        width, height, fps, header.frame_count
    );

    if matches!(mode, DecodeOutputMode::Null) {
        let mut stats = DecodeStats::new(width, height, fps);
        while decoder.has_more_frames() {
            decoder.decode_frame_null().context("Failed to decode frame")?;
            // accumulate timings for this frame and reset inside decoder
            stats.add_timings(reitero_decode::Decoder::drain_timings(&mut decoder));
            stats.update();
        }
        stats.print_summary();
        return Ok(());
    }

    ffmpeg_utils::init()?;

    let mut sink: Box<dyn FrameSink> = match mode {
        DecodeOutputMode::File => {
            let out_path =
                output.ok_or_else(|| anyhow::anyhow!("Output path required for 'file' mode"))?;
            Box::new(FileSink::new(out_path, width, height, fps)?)
        }
        DecodeOutputMode::Mpv => Box::new(MpvSink::new(width, height, fps)?),
        DecodeOutputMode::Null => unreachable!(),
    };

    let mut frame_idx = 0u64;
    while decoder.has_more_frames() {
        match decoder.decode_frame() {
            Ok(frame) => {
                sink.process_frame(&frame.data, frame_idx)?;

                frame_idx += 1;
                if frame_idx % 30 == 0 {
                    print!("\rProcessed {} frames...", frame_idx);
                    std::io::stdout().flush().ok();
                }
            }
            Err(e) => {
                eprintln!("Error decoding frame {frame_idx}: {e}");
                break;
            }
        }
    }

    sink.finish()?;
    println!("\nSuccessfully processed {} frames", frame_idx);

    Ok(())
}
