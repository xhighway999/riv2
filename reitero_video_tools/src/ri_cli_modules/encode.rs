use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use reitero_encode::{Encoder, EncoderConfig, Frame};
use std::time::Instant;

use crate::ffmpeg_utils::{self, VideoInput};
use crate::stats::{AccumulatedStats, print_stats};
use crate::summary;
use crate::writer::FileWriter;

pub fn encode_video(
    input_path: &str,
    output_path: &str,
    intra_quality: u8,
    inter_quality: u8,
    search_range: u8,
    skip_threshold: u8,
    me_zero_mv_threshold: i64,
    me_predictor_threshold: i64,
    max_frames: u64,
    rdo_lambda_mult: f64,
) -> Result<()> {
    println!("Initializing FFmpeg...");
    ffmpeg_utils::init()?;

    println!("Opening input file: {}", input_path);
    let mut video_input = VideoInput::open(input_path)?;
    let width = video_input.width;
    let height = video_input.height;
    let fps_value = video_input.fps;

    println!("Video info: {}x{} @ {} fps", width, height, fps_value);

    // Create encoder config and writer
    let config = EncoderConfig::new(width, height, fps_value)
        .with_intra_quality(intra_quality)
        .with_inter_quality(inter_quality)
        .with_search_range(search_range)
        .with_skip_threshold(skip_threshold)
        .with_me_zero_mv_threshold(me_zero_mv_threshold)
        .with_me_predictor_threshold(me_predictor_threshold)
        .with_rdo_lambda_mult(rdo_lambda_mult);
    let writer = FileWriter::new(output_path)?;
    let mut encoder = Encoder::new(config, writer).context("Failed to create encoder")?;

    println!("Starting encoding...");
    let mut frame_count = 0;

    // Accumulate stats for final summary
    let mut acc_stats = AccumulatedStats::new();

    // Best-effort total frame estimate for %/ETA.
    let total_frames = video_input.estimate_total_frames();
    let started = Instant::now();

    // Process frames
    'frame_loop: for (stream, packet) in video_input.format_context.packets() {
        if stream.index() == video_input.stream_index {
            video_input
                .decoder
                .send_packet(&packet)
                .context("Failed to send packet to decoder")?;

            let mut decoded_frame = ffmpeg::util::frame::video::Video::empty();
            while video_input
                .decoder
                .receive_frame(&mut decoded_frame)
                .is_ok()
            {
                // Convert frame to RGB format for encoding
                let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();
                let mut scaler = ffmpeg_utils::create_scaler(
                    decoded_frame.format(),
                    decoded_frame.width(),
                    decoded_frame.height(),
                    ffmpeg::util::format::Pixel::RGB24,
                    decoded_frame.width(),
                    decoded_frame.height(),
                )?;

                scaler.run(&decoded_frame, &mut rgb_frame)?;

                // Get frame data
                let frame_data = rgb_frame.data(0).to_vec();

                // Calculate timestamp (in milliseconds)
                let timestamp = (frame_count as f64 / fps_value as f64 * 1000.0) as u64;

                let frame = Frame::new(frame_data, width, height, timestamp);
                let stats = encoder
                    .encode_frame_with_stats(frame)
                    .context("Failed to encode frame")?;
                frame_count += 1;
                print_stats(&stats, frame_count as u64, total_frames, started);

                // Accumulate stats
                acc_stats.update(&stats);

                if max_frames > 0 && frame_count >= max_frames {
                    println!(
                        "\nStopping early: reached max-frames limit ({})",
                        max_frames
                    );
                    break 'frame_loop;
                }
            }
        }
    }

    // Flush decoder
    video_input
        .decoder
        .send_eof()
        .context("Failed to send EOF to decoder")?;
    let mut decoded_frame = ffmpeg::util::frame::video::Video::empty();
    while video_input
        .decoder
        .receive_frame(&mut decoded_frame)
        .is_ok()
    {
        let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();
        let mut scaler = ffmpeg_utils::create_scaler(
            decoded_frame.format(),
            decoded_frame.width(),
            decoded_frame.height(),
            ffmpeg::util::format::Pixel::RGB24,
            decoded_frame.width(),
            decoded_frame.height(),
        )?;

        scaler.run(&decoded_frame, &mut rgb_frame)?;

        let frame_data = rgb_frame.data(0).to_vec();
        let timestamp = (frame_count as f64 / fps_value as f64 * 1000.0) as u64;

        let frame = Frame::new(frame_data, width, height, timestamp);
        let stats = encoder
            .encode_frame_with_stats(frame)
            .context("Failed to encode frame")?;
        frame_count += 1;
        print_stats(&stats, frame_count as u64, total_frames, started);

        // Accumulate stats
        acc_stats.update(&stats);

        if max_frames > 0 && frame_count >= max_frames {
            println!(
                "\nStopping early: reached max-frames limit ({})",
                max_frames
            );
            break;
        }
    }

    println!("\nFinalizing encoding...");
    encoder.finish().context("Failed to finalize encoding")?;

    println!(
        "Successfully encoded {} frames to {}",
        frame_count, output_path
    );

    summary::print_summary(&acc_stats);

    Ok(())
}
