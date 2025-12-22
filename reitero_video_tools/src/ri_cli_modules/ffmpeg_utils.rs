use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;

pub fn init() -> Result<()> {
    ffmpeg::init().context("Failed to initialize FFmpeg")
}

pub struct VideoInput {
    pub format_context: ffmpeg::format::context::Input,
    pub decoder: ffmpeg::decoder::Video,
    pub stream_index: usize,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl VideoInput {
    pub fn open(path: &str) -> Result<Self> {
        let format_context = ffmpeg::format::input(&path).context("Failed to open input file")?;
        let stream = format_context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .context("No video stream found")?;
        let stream_index = stream.index();

        let decoder_context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .context("Failed to create decoder context")?;
        let decoder = decoder_context
            .decoder()
            .video()
            .context("Failed to create video decoder")?;

        let width = decoder.width();
        let height = decoder.height();
        let fps = stream.avg_frame_rate();
        let fps_value = if fps.1 != 0 {
            (fps.0 / fps.1) as u32
        } else {
            30
        };

        Ok(Self {
            format_context,
            decoder,
            stream_index,
            width,
            height,
            fps: fps_value,
        })
    }

    pub fn estimate_total_frames(&self) -> Option<u64> {
        let stream = self.format_context.stream(self.stream_index)?;
        let vf = stream.frames();
        if vf > 0 {
            Some(vf as u64)
        } else {
            let dur_us = self.format_context.duration();
            if dur_us > 0 && self.fps > 0 {
                let dur_s = (dur_us as f64) / 1_000_000.0;
                let est = (dur_s * self.fps as f64).round() as i64;
                if est > 0 { Some(est as u64) } else { None }
            } else {
                None
            }
        }
    }
}

pub struct VideoOutput {
    pub format_context: ffmpeg::format::context::Output,
    pub encoder: ffmpeg::encoder::Video,
    pub stream_index: usize,
    pub encoder_time_base: ffmpeg::Rational,
    pub output_time_base: ffmpeg::Rational,
}

impl VideoOutput {
    pub fn create(
        path: &str,
        width: u32,
        height: u32,
        fps: u32,
        pixel_format: ffmpeg::util::format::Pixel,
    ) -> Result<Self> {
        let mut format_context =
            ffmpeg::format::output(path).context("Failed to create output file")?;

        let codec = ffmpeg::encoder::find_by_name("libx264")
            .or_else(|| ffmpeg::encoder::find(ffmpeg::codec::Id::H264))
            .context("Failed to find H264 encoder (libx264 or builtin)")?;

        let global_header = format_context
            .format()
            .flags()
            .contains(ffmpeg::format::Flags::GLOBAL_HEADER);

        let mut output_stream = format_context
            .add_stream(Some(codec))
            .context("Failed to add stream")?;

        let mut encoder_ctx = ffmpeg::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;

        encoder_ctx.set_width(width);
        encoder_ctx.set_height(height);
        encoder_ctx.set_time_base((1, fps as i32));
        encoder_ctx.set_frame_rate(Some((fps as i32, 1)));
        encoder_ctx.set_format(pixel_format);

        if global_header {
            encoder_ctx.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
        }

        let mut options = ffmpeg::Dictionary::new();
        options.set("b", "288000000");
        options.set("maxrate", "288000000");
        options.set("profile", "high");

        let encoder = encoder_ctx
            .open_with(options)
            .map_err(|e| anyhow::anyhow!("Failed to open libx264 encoder: {e:?}"))?;

        let encoder_time_base = encoder.time_base();
        let stream_index = output_stream.index();

        output_stream.set_parameters(&encoder);
        drop(output_stream);

        format_context
            .write_header()
            .context("Failed to write MP4 header")?;

        let output_time_base = format_context
            .stream(stream_index)
            .ok_or_else(|| anyhow::anyhow!("Output stream missing after write_header"))?
            .time_base();

        Ok(Self {
            format_context,
            encoder,
            stream_index,
            encoder_time_base,
            output_time_base,
        })
    }
}

pub fn create_scaler(
    src_format: ffmpeg::util::format::Pixel,
    src_width: u32,
    src_height: u32,
    dst_format: ffmpeg::util::format::Pixel,
    dst_width: u32,
    dst_height: u32,
) -> Result<ffmpeg::software::scaling::context::Context> {
    ffmpeg::software::scaling::context::Context::get(
        src_format,
        src_width,
        src_height,
        dst_format,
        dst_width,
        dst_height,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )
    .context("Failed to create scaler")
}
