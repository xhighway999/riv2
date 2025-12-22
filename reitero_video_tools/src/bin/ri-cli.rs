use anyhow::Result;
use clap::{Parser, Subcommand};
use decode::DecodeOutputMode;
use reitero_video_tools::{decode, encode, riv_extract};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "ri-cli")]
#[command(about = "ReItero video encoding and decoding CLI tool", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Encode a video file
    Encode {
        /// Input file path
        #[arg(short, long)]
        input: String,
        /// Output file path
        #[arg(short, long)]
        output: String,
        /// JPEG intra frame quality (1-100, default: 90)
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100), default_value_t = 90)]
        intra_quality: u8,
        /// JPEG inter (residual) frame quality (1-100, default: 35)
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100), default_value_t = 35)]
        inter_quality: u8,
        /// Motion search range in pixels (±N). Must fit in signed 6-bit int part (0-31). Default: 12
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=31), default_value_t = 12)]
        search_range: u8,
        /// Block skip SAD threshold (average abs diff per byte, 0..=255). Default: 3
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=255), default_value_t = 3)]
        skip_threshold: u8,
        /// Early termination threshold for zero-MV search (SAD <= threshold). Default: 0 (disabled)
        #[arg(long, default_value_t = 0)]
        me_zero_mv_threshold: i64,
        /// Early termination threshold for predictor-based search (SAD <= threshold). Default: 0 (disabled)
        #[arg(long, default_value_t = 0)]
        me_predictor_threshold: i64,
        /// Maximum number of frames to encode (0 = encode all). Default: 0
        #[arg(long, default_value_t = 0)]
        max_frames: u64,
        /// RDO Lambda multiplier. Default: 0.49
        #[arg(long, default_value_t = 0.49)]
        rdo_lambda_mult: f64,
    },
    /// Decode a video file
    Decode {
        /// Input file path
        #[arg(short, long)]
        input: String,
        /// Output file path (required for 'file' mode)
        #[arg(short, long)]
        output: Option<String>,
        /// Output mode: file (default), null (benchmark), or mpv (pipe)
        #[arg(short, long, value_enum, default_value_t = DecodeOutputMode::File)]
        mode: DecodeOutputMode,
        /// Skip residual decoding (output motion-predicted frames only)
        #[arg(long)]
        skip_residuals: bool,
        /// Enable instrumentation output
        #[arg(long)]
        instrument: bool,
    },
    /// Extract the Nth frame from a .riv file into ./reconstructed/frame_N/
    ExtractFrame {
        /// Input .riv file path
        #[arg(short, long)]
        input: String,
        /// 0-based frame index to extract
        #[arg(short, long)]
        index: u64,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Encode {
            input,
            output,
            intra_quality,
            inter_quality,
            search_range,
            skip_threshold,
            me_zero_mv_threshold,
            me_predictor_threshold,
            max_frames,
            rdo_lambda_mult,
        } => {
            encode::encode_video(
                &input,
                &output,
                intra_quality,
                inter_quality,
                search_range,
                skip_threshold,
                me_zero_mv_threshold,
                me_predictor_threshold,
                max_frames,
                rdo_lambda_mult,
            )?;
        }
        Command::Decode {
            input,
            output,
            mode,
            skip_residuals,
        } => {
            decode::decode_video(&input, output.as_deref(), mode, skip_residuals)?;
        }
        Command::ExtractFrame { input, index } => {
            let out_dir = riv_extract::extract_frame_to_pwd(Path::new(&input), index)?;
            println!("Extracted frame {index} into {}", out_dir.display());
        }
    }

    Ok(())
}
