use crate::stats::{AccumulatedStats, human_bytes};

pub fn print_summary(acc_stats: &AccumulatedStats) {
    // Print final statistics with explanations
    println!("\n=== Encoding Statistics ===");
    println!(
        "Total frames: {} (I-frames: {}, P-frames: {})",
        acc_stats.total_frames, acc_stats.intra_frames, acc_stats.inter_frames
    );
    println!("  - I-frames: Intra-coded frames (full JPEG compression, no motion prediction)");
    println!("  - P-frames: Inter-coded frames (motion vectors + residual compression)");

    println!("\nSize breakdown:");
    println!(
        "  Total size: {} (average: {}/frame)",
        human_bytes(acc_stats.total_bytes),
        human_bytes(if acc_stats.total_frames > 0 {
            acc_stats.total_bytes / acc_stats.total_frames
        } else {
            0
        })
    );
    println!("    - Complete encoded video file size including headers and all frame data");

    println!(
        "  mv (motion vectors): {} (average: {}/frame)",
        human_bytes(acc_stats.total_mv_bytes),
        human_bytes(if acc_stats.total_frames > 0 {
            acc_stats.total_mv_bytes / acc_stats.total_frames
        } else {
            0
        })
    );
    println!("    - Encodes how each 16x16 block moved from the previous frame");
    println!(
        "    - Uses delta coding plus per-frame RANS32 entropy coding with VP8-style contexts"
    );
    println!("    - Contexts persist across frames so MV statistics are learned over time");
    println!("    - Only present in P-frames (I-frames have no motion vectors)");

    println!(
        "  resi (residuals): {} (average: {}/frame)",
        human_bytes(acc_stats.total_residual_bytes),
        human_bytes(if acc_stats.total_frames > 0 {
            acc_stats.total_residual_bytes / acc_stats.total_frames
        } else {
            0
        })
    );
    println!(
        "    - Encodes the difference between predicted frame (from motion vectors) and actual frame"
    );
    println!("    - I-frames: JPEG-compressed RGB data (no residuals)");
    println!("    - P-frames: DCT + quantization + zigzag + RANS compressed residual data");

    if acc_stats.resi_frames_with_stats > 0 {
        let avg_raw = acc_stats.total_resi_raw / acc_stats.resi_frames_with_stats;
        let avg_rans = acc_stats.total_resi_rans / acc_stats.resi_frames_with_stats;
        println!("\nResidual compression pipeline (P-frames only):");
        println!("  This shows how residual data is compressed:");
        println!(
            "  resi_raw: {} (average: {}/frame)",
            human_bytes(acc_stats.total_resi_raw),
            human_bytes(avg_raw)
        );
        println!("    - Size after DCT transform, quantization, and zigzag scanning");
        println!(
            "    - Represents raw quantized DCT coefficients (i16 values) for non-skipped blocks only"
        );
        println!(
            "    - Each block: 256 Y coefficients + 64 U coefficients + 64 V coefficients = 384 coeffs × 2 bytes"
        );

        println!(
            "  resi_rans: {} (average: {}/frame)",
            human_bytes(acc_stats.total_resi_rans),
            human_bytes(avg_rans)
        );
        println!("    - Final size after RANS (Range Asymmetric Numeral Systems) compression");
        println!("    - This is the actual size written to the file");
        println!("    - RANS provides entropy coding optimized for DCT coefficient distributions");

        // Calculate compression ratio
        if avg_raw > 0 {
            let rans_ratio = avg_raw as f64 / avg_rans.max(1) as f64;
            println!("\n  Compression ratio:");
            println!(
                "    - RANS compression: {:.2}:1 (reduced from {} to {})",
                rans_ratio,
                human_bytes(avg_raw),
                human_bytes(avg_rans)
            );
            println!(
                "    - Overall, residual data is {:.1}× smaller than raw zigzag coefficients",
                rans_ratio
            );
        }
    }

    if acc_stats.total_blocks > 0 {
        let skip_pct =
            (acc_stats.total_skipped_blocks as f64 * 100.0) / (acc_stats.total_blocks as f64);
        println!("\nBlock statistics:");
        println!(
            "  skip (skipped blocks): {:.1}% ({}/{})",
            skip_pct, acc_stats.total_skipped_blocks, acc_stats.total_blocks
        );
        println!("    - Blocks that were similar enough to the previous frame to skip encoding");
        println!("    - Includes blocks that quantized to all zeros after DCT");
        println!("    - Skipped blocks use motion vectors only (no residual data needed)");
        println!("    - Higher skip percentage = better compression (fewer residuals to encode)");
    }

    if acc_stats.rans_frames_with_stats > 0 {
        let avg_rans_ratio =
            acc_stats.total_rans_compression_ratio / (acc_stats.rans_frames_with_stats as f64);
        let avg_rans_zero_pct =
            acc_stats.total_rans_zero_percentage / (acc_stats.rans_frames_with_stats as f64);
        println!("\nRANS (Range Asymmetric Numeral Systems) statistics:");
        println!(
            "  rans compression ratio: {:.2}:1 (average across all P-frames)",
            avg_rans_ratio
        );
        println!("    - Shows how much RANS compressed the raw zigzag coefficients");
        println!("    - Higher ratio = better compression (more efficient entropy coding)");
        println!("  zero percentage: {:.1}%", avg_rans_zero_pct);
        println!("    - Percentage of DCT coefficients that were zero after quantization");
        println!(
            "    - Higher percentage = more compressible data (RANS works well with sparse data)"
        );
    }
}
