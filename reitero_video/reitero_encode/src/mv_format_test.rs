// Test to visualize motion vector data format in memory before DEFLATE
// Run with: cargo test --package reitero_encode mv_format_test -- --nocapture

#[cfg(test)]
mod tests {

    #[test]
    fn mv_format_test() {
        println!("\n=== Motion Vector Data Format in Memory (Before DEFLATE) ===\n");

        // Simulate what the encoder creates
        let blocks_w = 10;
        let blocks_h = 5;
        let num_blocks = blocks_w * blocks_h;

        println!(
            "Example: {} blocks ({}x{}) = {} bytes uncompressed\n",
            num_blocks,
            blocks_w,
            blocks_h,
            num_blocks * 3
        );

        // Simulate delta-coded motion vectors
        let mut mv_raw: Vec<u8> = Vec::new();
        let mut prev_dx: i8 = 0;
        let mut prev_dy: i8 = 0;

        // Create some example motion vectors
        let example_mvs = vec![
            (0i8, 0i8, 0x00),  // Block 0: dx=0, dy=0, integer-aligned
            (1i8, 0i8, 0x01),  // Block 1: dx=1, dy=0, +0.5px on X
            (1i8, 1i8, 0x04),  // Block 2: dx=1, dy=1, +0.5px on Y
            (0i8, 1i8, 0x40),  // Block 3: dx=0, dy=1, skip flag set
            (-1i8, 0i8, 0x02), // Block 4: dx=-1, dy=0, -0.5px on X
        ];

        println!("Memory layout (3 bytes per block):");
        println!("[ddx][ddy][flags] [ddx][ddy][flags] [ddx][ddy][flags] ...\n");

        println!("Example blocks (first 5):");
        for (i, (dx, dy, flags)) in example_mvs.iter().enumerate() {
            let ddx = (dx - prev_dx) as i8;
            let ddy = (dy - prev_dy) as i8;

            // Show the actual bytes
            let byte_ddx = ddx as u8;
            let byte_ddy = ddy as u8;

            println!("Block {}:", i);
            println!("  Absolute MV: dx={}, dy={}, flags=0x{:02X}", dx, dy, flags);
            println!(
                "  Delta MV:    ddx={} (0x{:02X}), ddy={} (0x{:02X}), flags=0x{:02X}",
                ddx, byte_ddx, ddy, byte_ddy, flags
            );
            println!(
                "  Memory bytes: [0x{:02X}][0x{:02X}][0x{:02X}]",
                byte_ddx, byte_ddy, flags
            );

            // Decode flags
            let sub_x = flags & 0x03;
            let sub_y = (flags >> 2) & 0x03;
            let skip = (flags & 0x40) != 0;
            println!(
                "  Decoded:      sub_x={} ({:.2}px), sub_y={} ({:.2}px), skip={}",
                sub_x,
                match sub_x {
                    1 => 0.5,
                    2 => -0.5,
                    _ => 0.0,
                },
                sub_y,
                match sub_y {
                    1 => 0.5,
                    2 => -0.5,
                    _ => 0.0,
                },
                skip
            );
            println!();

            mv_raw.push(byte_ddx);
            mv_raw.push(byte_ddy);
            mv_raw.push(*flags);

            prev_dx = *dx;
            prev_dy = *dy;
        }

        println!("Full memory layout (hex dump of first 15 bytes):");
        for (i, byte) in mv_raw.iter().take(15).enumerate() {
            if i % 3 == 0 {
                print!("\n  Block {}: ", i / 3);
            }
            print!("{:02X} ", byte);
        }
        println!("\n");

        println!("Data characteristics:");
        println!("  - Delta-coded: Each block stores delta from previous block");
        println!(
            "  - Signed integers: dx/dy deltas are signed i8, stored as u8 (two's complement)"
        );
        println!("  - Flags byte: bits 0-1=sub-x, bits 2-3=sub-y, bit 6=skip, bit 7=reserved");
        println!(
            "  - Layout: Sequential 3-byte tuples in raster order (left-to-right, top-to-bottom)"
        );
        println!(
            "  - Total size: {} blocks × 3 bytes = {} bytes",
            num_blocks,
            num_blocks * 3
        );
        println!("\nThis is what gets passed to DEFLATE compression.\n");
    }
}
