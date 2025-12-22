/// Utilities for reordering motion vector byte streams for compression.
///
/// We store per-block motion vectors as 3-byte tuples:
///   [ddx][ddy][flags]
/// in raster order.
///
/// For some compressors (like DEFLATE), it can be beneficial to rearrange
/// this stream into planar form:
///   [ddx_0..ddx_{N-1}] [ddy_0..ddy_{N-1}] [flags_0..flags_{N-1}]
///
/// The functions below provide reversible transforms between these layouts.

/// Convert interleaved MV bytes [ddx, ddy, flags] * N into planar layout:
/// [ddx_run][ddy_run][flags_run].
///
/// Panics if `mv_interleaved.len()` is not a multiple of 3.
pub fn mv_interleaved_to_planar(mv_interleaved: &[u8]) -> Vec<u8> {
    assert!(
        mv_interleaved.len() % 3 == 0,
        "mv_interleaved length must be multiple of 3"
    );
    let blocks_total = mv_interleaved.len() / 3;

    let mut ddx_run = Vec::with_capacity(blocks_total);
    let mut ddy_run = Vec::with_capacity(blocks_total);
    let mut flags_run = Vec::with_capacity(blocks_total);

    for i in 0..blocks_total {
        let base = i * 3;
        ddx_run.push(mv_interleaved[base]);
        ddy_run.push(mv_interleaved[base + 1]);
        flags_run.push(mv_interleaved[base + 2]);
    }

    let mut out = Vec::with_capacity(mv_interleaved.len());
    out.extend_from_slice(&ddx_run);
    out.extend_from_slice(&ddy_run);
    out.extend_from_slice(&flags_run);
    out
}

/// Convert planar MV bytes [ddx_run][ddy_run][flags_run] back to interleaved
/// [ddx, ddy, flags] * N.
///
/// Panics if `mv_planar.len()` is not a multiple of 3.
pub fn mv_planar_to_interleaved(mv_planar: &[u8]) -> Vec<u8> {
    assert!(
        mv_planar.len() % 3 == 0,
        "mv_planar length must be multiple of 3"
    );
    let blocks_total = mv_planar.len() / 3;
    let ddx_off = 0;
    let ddy_off = blocks_total;
    let flags_off = blocks_total * 2;

    let mut out = Vec::with_capacity(mv_planar.len());
    for i in 0..blocks_total {
        out.push(mv_planar[ddx_off + i]);
        out.push(mv_planar[ddy_off + i]);
        out.push(mv_planar[flags_off + i]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn roundtrip_interleaved_planar_small_patterns() {
        // A few handcrafted patterns
        let patterns: Vec<Vec<u8>> = vec![
            vec![],                          // empty
            vec![0, 0, 0],                   // single zero block
            vec![1, 2, 3],                   // single non-zero block
            vec![0, 0, 0, 0, 0, 0],          // two zero blocks
            vec![1, 0, 0, 0, 2, 0, 0, 0, 3], // diagonal deltas
        ];

        for (idx, pattern) in patterns.iter().enumerate() {
            let planar = mv_interleaved_to_planar(pattern);
            let back = mv_planar_to_interleaved(&planar);
            assert_eq!(&back, pattern, "pattern {} failed roundtrip", idx);
        }
    }

    #[test]
    fn roundtrip_random_interleaved_planar() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

        for blocks in [1usize, 2, 3, 10, 31, 64, 127, 256] {
            let len = blocks * 3;
            let mut data = vec![0u8; len];
            rng.fill(&mut data[..]);

            let planar = mv_interleaved_to_planar(&data);
            let back = mv_planar_to_interleaved(&planar);
            assert_eq!(back, data, "random roundtrip failed for blocks={}", blocks);
        }
    }
}
