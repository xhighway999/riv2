#![cfg(feature = "bench-luma")]

use criterion::{Criterion, criterion_group, criterion_main};
use reitero_encode::bench_api::{self, PreparedPlanes};

const BENCH_DIMS: &[(usize, usize)] = &[(640, 360), (1280, 720), (1920, 1080), (1920, 1280)];
const BENCH_DIMS_MULTIPLES_OF_BLOCK: &[(usize, usize)] = &[(640, 480), (1280, 720), (1920, 1080)];
fn make_rgb_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let x_u8 = (x & 0xFF) as u8;
            let y_u8 = (y & 0xFF) as u8;
            data.push(x_u8);
            data.push(y_u8);
            data.push(x_u8 ^ y_u8);
        }
    }
    data
}

fn make_motion_pair(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    let prev = make_rgb_pattern(width, height);
    let mut curr = prev.clone();
    for (i, pixel) in curr.chunks_mut(3).enumerate() {
        let tweak = ((i as u32 * 13) & 0xFF) as u8;
        pixel[0] = pixel[0].wrapping_add(tweak);
        pixel[1] = pixel[1].wrapping_sub(tweak.rotate_left(1));
        pixel[2] = pixel[2].wrapping_add(tweak.rotate_right(1));
    }
    (prev, curr)
}

fn bench_rgb_to_luma(c: &mut Criterion) {
    for &(width, height) in BENCH_DIMS {
        let label_prefix = format!("{}x{}", width, height);
        let rgb = make_rgb_pattern(width, height);

        c.bench_function(
            format!("scalar_rgb_to_luma_plane_{}", label_prefix).as_str(),
            |b| {
                b.iter(|| bench_api::run_scalar_rgb_to_luma_plane(&rgb, width, height));
            },
        );

        #[cfg(feature = "simd")]
        c.bench_function(
            format!("simd_rgb_to_luma_plane_{}", label_prefix).as_str(),
            |b| {
                b.iter(|| bench_api::run_simd_rgb_to_luma_plane(&rgb, width, height));
            },
        );
    }
}

fn bench_sample_luma_halfpel(c: &mut Criterion) {
    for &(width, height) in BENCH_DIMS {
        let label_prefix = format!("{}x{}", width, height);
        let rgb = make_rgb_pattern(width, height);

        c.bench_function(
            format!("scalar_sample_luma_halfpel_{}", label_prefix).as_str(),
            |b| {
                b.iter(|| bench_api::run_scalar_sample_luma_halfpel(&rgb, width, height));
            },
        );

        #[cfg(feature = "simd")]
        c.bench_function(
            format!("simd_sample_luma_halfpel_{}", label_prefix).as_str(),
            |b| {
                b.iter(|| bench_api::run_simd_sample_luma_halfpel(&rgb, width, height));
            },
        );
    }
}

fn bench_sad_block_halfpel(c: &mut Criterion) {
    for &(width, height) in BENCH_DIMS {
        let label_prefix = format!("{}x{}", width, height);
        let (prev, curr) = make_motion_pair(width, height);
        let planes = PreparedPlanes::from_rgb(&prev, &curr, width, height);

        let scalar_planes = planes.clone();
        c.bench_function(
            format!("scalar_sad_block_halfpel_{}", label_prefix).as_str(),
            move |b| {
                b.iter(|| bench_api::run_scalar_sad_halfpel(&scalar_planes));
            },
        );

        #[cfg(feature = "simd")]
        {
            let simd_planes = planes.clone();
            c.bench_function(
                format!("simd_sad_block_halfpel_{}", label_prefix).as_str(),
                move |b| {
                    b.iter(|| bench_api::run_simd_sad_halfpel(&simd_planes));
                },
            );
        }
    }
}

fn bench_satd_block_int(c: &mut Criterion) {
    for &(width, height) in BENCH_DIMS {
        let label_prefix = format!("{}x{}", width, height);
        let (prev, curr) = make_motion_pair(width, height);
        let planes = PreparedPlanes::from_rgb(&prev, &curr, width, height);

        let scalar_planes = planes.clone();
        c.bench_function(
            format!("scalar_satd_block_int_{}", label_prefix).as_str(),
            move |b| {
                b.iter(|| bench_api::run_scalar_satd_block_int(&scalar_planes));
            },
        );

        #[cfg(feature = "simd")]
        {
            let simd_planes = planes.clone();
            c.bench_function(
                format!("simd_satd_block_int_{}", label_prefix).as_str(),
                move |b| {
                    b.iter(|| bench_api::run_simd_satd_block_int(&simd_planes));
                },
            );
        }
    }
}

fn bench_sad_block_int(c: &mut Criterion) {
    for &(width, height) in BENCH_DIMS {
        let label_prefix = format!("{}x{}", width, height);
        let (prev, curr) = make_motion_pair(width, height);
        let planes = PreparedPlanes::from_rgb(&prev, &curr, width, height);

        let scalar_planes = planes.clone();
        c.bench_function(
            format!("scalar_sad_block_int_{}", label_prefix).as_str(),
            move |b| {
                b.iter(|| bench_api::run_scalar_sad_block_int(&scalar_planes));
            },
        );

        #[cfg(feature = "simd")]
        {
            let simd_planes = planes.clone();
            c.bench_function(
                format!("simd_sad_block_int_{}", label_prefix).as_str(),
                move |b| {
                    b.iter(|| bench_api::run_simd_sad_block_int(&simd_planes));
                },
            );
        }
    }
}

fn bench_diamond_search(c: &mut Criterion) {
    const SEARCH_RANGE: u8 = 8;
    for &(width, height) in BENCH_DIMS {
        let label_prefix = format!("{}x{}", width, height);
        let (prev, curr) = make_motion_pair(width, height);
        let planes = PreparedPlanes::from_rgb(&prev, &curr, width, height);

        let scalar_planes = planes.clone();
        c.bench_function(
            format!("scalar_diamond_search_{}", label_prefix).as_str(),
            move |b| {
                b.iter(|| bench_api::run_scalar_diamond_search(&scalar_planes, SEARCH_RANGE));
            },
        );

        #[cfg(feature = "simd")]
        {
            let simd_planes = planes.clone();
            c.bench_function(
                format!("simd_diamond_search_{}", label_prefix).as_str(),
                move |b| {
                    b.iter(|| bench_api::run_simd_diamond_search(&simd_planes, SEARCH_RANGE));
                },
            );
        }
    }
}

criterion_group!(
    name = luma_plane_benches;
    config = Criterion::default();
    targets = bench_rgb_to_luma, bench_sample_luma_halfpel, bench_sad_block_halfpel, bench_sad_block_int, bench_satd_block_int, bench_diamond_search
);
criterion_main!(luma_plane_benches);
