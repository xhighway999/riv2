//! DCT benchmarks for encode/decode performance.
//! run with cargo bench -p reitero_dct
use criterion::{criterion_group, criterion_main, Criterion, BatchSize, black_box};
use rand::Rng;
use reitero_dct::{encode_plane_8x8_aq, decode_plane_8x8_aq, encode_plane_16x16_aq, decode_plane_16x16_aq};

fn gen_plane(width: usize, height: usize, stride: usize) -> Vec<i16> {
    let mut rng = rand::thread_rng();
    let mut plane = vec![0i16; stride * height];
    for y in 0..height {
        for x in 0..width {
            plane[y * stride + x] = rng.gen_range(-255..=255);
        }
    }
    plane
}

fn bench_dct8_1280x720(c: &mut Criterion) {
    let width = 1280usize;
    let height = 720usize;
    let stride = width;
    let quant_step = 2.0f32;
    let num_blocks = (width / 8) * (height / 8);
    let quant_steps = vec![quant_step; num_blocks];

    c.bench_function("dct8_encode_1280x720", |b| {
        b.iter_batched(
            || gen_plane(width, height, stride),
            |plane| {
                let _encoded = encode_plane_8x8_aq(black_box(&plane), stride, width, height, &quant_steps, None, 0.5);
            },
            BatchSize::LargeInput,
        )
    });

    // Precompute coeffs for decode bench
    let plane = gen_plane(width, height, stride);
    let encoded = encode_plane_8x8_aq(&plane, stride, width, height, &quant_steps, None, 0.5);
    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        match block {
            Some(b) => { coeffs.extend(b); skip_mask.push(false); }
            None => { coeffs.extend(vec![0i16; 64]); skip_mask.push(true); }
        }
    }

    c.bench_function("dct8_decode_1280x720", |b| {
        b.iter_batched(
            || vec![0i16; stride * height],
            |mut output| {
                decode_plane_8x8_aq(
                    black_box(&coeffs),
                    &mut output,
                    stride,
                    width,
                    height,
                    &quant_steps,
                    &skip_mask,
                );
            },
            BatchSize::LargeInput,
        )
    });
}

fn bench_dct16_1280x720(c: &mut Criterion) {
    let width = 1280usize;
    let height = 720usize;
    let stride = width;
    let quant_step = 2.0f32;
    let num_blocks = (width / 16) * (height / 16);
    let quant_steps = vec![quant_step; num_blocks];

    c.bench_function("dct16_encode_1280x720", |b| {
        b.iter_batched(
            || gen_plane(width, height, stride),
            |plane| {
                let _encoded = encode_plane_16x16_aq(black_box(&plane), stride, width, height, &quant_steps, None, 0.5);
            },
            BatchSize::LargeInput,
        )
    });

    // Precompute coeffs for decode bench
    let plane = gen_plane(width, height, stride);
    let encoded = encode_plane_16x16_aq(&plane, stride, width, height, &quant_steps, None, 0.5);
    let mut coeffs = Vec::new();
    let mut skip_mask = Vec::new();
    for block in encoded {
        match block {
            Some(b) => { coeffs.extend(b); skip_mask.push(false); }
            None => { coeffs.extend(vec![0i16; 256]); skip_mask.push(true); }
        }
    }

    c.bench_function("dct16_decode_1280x720", |b| {
        b.iter_batched(
            || vec![0i16; stride * height],
            |mut output| {
                decode_plane_16x16_aq(black_box(&coeffs), &mut output, stride, width, height, &quant_steps, &skip_mask);
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_dct8_1280x720, bench_dct16_1280x720);
criterion_main!(benches);
