use criterion::{criterion_group, criterion_main, Criterion};
use reitero_video_common::{build_predicted, reference_build_predicted, MotionVector, Yuv420Frame};

const BLOCK_SIZE: usize = 16;
const BENCH_DIMS: &[(usize, usize)] = &[(640, 360), (1280, 720), (1920, 1080)];

// Simple deterministic PRNG (xorshift64) for reproducible pseudo-random data
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid zero seed which would produce a zero stream
        Self { state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        // Marsaglia xorshift64
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
    fn next_bool(&mut self) -> bool {
        (self.next_u64() & 1) != 0
    }
    fn choose_sign(&mut self) -> i8 {
        if self.next_bool() { 1 } else { -1 }
    }
}

fn make_yuv420_pattern(width: usize, height: usize) -> Yuv420Frame {
    // Seed is deterministic across runs and unique-ish per size
    let mut rng = XorShift64::new(0xC0FF_EE00_D15C_A11D ^ ((width as u64) << 17) ^ (height as u64));

    let mut y = vec![0u8; width * height];
    for px in y.iter_mut() {
        *px = rng.next_u8();
    }

    let cw = width / 2;
    let ch = height / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for (pu, pv) in u.iter_mut().zip(v.iter_mut()) {
        *pu = rng.next_u8();
        *pv = rng.next_u8();
    }
    Yuv420Frame::from_planes(width, height, y, u, v).expect("valid yuv420")
}

fn make_mvs_half_subpel(width: usize, height: usize, base_dx: i8, base_dy: i8) -> Vec<MotionVector> {
    let bw = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let bh = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut rng = XorShift64::new(0xA5A5_5A5A_1234_5678 ^ ((bw as u64) << 9) ^ (bh as u64));
    let mut out = Vec::with_capacity(bw * bh);
    for _ in 0..(bw * bh) {
        // 50% chance of subpixel; when enabled, choose +/- 0.5 independently per axis
        let (sx, sy) = if rng.next_bool() {
            (rng.choose_sign(), rng.choose_sign())
        } else {
            (0, 0)
        };
        out.push(MotionVector::new(base_dx, base_dy, sx, sy, false));
    }
    out
}

fn mse_frames(a: &Yuv420Frame, b: &Yuv420Frame) -> f64 {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());
    let mut acc: f64 = 0.0;
    let mut n: usize = 0;

    for (pa, pb) in a.y_plane().iter().zip(b.y_plane().iter()) {
        let d = *pa as f64 - *pb as f64;
        acc += d * d;
        n += 1;
    }
    for (pa, pb) in a.u_plane().iter().zip(b.u_plane().iter()) {
        let d = *pa as f64 - *pb as f64;
        acc += d * d;
        n += 1;
    }
    for (pa, pb) in a.v_plane().iter().zip(b.v_plane().iter()) {
        let d = *pa as f64 - *pb as f64;
        acc += d * d;
        n += 1;
    }

    if n == 0 { 0.0 } else { acc / (n as f64) }
}

fn mse_planes(a: &Yuv420Frame, b: &Yuv420Frame) -> (f64, f64, f64) {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());
    let mut acc_y = 0f64;
    let mut n_y = 0usize;
    let mut acc_u = 0f64;
    let mut n_u = 0usize;
    let mut acc_v = 0f64;
    let mut n_v = 0usize;

    for (pa, pb) in a.y_plane().iter().zip(b.y_plane().iter()) {
        let d = *pa as f64 - *pb as f64;
        acc_y += d * d;
        n_y += 1;
    }
    for (pa, pb) in a.u_plane().iter().zip(b.u_plane().iter()) {
        let d = *pa as f64 - *pb as f64;
        acc_u += d * d;
        n_u += 1;
    }
    for (pa, pb) in a.v_plane().iter().zip(b.v_plane().iter()) {
        let d = *pa as f64 - *pb as f64;
        acc_v += d * d;
        n_v += 1;
    }
    (
        if n_y == 0 { 0.0 } else { acc_y / (n_y as f64) },
        if n_u == 0 { 0.0 } else { acc_u / (n_u as f64) },
        if n_v == 0 { 0.0 } else { acc_v / (n_v as f64) },
    )
}

fn bench_build_pred(c: &mut Criterion) {
    for &(width, height) in BENCH_DIMS {
        let label = format!("{}x{}", width, height);
        let prev = make_yuv420_pattern(width, height);
        // Mixed integer and sub-pixel motion (about 50% subpixel)
        let mvs_int = make_mvs_half_subpel(width, height, 1, 0);

        // Precompute reference outputs once and compare MSE, outside timing
        let pred_v1 = reference_build_predicted(&prev, width, height, &mvs_int);
        let pred_v2 = build_predicted(&prev, width, height, &mvs_int);
        let mse = mse_frames(&pred_v1, &pred_v2);
        let (ym, um, vm) = mse_planes(&pred_v1, &pred_v2);
        println!("build_pred vs v2 MSE ({}): total {:.3} | Y {:.3} U {:.3} V {:.3}", label, mse, ym, um, vm);

        // Time the original
        let prev_ref = prev.clone();
        let mvs_ref = mvs_int.clone();
        c.bench_function(format!("reference_build_predicted_{}", label).as_str(), move |b| {
            b.iter(|| {
                let _ = reference_build_predicted(&prev_ref, width, height, &mvs_ref);
            })
        });

        // Time the placeholder
        let prev_ref2 = prev.clone();
        let mvs_ref2 = mvs_int.clone();
        c.bench_function(format!("build_predicted_{}", label).as_str(), move |b| {
            b.iter(|| {
                let _ = build_predicted(&prev_ref2, width, height, &mvs_ref2);
            })
        });
    }
}

criterion_group!(build_pred_group, bench_build_pred);
criterion_main!(build_pred_group);
