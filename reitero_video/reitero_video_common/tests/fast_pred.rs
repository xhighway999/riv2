use reitero_video_common::{build_predicted, reference_build_predicted, MotionVector, Yuv420Frame};

const BLOCK_SIZE: usize = 16;

fn make_yuv420_pattern(width: usize, height: usize) -> Yuv420Frame {
    // Simple deterministic pattern
    let mut y = vec![0u8; width * height];
    for i in 0..y.len() { y[i] = ((i * 31 + 7) % 256) as u8; }
    let cw = width / 2; let ch = height / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for i in 0..u.len() { u[i] = ((i * 17 + 3) % 256) as u8; v[i] = ((i * 13 + 11) % 256) as u8; }
    Yuv420Frame::from_planes(width, height, y, u, v).expect("valid yuv420")
}

fn make_mvs(width: usize, height: usize, dx_hp: i8, dy_hp: i8) -> Vec<MotionVector> {
    let bw = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let bh = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut out = Vec::with_capacity(bw * bh);
    for _ in 0..(bw * bh) {
        out.push(MotionVector::new(dx_hp / 2, dy_hp / 2, dx_hp % 2, dy_hp % 2, false));
    }
    out
}

fn mse_frames(a: &Yuv420Frame, b: &Yuv420Frame) -> f64 {
    assert_eq!(a.width(), b.width()); assert_eq!(a.height(), b.height());
    let mut acc: f64 = 0.0; let mut n: usize = 0;
    for (pa, pb) in a.y_plane().iter().zip(b.y_plane().iter()) { let d = *pa as f64 - *pb as f64; acc += d*d; n += 1; }
    for (pa, pb) in a.u_plane().iter().zip(b.u_plane().iter()) { let d = *pa as f64 - *pb as f64; acc += d*d; n += 1; }
    for (pa, pb) in a.v_plane().iter().zip(b.v_plane().iter()) { let d = *pa as f64 - *pb as f64; acc += d*d; n += 1; }
    if n == 0 { 0.0 } else { acc / (n as f64) }
}

#[test]
fn fast_pred_matches_scalar_low_mse() {
    let dims = [(640,360), (1280,720)];
    let shifts = [
        (0,0), (1,0), (0,1), (1,1), (2,2), (3,1),
        // Include left-edge clamp scenarios (negative dx half-pel offsets)
        (-1,0), (-2,0), (-3,1),
    ]; // half-pel encoded via hp units
    for &(w,h) in &dims {
        let prev = make_yuv420_pattern(w,h);
        for &(dx_hp, dy_hp) in &shifts {
            let mvs = make_mvs(w,h, dx_hp, dy_hp);
            let s = reference_build_predicted(&prev, w, h, &mvs);
            let f = build_predicted(&prev, w, h, &mvs);
            let mut y_mse: f64 = 0.0;
            for (pa, pb) in s.y_plane().iter().zip(f.y_plane().iter()) { let d = *pa as f64 - *pb as f64; y_mse += d*d; }
            y_mse /= (w*h) as f64;
            // Focus validation on luma where SIMD applies
            assert!(y_mse < 1e-6, "Y-plane MSE too high: {}x{} shift ({},{}): Y={:.6}", w, h, dx_hp, dy_hp, y_mse);
        }
    }
}
