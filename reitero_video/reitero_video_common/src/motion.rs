use crate::yuv::Yuv420Frame;

const BLOCK_SIZE: usize = 16;

/// Motion vector with half-pixel (sub-pixel) precision
#[derive(Clone, Copy, Debug)]
pub struct MotionVector {
    /// Motion in pixels (signed, -128 to +127)
    dx: i8,
    dy: i8,
    /// Sub-pixel shift: bits 0-1 for X half-pel code, bits 2-3 for Y half-pel code, bit 6 for skip
    /// Half-pixel codes: 0=0.0px, 1=+0.5px, 2=-0.5px, 3=reserved (treated as 0)
    flags: u8,
}

impl MotionVector {
    pub fn new(dx: i8, dy: i8, subpixel_x: i8, subpixel_y: i8, skip: bool) -> Self {
        let mut flags = 0;
        if skip {
            flags |= 0x40;
        }
        flags |= encode_halfpel(subpixel_x);
        flags |= encode_halfpel(subpixel_y) << 2;
        Self { dx, dy, flags }
    }

    pub fn from_raw(dx: i8, dy: i8, flags: u8) -> Self {
        Self { dx, dy, flags }
    }

    pub fn dx(&self) -> i8 {
        self.dx
    }

    pub fn dy(&self) -> i8 {
        self.dy
    }

    pub fn set_dx(&mut self, dx: i8) {
        self.dx = dx;
    }

    pub fn set_dy(&mut self, dy: i8) {
        self.dy = dy;
    }

    pub fn raw_flags(&self) -> u8 {
        self.flags
    }

    /// Subpixel X offset (-1, 0, +1)
    #[inline]
    pub fn subpixel_x(&self) -> i8 {
        decode_halfpel_i8(self.flags & 0x03)
    }

    /// Set subpixel X offset (-1, 0, +1)
    #[inline]
    pub fn set_subpixel_x(&mut self, val: i8) {
        let code = encode_halfpel(val);
        self.flags = (self.flags & !0x03) | (code & 0x03);
    }

    /// Subpixel Y offset (-1, 0, +1)
    #[inline]
    pub fn subpixel_y(&self) -> i8 {
        decode_halfpel_i8((self.flags >> 2) & 0x03)
    }

    /// Set subpixel Y offset (-1, 0, +1)
    #[inline]
    pub fn set_subpixel_y(&mut self, val: i8) {
        let code = encode_halfpel(val);
        self.flags = (self.flags & !(0x03 << 2)) | ((code & 0x03) << 2);
    }

    /// Fractional X offset in half-pixel units (-1, 0, +1)
    #[inline]
    pub fn frac_x_hp(&self) -> i32 {
        self.subpixel_x() as i32
    }

    /// Fractional Y offset in half-pixel units (-1, 0, +1)
    #[inline]
    pub fn frac_y_hp(&self) -> i32 {
        self.subpixel_y() as i32
    }

    /// Fractional X offset in pixels (-0.5, 0.0, +0.5)
    pub fn frac_x(&self) -> f32 {
        0.5 * self.frac_x_hp() as f32
    }

    /// Fractional Y offset in pixels (-0.5, 0.0, +0.5)
    pub fn frac_y(&self) -> f32 {
        0.5 * self.frac_y_hp() as f32
    }

    /// Motion in half-pixel units (integer pixels * 2 + fractional half)
    pub fn dx_hp(&self) -> i32 {
        (self.dx as i32) * 2 + self.frac_x_hp()
    }

    /// Motion in half-pixel units (integer pixels * 2 + fractional half)
    pub fn dy_hp(&self) -> i32 {
        (self.dy as i32) * 2 + self.frac_y_hp()
    }

    /// Check if this block is skipped
    pub fn is_skip(&self) -> bool {
        (self.flags & 0x40) != 0
    }

    /// Set skip status
    pub fn set_skip(&mut self, skip: bool) {
        if skip {
            self.flags |= 0x40;
        } else {
            self.flags &= !0x40;
        }
    }
}

#[inline]
fn decode_halfpel(code: u8) -> i32 {
    match code & 0x03 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

#[inline]
pub fn decode_halfpel_i8(code: u8) -> i8 {
    match code & 0x03 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

#[inline]
fn encode_halfpel(val: i8) -> u8 {
    match val {
        1 => 1,
        -1 => 2,
        _ => 0,
    }
}

// Old unpack_mv_delta_hp removed - using new 3-byte format

#[inline]
fn clamp_hp(v_hp: i32, max_px: i32) -> i32 {
    v_hp.clamp(0, max_px * 2)
}

#[inline]
fn sample_plane_halfpel(plane: &[u8], width: usize, height: usize, x_hp: i32, y_hp: i32) -> u8 {
    let x_hp = clamp_hp(x_hp, width as i32 - 1);
    let y_hp = clamp_hp(y_hp, height as i32 - 1);

    let x0 = (x_hp / 2) as usize;
    let y0 = (y_hp / 2) as usize;
    let x_odd = (x_hp & 1) != 0;
    let y_odd = (y_hp & 1) != 0;

    let x1 = if x0 + 1 < width { x0 + 1 } else { x0 };
    let y1 = if y0 + 1 < height { y0 + 1 } else { y0 };

    let i00 = y0 * width + x0;
    if !x_odd && !y_odd {
        return plane[i00];
    }

    let i10 = y0 * width + x1;
    let i01 = y1 * width + x0;
    let i11 = y1 * width + x1;

    let a = plane[i00] as u16;
    let b = plane[i10] as u16;
    let d = plane[i01] as u16;
    let e = plane[i11] as u16;

    match (x_odd, y_odd) {
        (true, false) => ((a + b + 1) / 2) as u8,
        (false, true) => ((a + d + 1) / 2) as u8,
        (true, true) => ((a + b + d + e + 2) / 4) as u8,
        _ => a as u8,
    }
}

/// Sample YUV420 with half-pixel precision using bilinear interpolation on each plane.
#[inline]
pub fn sample_rgb_halfpel(
    prev: &Yuv420Frame,
    width: usize,
    height: usize,
    x_hp: i32,
    y_hp: i32,
) -> [u8; 3] {
    debug_assert_eq!(prev.width(), width);
    debug_assert_eq!(prev.height(), height);

    let y = sample_plane_halfpel(prev.y_plane(), width, height, x_hp, y_hp);
    let chroma_width = width / 2;
    let chroma_height = height / 2;
    let u = sample_plane_halfpel(
        prev.u_plane(),
        chroma_width,
        chroma_height,
        x_hp / 2,
        y_hp / 2,
    );
    let v = sample_plane_halfpel(
        prev.v_plane(),
        chroma_width,
        chroma_height,
        x_hp / 2,
        y_hp / 2,
    );
    [y, u, v]
}

/// Build predicted frame from previous frame and motion vectors
pub fn build_predicted(
    prev: &Yuv420Frame,
    width: usize,
    height: usize,
    mvs: &[MotionVector],
) -> Yuv420Frame {
    let blocks_w = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let blocks_h = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    debug_assert_eq!(prev.width(), width);
    debug_assert_eq!(prev.height(), height);

    let mut predicted_y = vec![0u8; width * height];
    let mut predicted_u = vec![0u8; (width / 2) * (height / 2)];
    let mut predicted_v = vec![0u8; (width / 2) * (height / 2)];

    // Optimize: sample Y for every pixel, but sample U/V only once per 2x2 luma area.
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let mv = mvs[by * blocks_w + bx];
            let mv_dx_hp = mv.dx_hp();
            let mv_dy_hp = mv.dy_hp();
            let x0 = (bx * BLOCK_SIZE) as i32;
            let y0 = (by * BLOCK_SIZE) as i32;

            for yy in 0..BLOCK_SIZE as i32 {
                let y = (y0 + yy).clamp(0, (height as i32) - 1);
                let y_usize = y as usize;
                let y_idx_base = y_usize * width;
                let even_y = (y_usize & 1) == 0;

                for xx in 0..BLOCK_SIZE as i32 {
                    let x = (x0 + xx).clamp(0, (width as i32) - 1);
                    let x_usize = x as usize;
                    let rx_hp = x * 2 + mv_dx_hp;
                    let ry_hp = y * 2 + mv_dy_hp;

                    // Luma sample
                    let y_sample = sample_plane_halfpel(prev.y_plane(), width, height, rx_hp, ry_hp);
                    predicted_y[y_idx_base + x_usize] = y_sample;

                    // Chroma sample only on even luma coordinates (maps to one chroma sample)
                    if even_y && (x_usize & 1) == 0 {
                        let chroma_width = width / 2;
                        let chroma_height = height / 2;
                        let u_sample = sample_plane_halfpel(
                            prev.u_plane(),
                            chroma_width,
                            chroma_height,
                            rx_hp / 2,
                            ry_hp / 2,
                        );
                        let v_sample = sample_plane_halfpel(
                            prev.v_plane(),
                            chroma_width,
                            chroma_height,
                            rx_hp / 2,
                            ry_hp / 2,
                        );
                        let chroma_x = x_usize / 2;
                        let chroma_y = y_usize / 2;
                        let chroma_idx = chroma_y * chroma_width + chroma_x;
                        predicted_u[chroma_idx] = u_sample;
                        predicted_v[chroma_idx] = v_sample;
                    }
                }
            }
        }
    }

    Yuv420Frame::from_planes(width, height, predicted_y, predicted_u, predicted_v)
        .unwrap_or_else(|e| panic!("failed to build predicted YUV420 frame: {e}"))
}
