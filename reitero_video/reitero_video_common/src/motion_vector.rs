/// Motion vector with half-pixel (sub-pixel) precision
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionVector {
    /// Motion in pixels (signed, -128 to +127)
    dx: i8,
    dy: i8,
    /// Subpixel offsets in half-pixel units (-1, 0, +1)
    subpixel_x: i8,
    subpixel_y: i8,
    /// Skip flag
    skip: bool,
}

impl MotionVector {
    pub fn new(dx: i8, dy: i8, subpixel_x: i8, subpixel_y: i8, skip: bool) -> Self {
        // Clamp subpixel to allowed values {-1,0,1}
        let spx = match subpixel_x {
            -1 | 0 | 1 => subpixel_x,
            v if v < -1 => -1,
            v if v > 1 => 1,
            _ => 0,
        };
        let spy = match subpixel_y {
            -1 | 0 | 1 => subpixel_y,
            v if v < -1 => -1,
            v if v > 1 => 1,
            _ => 0,
        };
        Self { dx, dy, subpixel_x: spx, subpixel_y: spy, skip }
    }

    pub fn dx(&self) -> i8 { self.dx }
    pub fn dy(&self) -> i8 { self.dy }
    pub fn set_dx(&mut self, dx: i8) { self.dx = dx; }
    pub fn set_dy(&mut self, dy: i8) { self.dy = dy; }

    /// Subpixel X offset (-1, 0, +1)
    #[inline]
    pub fn subpixel_x(&self) -> i8 { self.subpixel_x }
    /// Set subpixel X offset (-1, 0, +1)
    #[inline]
    pub fn set_subpixel_x(&mut self, val: i8) {
        self.subpixel_x = match val { -1 | 0 | 1 => val, v if v < -1 => -1, _ => 1 };
    }

    /// Subpixel Y offset (-1, 0, +1)
    #[inline]
    pub fn subpixel_y(&self) -> i8 { self.subpixel_y }
    /// Set subpixel Y offset (-1, 0, +1)
    #[inline]
    pub fn set_subpixel_y(&mut self, val: i8) {
        self.subpixel_y = match val { -1 | 0 | 1 => val, v if v < -1 => -1, _ => 1 };
    }

    /// Fractional X offset in half-pixel units (-1, 0, +1)
    #[inline]
    pub fn frac_x_hp(&self) -> i32 { self.subpixel_x as i32 }
    /// Fractional Y offset in half-pixel units (-1, 0, +1)
    #[inline]
    pub fn frac_y_hp(&self) -> i32 { self.subpixel_y as i32 }

    /// Fractional X offset in pixels (-0.5, 0.0, +0.5)
    pub fn frac_x(&self) -> f32 { 0.5 * self.frac_x_hp() as f32 }
    /// Fractional Y offset in pixels (-0.5, 0.0, +0.5)
    pub fn frac_y(&self) -> f32 { 0.5 * self.frac_y_hp() as f32 }

    /// Motion in half-pixel units (integer pixels * 2 + fractional half)
    pub fn dx_hp(&self) -> i32 { (self.dx as i32) * 2 + self.frac_x_hp() }
    /// Motion in half-pixel units (integer pixels * 2 + fractional half)
    pub fn dy_hp(&self) -> i32 { (self.dy as i32) * 2 + self.frac_y_hp() }

    /// Check if this block is skipped
    pub fn is_skip(&self) -> bool { self.skip }
    /// Set skip status
    pub fn set_skip(&mut self, skip: bool) { self.skip = skip; }

    /// Encode this motion vector's fractional and skip data into the legacy flag byte.
    /// bits 0-1: X half-pel (0=0, 1=+0.5, 2=-0.5)
    /// bits 2-3: Y half-pel (0=0, 1=+0.5, 2=-0.5)
    /// bit 6: skip flag
    pub fn to_flags(&self) -> u8 {
        let mut f = 0u8;
        match self.subpixel_x {
            1 => f |= 0x01,
            -1 => f |= 0x02,
            _ => {}
        }
        match self.subpixel_y {
            1 => f |= 0x04,
            -1 => f |= 0x08,
            _ => {}
        }
        if self.skip { f |= 0x40; }
        f
    }

    /// Return a canonicalized copy of this motion vector.
    /// Rules:
    /// - If `dx == -128` and `subpixel_x == -1` (−0.5), force `subpixel_x = 0`.
    /// - If `dy == -128` and `subpixel_y == -1` (−0.5), force `subpixel_y = 0`.
    /// - Otherwise, convert negative half-pel to positive by borrowing 1 pixel:
    ///   `(-0.5, dx)` → `(+0.5, dx-1)`, and same for Y. This preserves motion in half-pel units.
    pub fn as_canonicalized(mut self) -> Self {
        if self.dx == i8::MIN && self.subpixel_x == -1 { self.subpixel_x = 0; }
        if self.dy == i8::MIN && self.subpixel_y == -1 { self.subpixel_y = 0; }
        // Normalize remaining negative half-pel to positive by adjusting integer component
        if self.subpixel_x == -1 {
            self.subpixel_x = 1;
            self.dx = self.dx.saturating_sub(1);
        }
        if self.subpixel_y == -1 {
            self.subpixel_y = 1;
            self.dy = self.dy.saturating_sub(1);
        }
        self
    }
}
