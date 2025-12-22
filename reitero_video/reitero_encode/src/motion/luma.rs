//! Internal luminance storage helpers with stricter alignment.

use std::ops::{Add, Mul, Sub};

#[cfg(not(feature = "simd"))]
use super::satd_scalar;
#[cfg(feature = "simd")]
use super::satd_simd;

/// Fixed-point Y samples scaled by 2^8 and aligned to 32-bit boundaries so the
/// backing `Vec` can satisfy SIMD-friendly load requirements.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MotionLuma(i32);

impl MotionLuma {
    #[inline]
    pub(crate) fn from_fixed_point(value: i32) -> Self {
        debug_assert!(value <= u16::MAX as i32);
        Self(value as i32)
    }

    pub(crate) fn new(value: i32) -> Self {
        Self::from_fixed_point(value as i32)
    }

    #[inline]
    pub(crate) fn to_satd_sample(self) -> i32 {
        self.0 >> 8
    }

    #[inline]
    pub(crate) fn as_fixed_point(self) -> u16 {
        self.0 as u16
    }
}

pub(crate) struct LumaPlane {
    pub(crate) data: Vec<MotionLuma>,
}

impl LumaPlane {
    pub(super) fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Self {
        #[cfg(feature = "simd")]
        return satd_simd::rgb_to_luma_plane(rgb, width, height);
        #[cfg(not(feature = "simd"))]
        return satd_scalar::rgb_to_luma_plane(rgb, width, height);
    }

    pub(super) fn from_y_plane(y: &[u8], width: usize, height: usize) -> Self {
        assert_eq!(y.len(), width * height, "Y plane length mismatch");
        let mut data = Vec::with_capacity(y.len());
        for &sample in y {
            data.push(MotionLuma::from((sample as i32) << 8));
        }
        Self { data }
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[MotionLuma] {
        &self.data
    }

    pub fn as_i32_slice(&self) -> &[i32] {
        unsafe {
            std::slice::from_raw_parts(
                self.as_slice().as_ptr() as *const i32,
                self.as_slice().len(),
            )
        }
    }
}

impl From<Vec<MotionLuma>> for LumaPlane {
    fn from(data: Vec<MotionLuma>) -> Self {
        Self { data }
    }
}

impl From<MotionLuma> for i32 {
    #[inline]
    fn from(value: MotionLuma) -> Self {
        value.0
    }
}
impl From<u32> for MotionLuma {
    #[inline]
    fn from(value: u32) -> Self {
        MotionLuma(value as i32)
    }
}

impl From<i32> for MotionLuma {
    #[inline]
    fn from(value: i32) -> Self {
        MotionLuma(value)
    }
}

impl Add for MotionLuma {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for MotionLuma {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul for MotionLuma {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // When multiplying two fixed-point numbers, the fractional
        // bits double. We must shift back down to maintain the scale.
        Self((self.0 as i64 * rhs.0 as i64) as i32)
    }
}
