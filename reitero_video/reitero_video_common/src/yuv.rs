use std::fmt;

use yuv::{
    YuvChromaSubsampling, YuvConversionMode, YuvPlanarImage, YuvPlanarImageMut, YuvRange,
    YuvStandardMatrix, rgb_to_yuv420, yuv420_to_rgb,
};

const YUV_RANGE: YuvRange = YuvRange::Limited;
const YUV_MATRIX: YuvStandardMatrix = YuvStandardMatrix::Bt709;
const YUV_MODE: YuvConversionMode = YuvConversionMode::Balanced;

/// Errors that can occur while converting between RGB24 and YUV420.
#[derive(Debug)]
pub enum YuvConvertError {
    InvalidInput(String),
    ConversionFailed(String),
}

impl fmt::Display for YuvConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::ConversionFailed(msg) => write!(f, "conversion failed: {msg}"),
        }
    }
}

impl std::error::Error for YuvConvertError {}

/// Contiguous YUV420 frame representation (planar Y, subsampled U/V).
#[derive(Clone, Debug)]
pub struct Yuv420Frame {
    width: usize,
    height: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Yuv420Frame {
    pub fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Result<Self, YuvConvertError> {
        validate_dimensions(width, height)?;
        let expected = width
            .checked_mul(height)
            .and_then(|px| px.checked_mul(3))
            .ok_or_else(|| YuvConvertError::InvalidInput("frame dimensions overflow".into()))?;
        if rgb.len() != expected {
            return Err(YuvConvertError::InvalidInput(format!(
                "rgb len mismatch: expected {expected}, got {}",
                rgb.len()
            )));
        }
        let (width_u32, height_u32) = to_u32_dims(width, height)?;
        let mut yuv =
            YuvPlanarImageMut::<u8>::alloc(width_u32, height_u32, YuvChromaSubsampling::Yuv420);
        rgb_to_yuv420(
            &mut yuv,
            rgb,
            width_u32 * 3,
            YUV_RANGE,
            YUV_MATRIX,
            YUV_MODE,
        )
        .map_err(|e| YuvConvertError::ConversionFailed(format!("rgb_to_yuv420 failed: {e:?}")))?;
        let planar = yuv.to_fixed();
        let (y, u, v) = copy_planes_to_contiguous(&planar, width, height);
        Ok(Self {
            width,
            height,
            y,
            u,
            v,
        })
    }

    pub fn from_planes(
        width: usize,
        height: usize,
        y: Vec<u8>,
        u: Vec<u8>,
        v: Vec<u8>,
    ) -> Result<Self, YuvConvertError> {
        validate_dimensions(width, height)?;
        validate_plane_lengths(width, height, &y, &u, &v)?;
        Ok(Self {
            width,
            height,
            y,
            u,
            v,
        })
    }

    pub fn to_rgb24(&self) -> Result<Vec<u8>, YuvConvertError> {
        let (width_u32, height_u32) = to_u32_dims(self.width, self.height)?;
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|px| px.checked_mul(3))
            .ok_or_else(|| YuvConvertError::InvalidInput("frame dimensions overflow".into()))?;
        let mut rgb = vec![0u8; expected];
        let planar = self.as_planar_image(width_u32, height_u32);
        planar
            .check_constraints(YuvChromaSubsampling::Yuv420)
            .map_err(|e| YuvConvertError::InvalidInput(format!("yuv420 constraints: {e:?}")))?;
        yuv420_to_rgb(&planar, &mut rgb, width_u32 * 3, YUV_RANGE, YUV_MATRIX).map_err(|e| {
            YuvConvertError::ConversionFailed(format!("yuv420_to_rgb failed: {e:?}"))
        })?;
        Ok(rgb)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn y_plane(&self) -> &[u8] {
        &self.y
    }

    pub fn u_plane(&self) -> &[u8] {
        &self.u
    }

    pub fn v_plane(&self) -> &[u8] {
        &self.v
    }

    pub fn y_plane_mut(&mut self) -> &mut [u8] {
        &mut self.y
    }

    pub fn u_plane_mut(&mut self) -> &mut [u8] {
        &mut self.u
    }

    pub fn v_plane_mut(&mut self) -> &mut [u8] {
        &mut self.v
    }

    pub fn clone_planes(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (self.y.clone(), self.u.clone(), self.v.clone())
    }

    pub fn as_planar(&self) -> YuvPlanarImage<'_, u8> {
        let (width_u32, height_u32) =
            to_u32_dims(self.width, self.height).expect("frame dims exceed u32");
        self.as_planar_image(width_u32, height_u32)
    }

    fn as_planar_image(&self, width: u32, height: u32) -> YuvPlanarImage<'_, u8> {
        YuvPlanarImage {
            y_plane: &self.y,
            y_stride: width,
            u_plane: &self.u,
            u_stride: width / 2,
            v_plane: &self.v,
            v_stride: width / 2,
            width,
            height,
        }
    }
}

fn validate_dimensions(width: usize, height: usize) -> Result<(), YuvConvertError> {
    if width == 0 || height == 0 {
        return Err(YuvConvertError::InvalidInput(
            "frame dims must be > 0".into(),
        ));
    }
    if width % 2 != 0 || height % 2 != 0 {
        return Err(YuvConvertError::InvalidInput(
            "frame dims must be even for YUV420".into(),
        ));
    }
    Ok(())
}

fn validate_plane_lengths(
    width: usize,
    height: usize,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Result<(), YuvConvertError> {
    let y_len = width * height;
    let uv_len = (width / 2) * (height / 2);
    if y.len() != y_len || u.len() != uv_len || v.len() != uv_len {
        return Err(YuvConvertError::InvalidInput(format!(
            "plane len mismatch: expected y={y_len}, uv={uv_len}; got y={} u={} v={}",
            y.len(),
            u.len(),
            v.len()
        )));
    }
    Ok(())
}

fn to_u32_dims(width: usize, height: usize) -> Result<(u32, u32), YuvConvertError> {
    let w = u32::try_from(width)
        .map_err(|_| YuvConvertError::InvalidInput("width exceeds u32".into()))?;
    let h = u32::try_from(height)
        .map_err(|_| YuvConvertError::InvalidInput("height exceeds u32".into()))?;
    Ok((w, h))
}

fn copy_planes_to_contiguous(
    yuv: &YuvPlanarImage<'_, u8>,
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let storage_w = width;
    let storage_h = height;
    let y_len = width * height;
    let uv_len = (width / 2) * (height / 2);

    let mut y_plane = vec![0u8; y_len];
    let mut u_plane = vec![0u8; uv_len];
    let mut v_plane = vec![0u8; uv_len];

    for y in 0..storage_h {
        let src_row = (y as u32 * yuv.y_stride) as usize;
        let dst_row = y * storage_w;
        y_plane[dst_row..dst_row + storage_w]
            .copy_from_slice(&yuv.y_plane[src_row..src_row + storage_w]);
    }

    let uv_h = storage_h / 2;
    for y in 0..uv_h {
        let src_u_row = (y as u32 * yuv.u_stride) as usize;
        let src_v_row = (y as u32 * yuv.v_stride) as usize;
        let dst_row = y * (storage_w / 2);
        u_plane[dst_row..dst_row + (storage_w / 2)]
            .copy_from_slice(&yuv.u_plane[src_u_row..src_u_row + (storage_w / 2)]);
        v_plane[dst_row..dst_row + (storage_w / 2)]
            .copy_from_slice(&yuv.v_plane[src_v_row..src_v_row + (storage_w / 2)]);
    }

    (y_plane, u_plane, v_plane)
}
