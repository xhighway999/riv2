use yuv::{
    YuvChromaSubsampling, YuvConversionMode, YuvPlanarImage, YuvPlanarImageMut, YuvRange,
    YuvStandardMatrix, rgb_to_yuv420, yuv420_to_rgb,
};

use crate::ResidualError;

/// YUV conversion constants
const YUV_RANGE: YuvRange = YuvRange::Limited;
const YUV_MATRIX: YuvStandardMatrix = YuvStandardMatrix::Bt709;
const YUV_MODE: YuvConversionMode = YuvConversionMode::Balanced;

/// Convert RGB24 to YUV420 planar image
pub fn rgb24_to_yuv420(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<YuvPlanarImageMut<'static, u8>, ResidualError> {
    let mut yuv = YuvPlanarImageMut::<u8>::alloc(width, height, YuvChromaSubsampling::Yuv420);
    rgb_to_yuv420(&mut yuv, rgb, width * 3, YUV_RANGE, YUV_MATRIX, YUV_MODE)
        .map_err(|e| ResidualError::InvalidInput(format!("rgb_to_yuv420 failed: {e:?}")))?;
    Ok(yuv)
}

/// Copy YUV420 planes from a YuvPlanarImage (which may have padding) to contiguous arrays
/// with stride = width. Returns (y_plane, u_plane, v_plane) as contiguous Vec<u8>.
pub fn copy_yuv420_planes_to_contiguous(
    yuv: &YuvPlanarImage<u8>,
    width: u32,
    height: u32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let storage_w = width as usize;
    let storage_h = height as usize;
    let y_len = (width * height) as usize;
    let uv_len = ((width / 2) * (height / 2)) as usize;

    let mut y_plane = vec![0u8; y_len];
    let mut u_plane = vec![0u8; uv_len];
    let mut v_plane = vec![0u8; uv_len];

    // Copy Y plane
    for y in 0..storage_h {
        let src_row = (y as u32 * yuv.y_stride) as usize;
        let dst_row = y * storage_w;
        for x in 0..storage_w {
            y_plane[dst_row + x] = yuv.y_plane[src_row + x];
        }
    }

    // Copy U/V planes (YUV420: half width and half height)
    let uv_h = storage_h / 2;
    for y in 0..uv_h {
        let src_u_row = (y as u32 * yuv.u_stride) as usize;
        let src_v_row = (y as u32 * yuv.v_stride) as usize;
        let dst_row = y * (storage_w / 2);
        for x in 0..(storage_w / 2) {
            u_plane[dst_row + x] = yuv.u_plane[src_u_row + x];
            v_plane[dst_row + x] = yuv.v_plane[src_v_row + x];
        }
    }

    (y_plane, u_plane, v_plane)
}

/// Build a YuvPlanarImage from contiguous planes and convert to RGB24
pub fn yuv420_planes_to_rgb24(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ResidualError> {
    let expected_rgb = (width * height * 3) as usize;
    let mut rgb = vec![0u8; expected_rgb];

    let planar = YuvPlanarImage {
        y_plane,
        y_stride: width,
        u_plane,
        u_stride: width / 2,
        v_plane,
        v_stride: width / 2,
        width,
        height,
    };

    planar
        .check_constraints(YuvChromaSubsampling::Yuv420)
        .map_err(|e| ResidualError::InvalidInput(format!("yuv420 constraints: {e:?}")))?;

    yuv420_to_rgb(&planar, &mut rgb, width * 3, YUV_RANGE, YUV_MATRIX)
        .map_err(|e| ResidualError::InvalidInput(format!("yuv420_to_rgb failed: {e:?}")))?;

    Ok(rgb)
}
