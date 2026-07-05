// Custom ReItero video format structures

use crate::MotionVector;

/// Magic bytes for ReItero Video format: "RIV\0"
pub const RIV_MAGIC: [u8; 4] = *b"RIV\0";

/// Format version
pub const RIV_VERSION: u32 = 5;

/// Frame type for video compression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    /// Intra frame (I-frame) - self-contained, doesn't depend on other frames
    Intra = 1,
    /// Inter frame (P-frame) - depends on previous frames
    Inter = 2,
}

impl FrameType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(FrameType::Intra),
            2 => Some(FrameType::Inter),
            _ => None,
        }
    }
}

/// Header for the custom video format
/// All frames are RGB24 format (3 bytes per pixel)
#[derive(Debug, Clone)]
pub struct VideoHeader {
    /// Display (original) dimensions.
    pub display_width: u32,
    pub display_height: u32,
    /// Storage dimensions (padded, used for block processing).
    pub storage_width: u32,
    pub storage_height: u32,
    pub fps: u32,
    pub frame_count: u64,
}

impl VideoHeader {
    pub fn new(
        display_width: u32,
        display_height: u32,
        storage_width: u32,
        storage_height: u32,
        fps: u32,
    ) -> Self {
        Self {
            display_width,
            display_height,
            storage_width,
            storage_height,
            fps,
            frame_count: 0,
        }
    }

    /// Serialize header to bytes (little-endian)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(36);

        // Magic (4 bytes)
        bytes.extend_from_slice(&RIV_MAGIC);

        // Version (4 bytes)
        bytes.extend_from_slice(&RIV_VERSION.to_le_bytes());

        // Display width/height (4+4)
        bytes.extend_from_slice(&self.display_width.to_le_bytes());
        bytes.extend_from_slice(&self.display_height.to_le_bytes());

        // Storage width/height (4+4)
        bytes.extend_from_slice(&self.storage_width.to_le_bytes());
        bytes.extend_from_slice(&self.storage_height.to_le_bytes());

        // FPS (4 bytes)
        bytes.extend_from_slice(&self.fps.to_le_bytes());

        // Frame count (8 bytes)
        bytes.extend_from_slice(&self.frame_count.to_le_bytes());

        bytes
    }

    /// Get the size of the header in bytes
    pub const fn header_size() -> usize {
        4 + 4 + 4 + 4 + 4 + 4 + 4 + 8 // magic + version + disp_w + disp_h + stor_w + stor_h + fps + frame_count
    }

    /// Get expected frame data size (RGB24 = 3 bytes per pixel)
    pub fn frame_data_size(&self) -> usize {
        (self.storage_width * self.storage_height * 3) as usize
    }
}

/// Frame payload (Rust "union-like" representation).
///
/// V4 format:
/// - Intra: timestamp(u64) + type(u8=1) + quality(u8) + size(u32) + residual_data
/// - Inter: timestamp(u64) + type(u8=2) + quality(u8) + global_mv(dx,dy,flags) + mv_size(u32) + mv_deflate + res_size(u32) + residual_rans_bytes
#[derive(Debug, Clone)]
pub enum PackedFrameData {
    /// DCT+RANS-encoded YUV420 intra frame.
    Intra { quality: u8, residual_data: Vec<u8> },
    /// Inter frame with motion vectors, a frame-level predictor, and residual data.
    Inter {
        /// Quality parameter (1-100, where 1 = lowest quality/max quantization, 100 = highest quality/min quantization)
        quality: u8,
        /// Frame-level motion-delta bias stored as a MotionVector (dx/dy shift, flags neutral)
        global_mv: MotionVector,
        /// DEFLATE-compressed motion vector bytes (delta-coded packed bytes, 2 bytes per 16x16 block).
        mv_deflate: Vec<u8>,
        /// DEFLATE-compressed residual data bytes (internal format: quantized i16 DCT coefficients in zigzag order)
        residual_yuv420: Vec<u8>,
    },
}

/// Packed frame (V2).
#[derive(Debug, Clone)]
pub struct PackedFrame {
    pub timestamp_ms: u64,
    pub data: PackedFrameData,
}

impl PackedFrame {
    pub fn new_intra(quality: u8, residual_data: Vec<u8>, timestamp_ms: u64) -> Self {
        Self {
            timestamp_ms,
            data: PackedFrameData::Intra { quality, residual_data },
        }
    }

    pub fn new_inter_with_mv(
        quality: u8,
        global_mv: MotionVector,
        mv_deflate: Vec<u8>,
        residual_yuv420: Vec<u8>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            timestamp_ms,
            data: PackedFrameData::Inter {
                quality,
                global_mv,
                mv_deflate,
                residual_yuv420,
            },
        }
    }

    pub fn frame_type(&self) -> FrameType {
        match self.data {
            PackedFrameData::Intra { .. } => FrameType::Intra,
            PackedFrameData::Inter { .. } => FrameType::Inter,
        }
    }

    pub fn payload(&self) -> &[u8] {
        match &self.data {
            PackedFrameData::Intra { residual_data, .. } => residual_data,
            PackedFrameData::Inter { residual_yuv420, .. } => residual_yuv420,
        }
    }

    /// Serialize frame to bytes (little-endian), V2 format.
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.data {
            PackedFrameData::Intra { quality, residual_data } => {
                let size = residual_data.len() as u32;
                let mut bytes = Vec::with_capacity(8 + 1 + 1 + 4 + residual_data.len());
                bytes.extend_from_slice(&self.timestamp_ms.to_le_bytes());
                bytes.push(FrameType::Intra as u8);
                bytes.push(*quality);
                bytes.extend_from_slice(&size.to_le_bytes());
                bytes.extend_from_slice(residual_data);
                bytes
            }
            PackedFrameData::Inter {
                quality,
                global_mv,
                mv_deflate,
                residual_yuv420,
            } => {
                let mv_size = mv_deflate.len() as u32;
                let res_size = residual_yuv420.len() as u32;
                let mut bytes = Vec::with_capacity(
                    8 + 1 + 1 + 3 + 4 + mv_deflate.len() + 4 + residual_yuv420.len(),
                );
                bytes.extend_from_slice(&self.timestamp_ms.to_le_bytes());
                bytes.push(FrameType::Inter as u8);
                bytes.push(*quality);
                bytes.push(global_mv.dx() as u8);
                bytes.push(global_mv.dy() as u8);
                bytes.push(global_mv.to_flags());
                bytes.extend_from_slice(&mv_size.to_le_bytes());
                bytes.extend_from_slice(mv_deflate);
                bytes.extend_from_slice(&res_size.to_le_bytes());
                bytes.extend_from_slice(residual_yuv420);
                bytes
            }
        }
    }

    /// Deserialize a PackedFrame from bytes (little-endian), V2 format.
    pub fn from_bytes(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < 9 {
            return None;
        }
        let timestamp_ms = u64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]);
        let frame_type = FrameType::from_u8(buf[8])?;
        match frame_type {
            FrameType::Intra => {
                // layout: quality(u8) + size(u32) + residual_data
                if buf.len() < 14 {
                    return None;
                }
                let quality = buf[9];
                let size = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]) as usize;
                let total = 8 + 1 + 1 + 4 + size;
                if buf.len() < total {
                    return None;
                }
                let residual_data = buf[14..14 + size].to_vec();
                Some((PackedFrame::new_intra(quality, residual_data, timestamp_ms), total))
            }
            FrameType::Inter => {
                if buf.len() < 13 {
                    return None;
                }
                let quality = buf[9];
                let flags = buf[12];
                let global_mv = MotionVector::new(
                    buf[10] as i8,
                    buf[11] as i8,
                    crate::decode_halfpel_i8(flags & 0x03),
                    crate::decode_halfpel_i8((flags >> 2) & 0x03),
                    (flags & 0x40) != 0,
                );
                if buf.len() < 17 {
                    return None;
                }
                let mv_size = u32::from_le_bytes([buf[13], buf[14], buf[15], buf[16]]) as usize;
                let mv_start = 17;
                let mv_end = mv_start + mv_size;
                if buf.len() < mv_end + 4 {
                    return None;
                }
                let res_size = u32::from_le_bytes([
                    buf[mv_end],
                    buf[mv_end + 1],
                    buf[mv_end + 2],
                    buf[mv_end + 3],
                ]) as usize;
                let res_start = mv_end + 4;
                let res_end = res_start + res_size;
                if buf.len() < res_end {
                    return None;
                }
                let mv_deflate = buf[mv_start..mv_end].to_vec();
                let residual_yuv420 = buf[res_start..res_end].to_vec();
                Some((
                    PackedFrame::new_inter_with_mv(
                        quality,
                        global_mv,
                        mv_deflate,
                        residual_yuv420,
                        timestamp_ms,
                    ),
                    res_end,
                ))
            }
        }
    }
}

/// Convert Vec<i16> to little-endian bytes
pub fn i16_vec_to_bytes(vec: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vec.len() * 2);
    for &value in vec {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Convert little-endian bytes to Vec<i16>
pub fn bytes_to_i16_vec(bytes: &[u8]) -> Option<Vec<i16>> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    let mut vec = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        vec.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Some(vec)
}
