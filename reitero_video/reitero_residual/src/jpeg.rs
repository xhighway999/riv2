use jpeg_decoder::Decoder as JpegDecoder;
use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
use std::io::Cursor;

use crate::residual::{ResidualError, Result};

pub fn encode_jpeg_rgb(rgb24: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let enc = JpegEncoder::new(&mut out, quality);
    enc.encode(rgb24, width as u16, height as u16, JpegColorType::Rgb)
        .map_err(|e| ResidualError::JpegEncode(format!("{e:?}")))?;
    Ok(out)
}

pub fn decode_jpeg_rgb(jpeg: &[u8]) -> Result<Vec<u8>> {
    let (rgb, _w, _h) = decode_jpeg_rgb_with_dims(jpeg)?;
    Ok(rgb)
}

pub fn decode_jpeg_rgb_with_dims(jpeg: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let mut dec = JpegDecoder::new(Cursor::new(jpeg));
    let pixels = dec
        .decode()
        .map_err(|e| ResidualError::JpegDecode(format!("{e:?}")))?;
    let info = dec
        .info()
        .ok_or_else(|| ResidualError::JpegDecode("missing jpeg info".to_string()))?;
    if info.pixel_format != jpeg_decoder::PixelFormat::RGB24 {
        return Err(ResidualError::JpegDecode(format!(
            "unexpected pixel format: {:?}",
            info.pixel_format
        )));
    }
    Ok((pixels, info.width as u32, info.height as u32))
}
