use crate::error::{EncodeError, Result};

/// Keyframe (I-frame) interval configuration
#[derive(Debug, Clone, Copy)]
pub enum KeyframeInterval {
    /// Every frame is an I-frame (no compression between frames)
    AllIntra,
    /// Automatic keyframe placement based on content
    Automatic,
    /// Fixed interval - I-frame every N frames
    Fixed(u64),
}

impl Default for KeyframeInterval {
    fn default() -> Self {
        // Default back to classic GOP=30 (I-frame every 30 frames).
        KeyframeInterval::Fixed(30)
    }
}

/// Configuration for the Reitero video encoder.
///
/// Construct with [`EncoderConfig::new`] and customize with the `with_*` builder methods.
pub struct EncoderConfig {
    /// Display (original) dimensions.
    pub display_width: u32,
    pub display_height: u32,
    /// Storage dimensions (padded, used for block processing).
    pub storage_width: u32,
    pub storage_height: u32,
    pub fps: u32,
    pub keyframe_interval: KeyframeInterval,
    pub intra_quality: u8, // 1-100 quality for intra residuals
    pub inter_quality: u8, // 1-100 quality for inter residual maps
    /// Motion search range (±N pixels). Must fit in i8 (<=127).
    pub search_range: u8,
    /// Motion vector DEFLATE compression level (miniz_oxide), 0..=10.
    pub mv_deflate_level: u8,
    /// Block skip SAD threshold (average abs diff per byte). If block_sad <= threshold*(16*16*3),
    /// and MV is integer-aligned, encoder can mark block as skip (no residual stored).
    pub skip_threshold: u8,
    /// Early termination threshold for zero-MV search.
    /// If SAD(0,0) <= threshold, skip search. 0 = disabled.
    pub me_zero_mv_threshold: i64,
    /// Early termination threshold for predictor-based search.
    /// If best predictor SAD <= threshold, skip refinement. 0 = disabled.
    pub me_predictor_threshold: i64,
    /// RDO Lambda multiplier. Default: 0.49
    pub rdo_lambda_mult: f64,
    /// Inter dead zone threshold. Coefficients with |coeff/step| < dead_zone quantize to 0.
    /// 0.5 = standard rounding, 0.75 = H.264-style wider dead zone (default).
    pub inter_dead_zone: f32,
}

impl EncoderConfig {
    pub fn new(display_width: u32, display_height: u32, fps: u32) -> Self {
        let storage_width = round_up_to_multiple(display_width, 16);
        let storage_height = round_up_to_multiple(display_height, 16);

        Self {
            display_width,
            display_height,
            storage_width,
            storage_height,
            fps,
            keyframe_interval: KeyframeInterval::default(),
            intra_quality: 90,
            inter_quality: 35,
            search_range: 12,
            mv_deflate_level: 6, // Default zlib/deflate level
            skip_threshold: 3,
            me_zero_mv_threshold: 0,
            me_predictor_threshold: 0,
            rdo_lambda_mult: 0.49,
            inter_dead_zone: 0.75,
        }
    }

    pub fn with_rdo_lambda_mult(mut self, mult: f64) -> Self {
        self.rdo_lambda_mult = mult;
        self
    }

    pub fn with_inter_dead_zone(mut self, dead_zone: f32) -> Self {
        self.inter_dead_zone = dead_zone;
        self
    }

    pub fn with_intra_quality(mut self, quality: u8) -> Self {
        self.intra_quality = quality;
        self
    }

    pub fn with_inter_quality(mut self, quality: u8) -> Self {
        self.inter_quality = quality;
        self
    }

    pub fn with_keyframe_interval(mut self, interval: KeyframeInterval) -> Self {
        self.keyframe_interval = interval;
        self
    }

    pub fn with_search_range(mut self, search_range: u8) -> Self {
        self.search_range = search_range;
        self
    }

    pub fn with_mv_deflate_level(mut self, level: u8) -> Self {
        self.mv_deflate_level = level;
        self
    }

    pub fn with_skip_threshold(mut self, threshold: u8) -> Self {
        self.skip_threshold = threshold;
        self
    }

    pub fn with_me_zero_mv_threshold(mut self, threshold: i64) -> Self {
        self.me_zero_mv_threshold = threshold;
        self
    }

    pub fn with_me_predictor_threshold(mut self, threshold: i64) -> Self {
        self.me_predictor_threshold = threshold;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.display_width == 0 || self.display_height == 0 {
            return Err(EncodeError::InvalidConfig(
                "Width and height must be greater than 0".to_string(),
            ));
        }
        if self.storage_width < self.display_width || self.storage_height < self.display_height {
            return Err(EncodeError::InvalidConfig(
                "Storage dimensions must be >= display dimensions".to_string(),
            ));
        }
        if self.fps == 0 {
            return Err(EncodeError::InvalidConfig(
                "FPS must be greater than 0".to_string(),
            ));
        }
        if self.search_range > 127 {
            return Err(EncodeError::InvalidConfig(
                "search_range must be <= 127 for current MV encoding".to_string(),
            ));
        }
        if self.mv_deflate_level > 10 {
            return Err(EncodeError::InvalidConfig(
                "mv_deflate_level must be in 0..=10".to_string(),
            ));
        }
        if let KeyframeInterval::Fixed(interval) = self.keyframe_interval {
            if interval == 0 {
                return Err(EncodeError::InvalidConfig(
                    "Keyframe interval must be greater than 0".to_string(),
                ));
            }
        }
        Ok(())
    }
}

fn round_up_to_multiple(v: u32, m: u32) -> u32 {
    if m == 0 {
        return v;
    }
    ((v + m - 1) / m) * m
}
