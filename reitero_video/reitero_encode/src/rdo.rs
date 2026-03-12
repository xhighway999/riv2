use std::cmp::Ordering;

/// Reason why the RDO pipeline forced a block to keep residuals.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RdoGuardReason {
    Disabled,
    FractionalMv,
    ThresholdExceeded,
}

/// Compact set of lambda-related parameters derived from encoder quality knobs.
#[derive(Debug, Copy, Clone)]
pub struct LambdaEntry {
    pub lambda: f64,
    distortion_retention: f64,
    residual_rate_slope: f64,
    residual_rate_intercept: f64,
    skip_flag_bits: f64,
    keep_flag_bits: f64,
}

impl LambdaEntry {
    pub fn for_quality(quality: u8, lambda_mult: f64) -> Self {
        let quality = quality.clamp(1, 100);
        let qp = quality_to_qp(quality);
        let lambda = lambda_from_qp(qp, lambda_mult);
        let normalized_q = quality as f64 / 100.0;
        // When quality is high we assume transforms preserve more energy (low distortion after coding)
        // and require more bits per unit of SATD. Lower quality means higher residual loss but cheaper bits.
        // Calibrate rates and distortion retention based on quantization step (data-driven proxy)
        // Use quant step from residual quantizer to scale residual bit estimates.
        // Fallback: if residual crate helper not available, approximate q from quality
        // q approximates quantization step; keep previous formula as safe fallback
        let q = reitero_residual::quant_step_from_quality(quality) as f64;
        let residual_rate_slope = 0.5 / q;
        let residual_rate_intercept = 5.0 + 0.56 * q;
        let distortion_retention = 0.15 + ((q - 1.0) / 111.0) * 0.55;
        let skip_flag_bits = 0.55 - normalized_q * 0.15;
        let keep_flag_bits = 1.20 + normalized_q * 0.35;

        Self {
            lambda,
            distortion_retention,
            residual_rate_slope,
            residual_rate_intercept,
            skip_flag_bits,
            keep_flag_bits,
        }
    }

    #[inline]
    fn skip_cost(&self, satd: f64) -> f64 {
        satd + self.lambda * self.skip_flag_bits
    }

    #[inline]
    fn residual_bits(&self, satd: f64) -> f64 {
        let est = self.residual_rate_intercept + self.residual_rate_slope * satd;
        est.max(0.0)
    }

    #[inline]
    fn coded_distortion(&self, satd: f64) -> f64 {
        satd * self.distortion_retention
    }

    #[inline]
    fn coded_cost(&self, satd: f64) -> f64 {
        let distortion = self.coded_distortion(satd);
        let rate = self.keep_flag_bits + self.residual_bits(satd);
        distortion + self.lambda * rate
    }
}

fn quality_to_qp(quality: u8) -> u8 {
    // Map 1..=100 to a rough QP range of 0..=60 (lower QP -> higher quality).
    let inverted = 100u32.saturating_sub(u32::from(quality));
    let qp = ((inverted as f64) * 0.6).round() as i32;
    qp.clamp(0, 60) as u8
}

fn lambda_from_qp(qp: u8, multiplier: f64) -> f64 {
    // x264-style approximation.
    let qp = qp as f64;
    multiplier * f64::powf(2.0, (qp - 12.0) / 3.0)
}

/// Result of evaluating a single block under the RDO heuristic.
#[derive(Debug, Copy, Clone)]
pub struct RdoDecision {
    pub skip: bool,
    pub skip_cost: f64,
    pub coded_cost: f64,
    pub guard_reason: Option<RdoGuardReason>,
}

impl RdoDecision {
    fn forced_keep(skip_cost: f64, coded_cost: f64, reason: RdoGuardReason) -> Self {
        Self {
            skip: false,
            skip_cost,
            coded_cost,
            guard_reason: Some(reason),
        }
    }
}

/// Stateful helper that applies lambda-weighted skip heuristics.
pub struct RdoContext {
    entry: LambdaEntry,
    skip_guard: Option<f64>,
    allow_skips: bool,
}

impl RdoContext {
    pub fn new(
        inter_quality: u8,
        block_area_bytes: usize,
        skip_threshold: u8,
        lambda_mult: f64,
    ) -> Self {
        let entry = LambdaEntry::for_quality(inter_quality, lambda_mult);
        let allow_skips = skip_threshold > 0;
        let skip_guard = if allow_skips {
            Some(block_area_bytes as f64 * skip_threshold as f64)
        } else {
            None
        };
        Self {
            entry,
            skip_guard,
            allow_skips,
        }
    }

    pub fn lambda(&self) -> f64 {
        self.entry.lambda
    }

    pub fn decide(&self, satd: i64, integer_aligned: bool) -> RdoDecision {
        let satd = satd.max(0) as f64;
        let skip_cost = self.entry.skip_cost(satd);
        let coded_cost = self.entry.coded_cost(satd);

        if !self.allow_skips {
            return RdoDecision::forced_keep(skip_cost, coded_cost, RdoGuardReason::Disabled);
        }
        if !integer_aligned {
            return RdoDecision::forced_keep(skip_cost, coded_cost, RdoGuardReason::FractionalMv);
        }
        if let Some(threshold) = self.skip_guard {
            if satd > threshold {
                return RdoDecision::forced_keep(
                    skip_cost,
                    coded_cost,
                    RdoGuardReason::ThresholdExceeded,
                );
            }
        }

        match skip_cost
            .partial_cmp(&coded_cost)
            .unwrap_or(Ordering::Greater)
        {
            Ordering::Less | Ordering::Equal => RdoDecision {
                skip: true,
                skip_cost,
                coded_cost,
                guard_reason: None,
            },
            Ordering::Greater => RdoDecision {
                skip: false,
                skip_cost,
                coded_cost,
                guard_reason: None,
            },
        }
    }
}

/// Aggregated telemetry for a frame.
#[derive(Debug, Clone)]
pub struct RdoFrameSummary {
    pub lambda: f64,
    pub evaluated_blocks: usize,
    pub skip_chosen: usize,
    pub forced_fractional: usize,
    pub forced_threshold: usize,
    pub forced_disabled: usize,
    pub avg_skip_cost: f64,
    pub avg_coded_cost: f64,
}

#[derive(Debug)]
pub struct RdoTelemetry {
    lambda: f64,
    evaluated_blocks: usize,
    skip_chosen: usize,
    forced_fractional: usize,
    forced_threshold: usize,
    forced_disabled: usize,
    total_skip_cost: f64,
    total_coded_cost: f64,
}

impl RdoTelemetry {
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            evaluated_blocks: 0,
            skip_chosen: 0,
            forced_fractional: 0,
            forced_threshold: 0,
            forced_disabled: 0,
            total_skip_cost: 0.0,
            total_coded_cost: 0.0,
        }
    }

    pub fn record(&mut self, decision: &RdoDecision) {
        self.evaluated_blocks += 1;
        if decision.skip {
            self.skip_chosen += 1;
        }
        if let Some(reason) = decision.guard_reason {
            match reason {
                RdoGuardReason::Disabled => self.forced_disabled += 1,
                RdoGuardReason::FractionalMv => self.forced_fractional += 1,
                RdoGuardReason::ThresholdExceeded => self.forced_threshold += 1,
            }
        }
        self.total_skip_cost += decision.skip_cost;
        self.total_coded_cost += decision.coded_cost;
    }

    pub fn finalize(self) -> RdoFrameSummary {
        let avg_skip = if self.evaluated_blocks > 0 {
            self.total_skip_cost / self.evaluated_blocks as f64
        } else {
            0.0
        };
        let avg_coded = if self.evaluated_blocks > 0 {
            self.total_coded_cost / self.evaluated_blocks as f64
        } else {
            0.0
        };
        RdoFrameSummary {
            lambda: self.lambda,
            evaluated_blocks: self.evaluated_blocks,
            skip_chosen: self.skip_chosen,
            forced_fractional: self.forced_fractional,
            forced_threshold: self.forced_threshold,
            forced_disabled: self.forced_disabled,
            avg_skip_cost: avg_skip,
            avg_coded_cost: avg_coded,
        }
    }
}
