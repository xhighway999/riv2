/// Fixed-point scale used by the forward and inverse DCT kernels.
///
/// All intermediate products are accumulated at this bit width above the input
/// scale before being shifted back down. `2^18 = 262144`.
pub const CONST_SCALE: u32 = 18;
