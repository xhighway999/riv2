//! Fixed-point DCT/IDCT transforms for the Reitero video codec.
//!
//! Provides 8×8 and 16×16 block transforms with optional SIMD acceleration
//! via the [`wide`](https://crates.io/crates/wide) crate.
//!
//! # Overview
//!
//! - **Encoding** (forward DCT + quantization): [`encode_plane_8x8_aq`], [`encode_plane_8x8_matrix`],
//!   [`encode_plane_16x16_aq`], [`encode_plane_16x16_matrix`]
//! - **Decoding** (dequantization + inverse DCT): [`decode_plane_8x8_aq`], [`decode_plane_8x8_matrix`],
//!   [`decode_plane_16x16_aq`], [`decode_plane_16x16_matrix`]
//!
//! The 8×8 encoder uses a matrix-multiply formulation; the 8×8 decoder uses a fast
//! AAN-style IDCT derived from stb_image/jpeg-decoder. The 16×16 transforms use
//! symmetric encode/decode kernels.
//!
//! All functions operate on signed `i16` residual planes (centered at 0, not 0-255).
//!
//! # Example
//!
//! ```
//! use reitero_dct::{encode_plane_8x8_aq, decode_plane_8x8_aq};
//!
//! let plane: Vec<i16> = (0..64).map(|i| (i as i16) - 32).collect();
//! let encoded = encode_plane_8x8_aq(&plane, 8, 8, 8, &[16.0f32; 1], None, 0.5);
//! let coeffs: Vec<i16> = encoded[0].as_ref().unwrap().clone();
//!
//! let mut decoded = vec![0i16; 64];
//! decode_plane_8x8_aq(&coeffs, &mut decoded, 8, 8, 8, &[16.0f32; 1], &[false]);
//! ```

/// Maximum supported block dimension (width or height) in pixels.
pub const MAX_BLOCK_SIZE: usize = 32;

/// Maximum supported block area in pixels (`MAX_BLOCK_SIZE²`).
pub const MAX_BLOCK_AREA: usize = MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;

mod common;
mod dct8;
mod dct16;
mod fast_dct8;

pub use dct8::{encode_plane_8x8_aq, encode_plane_8x8_matrix};
pub use dct16::{decode_plane_16x16_aq, decode_plane_16x16_matrix, encode_plane_16x16_aq, encode_plane_16x16_matrix};
pub use fast_dct8::{decode_plane_8x8_aq, decode_plane_8x8_matrix};
