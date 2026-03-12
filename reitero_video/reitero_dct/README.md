# reitero_dct

Fixed-point DCT/IDCT transforms for the [Reitero video format](https://github.com/xhighway999/riv).

## Usage

```toml
[dependencies]
reitero_dct = "0.1"
```

### Encode and decode an 8×8 block

```rust
use reitero_dct::{encode_plane_8x8, decode_plane_8x8};

let plane: Vec<i16> = (0..64).map(|i| (i as i16) - 32).collect();
let quant_step = 16.0_f32;

// Forward DCT + quantization
let encoded = encode_plane_8x8(&plane, 8, 8, 8, quant_step, None);
let coeffs: Vec<i16> = encoded[0].as_ref().unwrap().clone();

// Dequantization + inverse DCT
let mut decoded = vec![0i16; 64];
decode_plane_8x8(&coeffs, &mut decoded, 8, 8, 8, quant_step, &[false]);
```

## API

| Function | Direction | Block size |
|---|---|---|
| `encode_plane_8x8` / `encode_plane_8x8_matrix` | Forward DCT | 8×8 |
| `decode_plane_8x8` / `decode_plane_8x8_matrix` | Inverse DCT | 8×8 |
| `encode_plane_16x16` / `encode_plane_16x16_matrix` | Forward DCT | 16×16 |
| `decode_plane_16x16` / `decode_plane_16x16_matrix` | Inverse DCT | 16×16 |

All functions operate on signed `i16` residual planes (centered at 0, not 0–255).

## Performance

SIMD acceleration is provided unconditionally via the [`wide`](https://crates.io/crates/wide)
crate, which selects the best available instruction set at compile time (SSE2, AVX2, NEON, …)
with a portable scalar fallback.

For maximum throughput on your own machine, enable `target-cpu=native`:

```sh
cargo build --profile full-opt
```
