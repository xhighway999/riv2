# reitero_residual

Residual coding for the [Reitero video format](https://github.com/xhighway999/riv).

Handles the encode/decode pipeline between motion-compensated prediction and the bitstream:
DCT transform, quantization, and RANS entropy coding.

**Most applications should use `reitero_encode` / `reitero_decode` instead.**
This crate is an implementation detail of those higher-level crates.

## Usage

```toml
[dependencies]
reitero_residual = "0.1"
```

### Quality → quantization step

```rust
use reitero_residual::quant_step_from_quality;

// Map a 1–100 quality value to a DCT quantization step.
let step: u16 = quant_step_from_quality(75);
```

### Encode intra residuals

```rust
use reitero_residual::{ResidualEncoder, IntraResidualEncodeResult};

let mut encoder = ResidualEncoder::new();
let result: IntraResidualEncodeResult = encoder.encode_intra(&yuv_frame, quality)?;
// result.data — RANS-compressed bitstream bytes
```

### Decode intra residuals

```rust
use reitero_residual::{ResidualDecoder, InterResidualDecodeParams};

let mut decoder = ResidualDecoder::new();
decoder.decode_intra(&compressed_bytes, &mut yuv_frame, quality)?;
```

## Main types

| Type | Description |
|---|---|
| `ResidualEncoder` | Encodes intra and inter residuals to RANS bitstream |
| `ResidualDecoder` | Decodes intra and inter residuals from RANS bitstream |
| `MvRansEncoder` / `MvRansDecoder` | RANS entropy coding for motion vectors |
| `MvPredictors` / `MvNeighborSet` | Motion vector prediction context |
| `quant_step_from_quality` | Map quality 1–100 to DCT quantization step |
