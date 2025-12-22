# ReItero `.riv` Video Format Spec (v1)

This document describes the current on-disk format for ReItero video files (`.riv`) and the exact encode/decode behavior implemented in the codebase.

- **Authoritative code**:
  - `reitero_video/reitero_video_common/src/format.rs` (binary layout)
  - `reitero_video/reitero_encode/src/encoder.rs` + `reitero_video/reitero_encode/src/motion.rs` (encoder)
  - `reitero_video/reitero_decode/src/decoder.rs` (decoder)
  - `reitero_video/reitero_residual/src/residual.rs` + `reitero_video/reitero_residual/src/dct.rs` (residual encoding/decoding)
  - `reitero_video/reitero_residual/src/mv_predictor.rs` + `reitero_video/reitero_residual/src/mv_rans.rs` (MV predictor + entropy coder)
- **Endianness**: all integer fields are **little-endian**.
- **Pixel convention**: buffers are treated as **RGB24** (3 bytes per pixel, packed, row-major).

---

## 1. Terminology

- **Display size**: original intended output resolution `(display_width × display_height)`.
- **Storage size**: padded resolution `(storage_width × storage_height)`, always ≥ display size, used for block processing.
- **Storage frame**: RGB24 buffer sized `storage_width * storage_height * 3`.
- **Block**: fixed **16×16** macroblock over the storage frame.
- **Reference frame**: previous **reconstructed** storage frame (what the decoder would have), used for prediction.
- **Residual**: difference between current and predicted frame, encoded in **YUV420** space using DCT.

---

## 2. File header layout (`VideoHeader`)

### 2.1 Binary layout (36 bytes)

| Offset | Size | Type | Name |
|---:|---:|---|---|
| 0  | 4  | bytes | `magic` = `RIV\0` |
| 4  | 4  | u32 | `version` = `1` |
| 8  | 4  | u32 | `display_width` |
| 12 | 4  | u32 | `display_height` |
| 16 | 4  | u32 | `storage_width` |
| 20 | 4  | u32 | `storage_height` |
| 24 | 4  | u32 | `fps` |
| 28 | 8  | u64 | `frame_count` |

### 2.2 Constraints / meaning

- `storage_width >= display_width`, `storage_height >= display_height`
- Storage dimensions are padded to a multiple of 16 by the encoder config.
- `frame_count` is written as 0 initially and patched at finalize.

---

## 3. Frame record layout (`PackedFrame`)

Frames are written sequentially after the header. Each record starts with:

- `timestamp_ms: u64` (little-endian)
- `frame_type: u8`
  - `1` = Intra (I)
  - `2` = Inter (P, motion-compensated)

### 3.1 Intra (I-frame) record

| Field | Type | Notes |
|---|---|---|
| `timestamp_ms` | u64 | |
| `frame_type` | u8 = 1 | |
| `jpeg_size` | u32 | byte length of `jpeg_rgb` |
| `jpeg_rgb` | bytes | JPEG payload that decodes to **RGB24 storage frame** |

The JPEG is produced from storage-sized RGB24 with `jpeg-encoder` using `ColorType::Rgb`, and decoded with `jpeg-decoder` expecting `PixelFormat::RGB24`.

### 3.2 Inter (P-frame) record (motion vectors + residual)

| Field | Type | Notes |
|---|---|---|
| `timestamp_ms` | u64 | |
| `frame_type` | u8 = 2 | |
| `quality` | u8 | Quality parameter (1-100) for residual DCT quantization |
| `global_mv` | 3 × i8 | Per-frame motion-delta bias packed as `(dx, dy, flags)`. `dx`/`dy` recenter motion deltas before RANS; `flags` is reserved (0 today). |
| `mv_size` | u32 | byte length of `mv_deflate` |
| `mv_deflate` | bytes | **RANS-compressed motion vector bitstream** (field name kept for historical reasons; no DEFLATE layer) |
| `res_size` | u32 | byte length of `residual_yuv420` |
| `residual_yuv420` | bytes | RANS-compressed residual data (quantized DCT coefficients in zigzag order) |

### 3.3 Byte-level record layouts (normative)

All offsets below are **relative to the start of the frame record** (i.e. the first byte of `timestamp_ms`).

#### 3.3.1 Intra (I) record bytes

| Offset | Size | Type | Name |
|---:|---:|---|---|
| 0  | 8 | u64 | `timestamp_ms` |
| 8  | 1 | u8  | `frame_type` (=1) |
| 9  | 4 | u32 | `jpeg_size` |
| 13 | `jpeg_size` | bytes | `jpeg_rgb` |

`jpeg_rgb` must decode to a **storage-sized** RGB24 buffer: `storage_width * storage_height * 3` bytes.

#### 3.3.2 Inter (P) record bytes

| Offset | Size | Type | Name |
|---:|---:|---|---|
| 0  | 8 | u64 | `timestamp_ms` |
| 8  | 1 | u8  | `frame_type` (=2) |
| 9  | 1 | u8  | `quality` |
| 10 | 1 | i8  | `global_mv.dx` |
| 11 | 1 | i8  | `global_mv.dy` |
| 12 | 1 | u8  | `global_mv.flags` (reserved, currently 0) |
| 13 | 4 | u32 | `mv_size` |
| 17 | `mv_size` | bytes | `mv_deflate` (RANS-compressed motion vectors) |
| 17+mv_size | 4 | u32 | `res_size` |
| 21+mv_size | `res_size` | bytes | `residual_yuv420` |

`global_mv` carries a frame-level delta bias that only applies to blocks coded in **NEW** mode. Whenever a block emits explicit deltas, the encoder subtracts `(dx, dy)` from those deltas before entropy coding, and the decoder adds it back after RANS decoding. This recenters the NEW-mode residuals around zero without affecting ZERO/NEAREST/NEAR predictors.

**Constraints / validation**:
- `quality` must be in range `[1, 100]` (1 = lowest quality/max quantization, 100 = highest quality/min quantization)
- `residual_yuv420` contains RANS-compressed quantized DCT coefficients (see §5)

---

## 4. Motion vector payload (`mv_deflate`)

### 4.1 Block grid

- `blocks_w = storage_width / 16`
- `blocks_h = storage_height / 16`
- Motion vector count = `blocks_w * blocks_h`

Storage dimensions are padded so they're multiples of 16; thus these divisions are exact.

### 4.2 Logical motion vector values (per block)

For each 16×16 block (in raster order), the logical motion vector is:

- `dx: i8` – absolute horizontal motion in **pixels** (range [-128, 127])
- `dy: i8` – absolute vertical motion in **pixels** (range [-128, 127])
- `flags: u8` – half-pixel offsets and skip flag:
  - Bits 0-1: half-pixel X code (0-3, see §4.3)
  - Bits 2-3: half-pixel Y code (0-3, see §4.3)
  - Bit 6: skip flag (1 = skip, 0 = no skip)
  - Bit 7: reserved (must be 0)

These values are **not** stored directly. Instead, the encoder converts every block into a compact `MvCodedBlock` structure prior to entropy coding:

| Field | Type | Meaning |
|---|---|---|
| `mode` | `MvMode` | One of `{Zero, Nearest, Near, New}` describing how the block reuses predictors |
| `new_uses_near` | bool | Only relevant when `mode == New`; selects `near` (`true`) vs `nearest` (`false`) as the base reference |
| `delta_x`, `delta_y` | i8 | Bias-centered residuals emitted only for `New` blocks |
| `flags` | u8 | Same flag byte described above |

`MvMode` follows VP9 nomenclature:

- `Zero`: force `(0, 0)` motion.
- `Nearest`: copy the strongest spatial predictor (see §4.4).
- `Near`: copy the second-best predictor.
- `Temporal`: copy the motion vector from the same block in the previous frame.
- `New`: emit explicit deltas relative to either `nearest` or `near` (selected via `new_uses_near`).

The remainder of this section explains how the encoder derives the predictors, decides on a mode per block, and serializes the results into the RANS bitstream so the decoder can reconstruct the original `(dx, dy, flags)` tuples.

### 4.3 Half-pixel precision

Half-pixel code mapping (per axis):
- `0` = `0.0` pixels (integer-aligned)
- `1` = `+0.5` pixels
- `2` = `-0.5` pixels
- `3` = reserved (currently treated as `0.0`)

Half-pixel sampling uses bilinear interpolation with the appropriate fractional offset (see `sample_rgb_halfpel` in `reitero_video_common::motion`).

### 4.4 Prediction modes and NEW-mode deltas

Motion prediction now mirrors VP9's four-mode tree:

- For every block the encoder derives three candidate vectors `(nearest, near, temporal)`:
  - `nearest` and `near` are derived by scanning already-coded neighbors in this order: left, top, top-right, top-left. Duplicates are removed and the list is padded with `(0, 0)` as needed.
  - `temporal` is taken from the co-located block in the previous frame (if available).
- Based on those predictors the encoder assigns one of five modes:
  - **Zero** – force `(0, 0)` regardless of candidates.
  - **Nearest** – copy the first candidate verbatim.
  - **Near** – copy the second candidate verbatim.
  - **Temporal** – copy the temporal candidate verbatim.
  - **New** – start from whichever candidate has the lower rate-distortion score (selected via `new_uses_near`) and emit residual deltas.
- Only `New` blocks carry explicit integer deltas. For those blocks the raw delta is `raw_delta = (actual - base)`, clamped to `[-128, 127]` per axis. A frame-level bias `(bias_dx, bias_dy)`—stored in `global_mv`—is subtracted from every NEW delta before entropy coding so that the RANS model sees residuals tightly clustered around zero. ZERO/NEAREST/NEAR/TEMPORAL modes never write deltas, so the bias has no effect on them.
- The `flags` byte (half-pel offsets + skip bit) bypasses the predictor entirely. Bit 6 is updated after residual encoding so it always reflects the optimized skip mask that both encoder and decoder obey.

This mode information travels through `MvCodedBlock` so that decoder and tools can faithfully reconstruct the original absolute motion vectors even though only a subset of them carry explicit deltas.

### 4.5 RANS bitstream format (`mv_deflate`)

`mv_deflate` remains a per-frame RANS32 stream, but its symbol order now reflects the structured blocks above.

- The encoder instantiates a fresh `RansWriter32` for each frame, appends symbols exactly as described below, then `finish()`es the writer and copies the resulting bytes into the frame record. No extra headers or padding are inserted.
- Probability contexts are VP8-style and persist across inter frames. They reset whenever an intra frame appears so that decoding may restart from that key frame alone.
- Context inventory:
  - `skip_ctx` – skip flag
  - `mode_is_new_ctx[ctx]`, `mode_is_nonzero_ctx[ctx]`, `mode_is_near_ctx[ctx]`, `mode_is_temporal_ctx[ctx]` – four binary splits forming the mode tree (ctx depends on left/top modes; see `mv_mode_context` in code)
  - `new_ref_ctx[ctx]` – selects the base reference (`nearest` vs `near`) for NEW blocks
  - **Candidate history contexts** (for NEW blocks):
    - `candidate_hit_ctx[ctx]` – boolean indicating if the delta matches a frequent candidate
    - `candidate_idx_ctx[depth]` – binary tree for candidate index
  - Per-axis magnitude coding:
    - `class_ctx_x/y` – binary tree over magnitude classes (0‥10)
    - `sign_ctx_x/y` – sign bit when class > 0
    - `mag_bit_ctx_x/y[0..MV_MAX_MAG_BITS)` – raw magnitude bits for the residual offset within a class
  - Fractional contexts: `sub_x_has_ctx`, `sub_x_sign_ctx`, `sub_y_has_ctx`, `sub_y_sign_ctx`

Encoding order per block (raster scan):

1. **Skip flag** – emit `(flags & 0x40) != 0` via `skip_ctx`.
2. **Mode tree** – encode booleans using the context indexed by neighboring modes:
   1. `is_new` → `mode_is_new_ctx`
   2. If `false`, `is_nonzero` (distinguishes ZERO vs the other modes) → `mode_is_nonzero_ctx`
   3. If `is_nonzero` and `!is_new`, `is_near` → `mode_is_near_ctx`
   4. If `!is_near` and `is_nonzero` and `!is_new`, `is_temporal` → `mode_is_temporal_ctx`
3. **NEW reference selector** – if the block is `New`, emit `new_uses_near` via `new_ref_ctx[ctx]`.
4. **NEW deltas** – still only present for `New` blocks:
   - **Candidate history check**:
     - The encoder maintains a list of frequent `(delta_x, delta_y, frac_flags)` tuples.
     - Emit `is_candidate_hit` via `candidate_hit_ctx`.
     - If hit: emit candidate index via `candidate_idx_ctx`. The fractional flags are also recovered from the candidate.
     - If miss: encode `delta_x` and `delta_y` explicitly as below, then encode fractional flags.
   - **Explicit encoding** (if miss):
     - Let `delta = (delta_x, delta_y)` be the bias-centered residual pair. Each component is encoded independently.
     - Compute the magnitude class (`0`..`MV_MAX_CLASS`) from `abs(delta)` and emit it through the binary class tree using `class_ctx_*`.
     - If the class is `0`, the component is zero and no further symbols are written.
     - Otherwise emit the sign bit via `sign_ctx_*`, then the remaining magnitude bits (`mv_class_bits(class)`) via the corresponding `mag_bit_ctx_*` slots. This reproduces the VP9 "class + bits" coding scheme.
5. **Fractional offsets** – if not recovered from candidate history:
   - Encode `has_frac_x`/`sign_x`, then `has_frac_y`/`sign_y` using the fractional contexts listed above. These symbols encode flag bits 0-3.

The decoder mirrors the same steps and yields a `Vec<MvCodedBlock>` instead of a raw `[ddx, ddy, flags]` byte stream. Each entry contains the mode, optional NEW deltas (still bias-centered), the preserved `new_uses_near` selector, and the original flag byte.

### 4.6 Reconstructing absolute motion vectors and skip mask

Given:

- `mv_blocks` from `MvRansDecoder::decode_frame(blocks_w, blocks_h)`
- Block grid dimensions
- Frame-level bias `global_mv = (bias_dx, bias_dy, flags)`

Process every block in raster order:

1. Derive the `nearest`/`near`/`temporal` predictor triplet from neighbors that have already been reconstructed, plus the temporal reference if available (same logic as §4.4).
2. Determine the integer base vector:
   - `Zero` → `(0, 0)`
   - `Nearest` → `predictors.nearest`
   - `Near` → `predictors.near`
   - `Temporal` → `predictors.temporal`
   - `New` → `predictors.near` if `new_uses_near` is `true`, else `predictors.nearest`
3. Apply NEW deltas:
   - For ZERO/NEAREST/NEAR/TEMPORAL the deltas are implicitly zero.
   - For NEW, first remove the frame bias: `delta = ((block.delta_x as i16 + bias_dx) , (block.delta_y as i16 + bias_dy))`, clamped to `[-128, 127]` per axis.
   - Add the delta to the chosen base and clamp again to produce `(dx, dy)`.
4. Assemble the logical `MotionVector { dx, dy, flags }` and append it to the block list. The skip mask for residual decoding is still derived from `flags & 0x40`.

Because both encoder and decoder run the same predictor derivation and mode tree, the absolute motion field and authoritative skip mask are reproduced exactly even though only NEW blocks transmit explicit residuals.

### 4.7 Skip mode (authoritative semantics)

A block is in **skip mode** if bit 6 of the flags byte is set (`flags & 0x40 != 0`).

In skip mode:

- The decoder **applies no residual** to that block (i.e. `current_block = predicted_block`).
- Motion vector components `(dx, dy, qx, qy)` are still fully decoded and used for prediction.
- `skip_mask[block]` is built from flags during MV decoding and is treated as **authoritative**:
  - The encoder may additionally mark blocks as skipped if their residual quantizes to all zeros.
  - In that case it updates the MV flags bit 6 to reflect the optimized skip mask.

---

## 5. Residual data payload (`residual_yuv420`)

### 5.1 Overview

The residual data encodes the difference between the current frame and the motion-compensated predicted frame. The residual is computed and stored in **YUV420** color space, not RGB.

### 5.2 Residual computation

1. Convert current and predicted frames from RGB24 to YUV420 planar
2. Compute per-plane residual: `residual = current_yuv - predicted_yuv` (signed i16, range [-255, 255])
3. For each motion block (16×16 pixels):
   - Y plane: 16×16 block
   - U plane: 8×8 block (YUV420 halves both dimensions)
   - V plane: 8×8 block

### 5.3 DCT encoding

For each block (Y: 16×16, U/V: 8×8):

1. **2D DCT**: Apply 2D Discrete Cosine Transform to the residual block
2. **Quantization**: Quantize DCT coefficients using uniform quantization step based on `quality`:
   - Quality 100: `quant_step = 1.0` (minimal quantization)
   - Quality 1: `quant_step = 1000.0` (maximum quantization)
   - Linear interpolation for values in between: `quant_step = 1.0 + (100 - quality) / 99.0 * 999.0`
   - Quantized coefficient: `round(dct_coeff / quant_step)` (clamped to i16 range: -32768 to 32767)
3. **Zigzag scan**: Reorder quantized coefficients from 2D (row-major) to 1D zigzag order

### 5.4 RANS encoding

After zigzag scanning, quantized DCT coefficients are encoded using **RANS (Range Asymmetric Numeral Systems)** entropy coding:

- **RANS encoding format**: Each block (Y, U, or V) is encoded as a bitstream using RANS32 with VP8-style context modeling
- **Block encoding process** (per plane):
  1. Find the last non-zero coefficient index (EOB - End of Block)
  2. Encode EOB index using unary encoding with a shared EOB context
  3. For each coefficient from index 0 to EOB (inclusive):
     - **Significance bit**: Encode whether the coefficient is non-zero using a position-specific context (separate contexts for luma and chroma)
     - If non-zero:
       - **Sign bit**: Encode sign (negative/positive) using bypass coding (statistically 50/50)
       - **Magnitude**: Encode `abs(value) - 1` using unary encoding with a shared magnitude context (separate for luma and chroma)
- **Context separation**:
  - Luma (Y) blocks: 256 position-specific contexts (one per coefficient position in 16×16 block)
  - Chroma (U/V) blocks: 64 position-specific contexts (one per coefficient position in 8×8 block)
  - Separate magnitude contexts for luma and chroma
  - Shared EOB context across all blocks
- **Block layout**: For each motion block in raster order:
  - Y block (256 coefficients): RANS-encoded bitstream
  - U block (64 coefficients): RANS-encoded bitstream
  - V block (64 coefficients): RANS-encoded bitstream
- **Skipped blocks**: If a block is marked as skip (see §4.6), no RANS data is generated for that block (all three blocks Y, U, V are skipped)
- **Optimized skip mask**: Blocks that quantize to all zeros after DCT quantization are also marked as skipped (no RANS data generated), even if they weren't originally marked as skip candidates. The encoder updates the skip flag in the motion vector flags byte to reflect this optimized skip mask

#### Bitstream layout

- Residual coefficients for the entire frame share **one** `RansWriter32` output buffer. There is no per-block header inside the stream; decoders must follow the exact symbol order below.
- For each macroblock in raster order:
  1. If the block is skipped, no symbols are emitted for Y/U/V and the decoder implicitly fills the residual with zeros.
  2. Otherwise, emit plane data in Y, then U, then V order. Each plane performs:
     - Unary EOB code: emit `k` zero bits followed by a one bit, where `k` is the number of coefficients after the last non-zero entry (clamped to plane length - 1). Luma uses a dedicated EOB context, chroma planes use their own.
     - For every coefficient index from 0 through the reported EOB (inclusive):
       * **Significance bit**: boolean stating whether the coefficient is non-zero. Context index equals the coefficient index (0-255 for Y, 0-63 for U/V). If `false`, proceed to next coefficient.
       * **Sign bit** (if significant): bypass-coded boolean (0 = positive, 1 = negative).
       * **Magnitude**: unary-coded absolute value minus one using the shared magnitude context family for the plane type.
        * **Significance bit**: boolean stating whether the coefficient is non-zero. Context index equals the coefficient index (0-255 for Y, 0-63 for U/V). If `false`, proceed to next coefficient.
        * **Sign bit** (if significant): bypass-coded boolean (0 = positive, 1 = negative).
        * **Magnitude**: unary-coded absolute value minus one using the shared magnitude context family for the plane type.
- All unary codes follow the same convention as the motion-vector stream: emit zero bits while decrementing the counter, then a terminating one. The CABAC-style contexts adapt over the entire video; there is no reset between frames.

### 5.5 Storage format

The `residual_yuv420` field contains RANS-compressed bytes directly (no additional DEFLATE compression):

- Format: **RANS-encoded bitstream** (RANS32 with VP8 contexts)
- Layout: For each motion block in raster order (only non-skipped blocks):
  - Y block: RANS bitstream encoding 256 coefficients (zigzag order)
  - U block: RANS bitstream encoding 64 coefficients (zigzag order)
  - V block: RANS bitstream encoding 64 coefficients (zigzag order)
- **Skipped blocks**: No RANS data is written for skipped blocks (decoder uses skip_mask to determine which blocks to skip)
- The RANS encoder accumulates all blocks into a single bitstream, which is finalized when encoding is complete. The resulting bytes are copied verbatim into `residual_yuv420`.

### 5.6 Residual decoding

1. Create RANS decoder and consume `residual_yuv420` bytes directly (no DEFLATE decompression)
2. For each motion block in raster order:
   - If block is skipped (from skip_mask): skip all three blocks (Y, U, V), residual remains zero
   - Otherwise:
     - Decode Y block: decode RANS bitstream → 256 coefficients (zigzag order, padded with zeros if EOB < 255)
     - Decode U block: decode RANS bitstream → 64 coefficients (zigzag order, padded with zeros if EOB < 63)
     - Decode V block: decode RANS bitstream → 64 coefficients (zigzag order, padded with zeros if EOB < 63)
     - For each block (Y, U, V):
       - **Decode RANS**: Decode EOB index, then decode coefficients from 0 to EOB using significance bits, sign bits, and magnitude
       - **Reverse zigzag**: Convert 1D zigzag order back to 2D (row-major)
       - **Dequantization**: `dct_coeff = quantized * quant_step`
       - **Inverse 2D DCT**: Reconstruct residual block
3. Apply residuals to predicted YUV420 planes: `recon = predicted + residual` (clamped to [0, 255])
4. Convert reconstructed YUV420 back to RGB24

---

## 6. Padding and cropping rules

### 6.1 Encoder padding (display → storage)

Input frames arrive as display-sized RGB24.

The encoder pads to storage size via edge replication:

- **Right pad**: repeat the last pixel in each row.
- **Bottom pad**: repeat the last (already padded) row.

The intra JPEG operates on the **storage-sized** image.
The inter residual operates on the **storage-sized** YUV420 planes.

### 6.2 Decoder cropping (storage → display)

Decoder reconstructs the full storage frame, then outputs only the top-left `display_width × display_height` region.

---

## 7. Encoder algorithm (normative)

The encoder maintains:

- `prev_recon_rgb`: previous reconstructed storage frame (RGB24), i.e. what the decoder would have.

### 7.1 I-frame

1. Pad current display frame → `curr_storage_rgb`.
2. JPEG-encode `curr_storage_rgb` with `intra_quality`.
3. JPEG-decode that payload to `recon_storage_rgb`.
4. Set `prev_recon_rgb = recon_storage_rgb`.
5. Write an Intra record.

### 7.2 P-frame (motion-compensated inter)

Given `prev_recon_rgb` and `curr_storage_rgb`:

1. **Motion estimation (diamond search)** per 16×16 block:
   - **Motion vector prediction**: Start with prediction from neighboring blocks:
     - Try left block MV (if available and within search range)
     - Try top block MV (if available and within search range, preferred over left)
     - Try median predictor: median of (left MV, top MV, zero MV) if both neighbors available
     - Convert half-pixel MVs to integer pixels by rounding for prediction
   - **Integer search** using **SATD (Sum of Absolute Transformed Differences)** with Hadamard transform:
     - SATD is computed using 4×4 Hadamard transform on luminance differences (Y = 0.299*R + 0.587*G + 0.114*B)
     - Process 16×16 block as 4×4 sub-blocks, sum SATD across all sub-blocks
     - Multi-scale diamond search: start with step size = `search_range`, halve step size until step=1
     - At each step size, check diamond pattern (cardinal + diagonal directions) until no improvement
     - Always check zero motion (0,0) early as it's very common
     - Early termination: stop checking candidates if SATD exceeds current best
   - Search offsets `(dx,dy)` are constrained to `[-search_range, +search_range]` pixels
2. **Half-pixel refinement**:
   - Around the best integer `(dx,dy)`, evaluate all 9 combinations where each axis adds `{-0.5, 0, +0.5}`.
   - Use **RGB SAD** (Sum of Absolute Differences) with half-pixel sampling (bilinear interpolation).
   - Half-pixel sampling uses bilinear interpolation:
     - Horizontal only: average of left and right pixels
     - Vertical only: average of top and bottom pixels  
     - Both: average of all 4 corner pixels
   - Convert best half-pixel motion to flag layout:
     - Integer pixel component: `dx_px = dx_hp / 2`, `dy_px = dy_hp / 2`
     - Half-pixel code: map half-pixel fraction (`-1, 0, +1` half units) to bit codes:
       - `-0.5px` → code `2`
       - `0px` → code `0`
       - `+0.5px` → code `1`
3. Build `predicted_storage_rgb` by sampling `prev_recon_rgb` using motion vectors with half-pixel precision (bilinear interpolation with fractional offsets, see §4.3).
4. **Skip decision** (per block):
   - If `skip_threshold == 0`: no blocks skipped
   - Otherwise: block is candidate for skip if:
    - MV is integer-aligned (both sub-pixel codes are zero)
    - Block SAD (from half-pixel refinement) ≤ `skip_threshold * (16 * 16 * 3)`
    - Note: SAD score is reused from the half-pixel refinement step to avoid extra computation
   - Final skip decision: candidate skip flag is set in the flags byte (bit 6)
   - The encoder also checks if blocks quantize to all zeros after DCT encoding, and marks those as skipped in the optimized skip mask (updates flags byte bit 6)
5. **Residual encoding**:
   - Convert `curr_storage_rgb` and `predicted_storage_rgb` to YUV420 planar
   - Compute residual planes: `residual = current_yuv - predicted_yuv` (signed i16)
   - Create RANS encoder
   - For each motion block:
     - If skipped (from skip_mask): no RANS data generated, mark as optimized skip
     - Otherwise:
       - Encode Y block (16×16 DCT → quantization → zigzag → RANS encoding)
       - Encode U block (8×8 DCT → quantization → zigzag → RANS encoding)
       - Encode V block (8×8 DCT → quantization → zigzag → RANS encoding)
       - Check if all coefficients quantize to zero: if so, mark as optimized skip (no RANS data generated, update flags byte bit 6)
   - Finish RANS encoding to get final compressed bytes
6. **Motion vector encoding**:
   - For each block in raster order:
     - Derive predictors `(nearest, near, temporal)` from neighbors (see §4.4).
     - Determine the best mode (`Zero`, `Nearest`, `Near`, `Temporal`, or `New`) based on the chosen MV.
     - If `New`, compute deltas relative to the chosen base (`nearest` or `near`).
     - Construct `MvCodedBlock` with mode, deltas, and flags.
   - After residual encoding, update bit 6 of each `flags` byte to match the **optimized skip mask** (so the skip bit is authoritative).
   - Pass the list of `MvCodedBlock`s to `MvRansEncoder::encode_frame_and_get_data`, producing a per-frame RANS32 bitstream stored in `mv_deflate` (no DEFLATE, despite the field name).
7. **Residual data**: RANS-compressed residual bytes are used directly (no additional DEFLATE compression)
8. **Reconstruct reference** (to match decoder):
   - Decode residual DCT coefficients (reverse zigzag, dequantize, inverse DCT)
   - Apply residuals to predicted YUV420: `recon_yuv = predicted_yuv + residual`
   - Convert reconstructed YUV420 back to RGB24
   - Set `prev_recon_rgb = recon_rgb`
9. Write an Inter record containing `(quality, mv_deflate, residual_yuv420)`.

---

## 8. Decoder algorithm (normative)

Decoder maintains:

- `prev_recon_rgb`: previous reconstructed storage frame (RGB24).

### 8.1 I-frame

1. Decode `jpeg_rgb` to `curr_storage_rgb` (must be storage-sized).
2. Set `prev_recon_rgb = curr_storage_rgb`.
3. Crop to display size and output.

### 8.2 P-frame

1. Compute `blocks_w = storage_width/16`, `blocks_h = storage_height/16`, `total_blocks = blocks_w*blocks_h`.
2. Create `MvRansDecoder`, call `consume_frame(mv_deflate)`, then `decode_frame(blocks_w, blocks_h)`:
   - This yields a list of `MvCodedBlock` structures.
   - Reconstruct absolute MVs and `skip_mask[block]` from these blocks using the predictor logic (§4.4/§4.6).
4. Build `predicted_storage_rgb` from `prev_recon_rgb` using absolute MVs with half-pixel precision:
   - for each output pixel `(x,y)` in the block:
    - sample reference at `(x + dx + sub_x_offset, y + dy + sub_y_offset)` using bilinear interpolation with half-pixel offsets
     - clamp sampling coords to the valid frame bounds
5. Create RANS decoder and consume `residual_yuv420` bytes directly (no DEFLATE decompression)
6. Decode residual DCT coefficients:
   - For each motion block in raster order:
     - If `skip_mask[block]`: skip all three blocks (Y, U, V), residual remains zero
     - Otherwise:
       - Decode Y block: decode RANS bitstream → 256 coefficients (zigzag order, padded with zeros if EOB < 255)
       - Decode U block: decode RANS bitstream → 64 coefficients (zigzag order, padded with zeros if EOB < 63)
       - Decode V block: decode RANS bitstream → 64 coefficients (zigzag order, padded with zeros if EOB < 63)
       - For each block (Y, U, V):
         - Decode RANS: decode EOB index, then decode coefficients from 0 to EOB using significance bits, sign bits, and magnitude
         - Reverse zigzag scan
         - Dequantize using `quality` parameter
         - Inverse 2D DCT
         - Write to residual YUV420 planes
8. Apply residuals:
   - Convert `predicted_storage_rgb` to YUV420 planar
   - Apply residual planes: `recon_yuv = predicted_yuv + residual` (clamped to [0, 255])
   - Convert reconstructed YUV420 back to RGB24: `curr_storage_rgb`
9. Set `prev_recon_rgb = curr_storage_rgb`.
10. Crop to display size and output.

---

## 9. CLI knobs

`ri-cli encode` exposes:

- `--intra-quality <1..=100>`: JPEG quality for I-frames (default: 90)
- `--inter-quality <1..=100>`: Quality parameter for residual DCT quantization in P-frames (default: 35)
- `--search-range <0..=31>`: integer motion search radius (±N pixels); must fit in signed 6‑bit int part (default: 12)
- `--skip-threshold <0..=255>`: per-block SAD/SATD-based skip threshold (average abs diff per byte) (default: 3)
- `--max-frames <u64>`: encode at most this many frames (0 = no limit)

`ri-cli decode` exposes:

- `--skip-residuals`: if set, decodes motion vectors only and outputs motion‑predicted frames (no residuals applied)

Additional subcommands:

- `extract-frame`: extract a single decoded frame from a `.riv` file into a directory on disk
- `roundtrip`: encode to `.riv` in memory and immediately re‑encode to a conventional video file for visual inspection

---

## 10. Notes / caveats

- This is an early format; it is **not designed for backward compatibility** yet.
- Motion search cost grows quickly with `search_range`; keeping it small is recommended.
- **Motion search uses SATD (Sum of Absolute Transformed Differences)** with Hadamard transform for integer-pixel search, which is more accurate than SAD for motion estimation. Half-pixel refinement uses RGB SAD for computational efficiency.
- Motion vector prediction uses a **VP9-style mode tree** (`Zero`, `Nearest`, `Near`, `Temporal`, `New`) to select the best predictor from spatial and temporal neighbors.
- Motion vectors use **half-pixel precision** with bilinear interpolation for sub-pixel sampling.
- JPEG payloads (I-frames) are treated as **JPEG-encoded RGB24** at the API boundary (even if codecs internally use YCbCr).
- Residual encoding uses **YUV420** color space with DCT, not RGB JPEG.
- Quality parameter (1-100) controls DCT quantization step size, where 1 = maximum quantization (lowest quality) and 100 = minimal quantization (highest quality).
- Residual data uses **RANS (Range Asymmetric Numeral Systems)** entropy coding with VP8-style context modeling, which provides efficient compression for sparse DCT coefficient blocks (most coefficients are zero after quantization).
- RANS encoding uses separate contexts for luma (Y) and chroma (U/V) blocks, with position-specific contexts for significance bits and shared contexts for magnitude and EOB encoding.
- Skipped blocks generate no RANS data (decoder uses skip_mask to determine which blocks to skip).
- DCT coefficient values are stored as i16 (range -32768 to 32767) after quantization.
- Motion vectors are stored using a **RANS-encoded stream of `MvCodedBlock` structures**, which efficiently encodes modes, candidate indices (for repeated new vectors), and residual deltas.

