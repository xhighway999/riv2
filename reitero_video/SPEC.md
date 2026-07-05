# ReItero `.riv` Video Format Spec (v5)

This document describes the current on-disk format for ReItero video files (`.riv`) and the exact encode/decode behavior implemented in the codebase.

- **Authoritative code**:
  - `reitero_video/reitero_video_common/src/format.rs` (binary layout)
  - `reitero_video/reitero_video_common/src/motion_vector.rs` (motion vector struct)
  - `reitero_video/reitero_video_common/src/fast_motion.rs` (motion-compensated prediction)
  - `reitero_video/reitero_video_common/src/deblock.rs` (in-loop deblocking filter, v5+)
  - `reitero_video/reitero_encode/src/encoder.rs` + `reitero_video/reitero_encode/src/motion.rs` (encoder)
  - `reitero_video/reitero_decode/src/decoder.rs` (decoder)
  - `reitero_video/reitero_residual/src/residual.rs` + `reitero_video/reitero_residual/src/rans.rs` (residual encoding/decoding)
  - `reitero_video/reitero_residual/src/mv_predictor.rs` + `reitero_video/reitero_residual/src/mv_rans.rs` (MV predictor + entropy coder)
  - `reitero_video/reitero_dct/` (DCT transforms: 8×8 and 16×16)
- **Endianness**: all integer fields are **little-endian**.
- **Internal color space**: all prediction, residual, and reconstruction operations use **YUV420** planar. RGB24 is only the input/output format.

---

## 1. Terminology

- **Display size**: original intended output resolution `(display_width × display_height)`.
- **Storage size**: padded resolution `(storage_width × storage_height)`, always ≥ display size, used for block processing.
- **Storage frame**: YUV420 planar buffer at storage dimensions.
- **Block**: fixed **16×16** macroblock over the storage frame.
- **Reference frame**: previous **reconstructed** storage frame as YUV420 (what the decoder would have), used for prediction.
- **Residual**: difference between current and predicted frame, computed and encoded in **YUV420** space using DCT.

---

## 2. File header layout (`VideoHeader`)

### 2.1 Binary layout (36 bytes)

| Offset | Size | Type | Name |
|---:|---:|---|---|
| 0  | 4  | bytes | `magic` = `RIV\0` |
| 4  | 4  | u32 | `version` = `5` |
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

I‑frames encode the full frame as a **YUV420 residual** against an implicit mid-gray reference (128 per channel). DC prediction is enabled for intra RANS coding, and JPEG-style perceptual quantization matrices are used instead of uniform quantization.

| Field | Type | Notes |
|---|---|---|
| `timestamp_ms` | u64 | |
| `frame_type` | u8 = 1 | |
| `quality` | u8 | Quality parameter (1-100) for intra DCT quantization |
| `res_size` | u32 | byte length of `residual_yuv420` |
| `residual_yuv420` | bytes | RANS-compressed residual data (quantized DCT coefficients in zigzag order, with DC prediction) |

### 3.2 Inter (P-frame) record (motion vectors + residual)

| Field | Type | Notes |
|---|---|---|
| `timestamp_ms` | u64 | |
| `frame_type` | u8 = 2 | |
| `quality` | u8 | Quality parameter (1-100) for residual DCT quantization |
| `global_mv` | 3 × i8 | Per-frame motion-delta bias packed as `(dx, dy, flags)`. `dx`/`dy` recenter motion deltas before RANS; `flags` encodes half-pixel offsets (bits 0-3) and skip (bit 6). |
| `mv_size` | u32 | byte length of `mv_deflate` |
| `mv_deflate` | bytes | **RANS-compressed motion vector bitstream** (field name kept for historical reasons; no DEFLATE layer) |
| `res_size` | u32 | byte length of `residual_yuv420` |
| `residual_yuv420` | bytes | RANS-compressed residual data (quantized DCT coefficients in zigzag order, no DC prediction) |

### 3.3 Byte-level record layouts (normative)

All offsets below are **relative to the start of the frame record** (i.e. the first byte of `timestamp_ms`).

#### 3.3.1 Intra (I) record bytes

| Offset | Size | Type | Name |
|---:|---:|---|---|
| 0  | 8 | u64 | `timestamp_ms` |
| 8  | 1 | u8  | `frame_type` (=1) |
| 9  | 1 | u8  | `quality` |
| 10 | 4 | u32 | `res_size` |
| 14 | `res_size` | bytes | `residual_yuv420` |

#### 3.3.2 Inter (P) record bytes

| Offset | Size | Type | Name |
|---:|---:|---|---|
| 0  | 8 | u64 | `timestamp_ms` |
| 8  | 1 | u8  | `frame_type` (=2) |
| 9  | 1 | u8  | `quality` |
| 10 | 1 | i8  | `global_mv.dx` |
| 11 | 1 | i8  | `global_mv.dy` |
| 12 | 1 | u8  | `global_mv.flags` |
| 13 | 4 | u32 | `mv_size` |
| 17 | `mv_size` | bytes | `mv_deflate` (RANS-compressed motion vectors) |
| 17+mv_size | 4 | u32 | `res_size` |
| 21+mv_size | `res_size` | bytes | `residual_yuv420` |

`global_mv` carries a frame-level delta bias that only applies to blocks coded in **NEW** mode. Whenever a block emits explicit deltas, the encoder subtracts `(dx, dy)` from those deltas before entropy coding, and the decoder adds it back after RANS decoding. This recenters the NEW-mode residuals around zero without affecting ZERO/NEAREST/NEAR/TOPRIGHT/TOPLEFT/TEMPORAL predictors.

**Bias computation**: the encoder collects all raw NEW-mode deltas for the frame and searches over integer bias candidates in `[−24, 24]`. For each axis independently, it picks the bias value that (1) maximizes the count of NEW deltas that become zero after subtraction, breaking ties by (2) minimizing total L1 cost `Σ|delta − bias|` across all NEW blocks. X and Y axes are optimized independently. If no NEW blocks exist, the bias is `(0, 0)`.

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
- `subpixel_x: i8` – half-pixel X offset (`-1` = −0.5px, `0` = integer, `+1` = +0.5px)
- `subpixel_y: i8` – half-pixel Y offset (`-1` = −0.5px, `0` = integer, `+1` = +0.5px)
- `skip: bool` – skip flag

These values are **not** stored directly. Instead, the encoder converts every block into a compact `MvCodedBlock` structure prior to entropy coding:

| Field | Type | Meaning |
|---|---|---|
| `mode` | `MvMode` | One of `{Zero, Nearest, Near, TopRight, TopLeft, Temporal, New}` |
| `new_base` | u8 | Only for `New`; selects base predictor: 0=nearest, 1=near, 2=top-right, 3=top-left, 4=temporal |
| `delta_x`, `delta_y` | i8 | Bias-centered residuals emitted only for `New` blocks |
| `subpel_x`, `subpel_y` | Subpel | Half-pixel offsets (`Zero`, `PlusHalf`, `MinusHalf`) |
| `skip` | bool | Skip flag |

`MvMode` follows VP9 nomenclature with extensions:

- `Zero` (0): force `(0, 0, 0, 0)` motion.
- `Nearest` (1): copy the strongest spatial predictor (see §4.4).
- `Near` (2): copy the second-best predictor.
- `TopRight` (3): copy the top-right neighbor directly (falls back to `nearest` if unavailable).
- `TopLeft` (4): copy the top-left neighbor directly (falls back to `nearest` if unavailable).
- `New` (5): emit explicit deltas relative to a chosen base predictor (selected via `new_base`).
- `Temporal` (6): copy the motion vector from the same block in the previous frame.

### 4.3 Half-pixel precision

Half-pixel offsets are stored as signed values per axis:
- `0` = `0.0` pixels (integer-aligned)
- `+1` = `+0.5` pixels
- `-1` = `-0.5` pixels

Half-pixel sampling uses bilinear interpolation with the appropriate fractional offset on each YUV420 plane independently (see `build_predicted` in `reitero_video_common::fast_motion`).

### 4.4 Prediction modes and NEW-mode deltas

Motion prediction uses a VP9-style neighbor scan:

- For every block the encoder derives predictor candidates from already-coded neighbors:
  - **Spatial scan order**: left, top, top-right, top-left. Duplicates are removed and the list is padded with `(0, 0, 0, 0)` as needed.
  - `nearest` = first unique candidate (includes `(dx, dy, sub_x, sub_y)` — all four components).
  - `near` = second unique candidate.
  - `temporal` = co-located block from the previous frame's MV list (if available).
  - `top_right` and `top_left` = raw neighbor vectors (not deduped, used directly by their respective modes).
- Based on those predictors the encoder assigns one of seven modes:
  - **Zero** – force `(0, 0, 0, 0)` regardless of candidates.
  - **Nearest** – copy `predictors.nearest` verbatim (all four components).
  - **Near** – copy `predictors.near` verbatim.
  - **TopRight** – copy the top-right neighbor directly.
  - **TopLeft** – copy the top-left neighbor directly.
  - **Temporal** – copy `predictors.temporal` verbatim.
  - **New** – start from a chosen base predictor (selected via `new_base`) and emit residual deltas for the integer components. Subpixel offsets are encoded separately. **Encoder base selection**: the encoder uses an approximate bit-cost model instead of full RANS simulation. For each available base predictor it computes `cost = selector_bits(base_index) + delta_bits(delta_x, delta_y)`. `selector_bits` mirrors the binary tree depth: indices {0,1,2,3,4} → {1,2,3,4,4} bits. `delta_bits(dx, dy) = component_bits(dx) + component_bits(dy)` where `component_bits(d) = 4` if d=0, else `4 + class(|d|)` (4 class-tree bits + sign + `class−1` magnitude bits, matching the RANS coder in §4.5). The base with minimum cost is selected.
- Only `New` blocks carry explicit integer deltas. For those blocks the raw delta is `raw_delta = (actual - base)`, clamped to `[-128, 127]` per axis. A frame-level bias `(bias_dx, bias_dy)`—stored in `global_mv`—is subtracted from every NEW delta before entropy coding so that the RANS model sees residuals tightly clustered around zero. Other modes never write deltas, so the bias has no effect on them.
- The skip flag is encoded separately in the RANS stream and is authoritative for residual decoding.

### 4.5 RANS bitstream format (`mv_deflate`)

`mv_deflate` is a per-frame RANS32 stream.

- The encoder instantiates a fresh `RansWriter32` for each frame, appends symbols exactly as described below, then `finish()`es the writer and copies the resulting bytes into the frame record. No extra headers or padding are inserted.
- Probability contexts are VP8-style and persist across inter frames. They reset whenever an intra frame appears so that decoding may restart from that key frame alone.
- Context inventory:
  - `skip_ctx` – skip flag (single context)
  - Mode tree contexts (each array has `MODE_CTXS = 4` entries, indexed by `mv_mode_context`):
    - `mode_is_new_ctx[ctx]` – is the mode New?
    - `mode_is_nonzero_ctx[ctx]` – distinguishes Zero from other modes
    - `mode_is_near_ctx[ctx]` – is the mode Near?
    - `mode_is_temporal_ctx[ctx]` – is the mode Temporal?
    - `mode_is_top_right_ctx[ctx]` – is the mode TopRight?
    - `mode_is_top_left_ctx[ctx]` – is the mode TopLeft? (fallback = Nearest)
  - NEW base selector contexts (each `MODE_CTXS = 4` entries):
    - `new_base_is_nearest_ctx[ctx]` – is base = nearest?
    - `new_base_is_near_ctx[ctx]` – is base = near?
    - `new_base_is_top_right_ctx[ctx]` – is base = top-right?
    - `new_base_is_top_left_ctx[ctx]` – is base = top-left? (fallback = temporal)
  - **Candidate history contexts** (for NEW blocks):
    - `candidate_hit_ctx[ctx]` – 3 contexts indexed by `(left_hit as usize) + (top_hit as usize)`
    - `candidate_idx_ctx[depth]` – binary tree for candidate index (`CANDIDATE_IDX_DEPTH = 4`)
  - Per-axis magnitude coding:
    - `class_ctx_x/y[0..MV_CLASS_TREE_DEPTH)` – binary search tree over magnitude classes (0‥10), depth 4
    - `sign_ctx_x/y` – sign bit when class > 0
    - `mag_bit_ctx_x/y[0..MV_MAX_MAG_BITS)` – raw magnitude bits for the residual offset within a class (10 bits max)
  - Fractional contexts: `sub_x_has_ctx`, `sub_y_has_ctx` (presence-only; no sign contexts — see below)

**Mode context computation** (`mv_mode_context`): scores left and top neighbor modes as Zero=2, Nearest/Near/TopRight/TopLeft/Temporal=1, New=0. Sum is clamped to `[0, 3]`.

Encoding order per block (raster scan):

1. **Skip flag** – emit `block.skip` via `skip_ctx`.
2. **Mode tree** – encode booleans using the context indexed by neighboring modes:
   1. `is_new` → `mode_is_new_ctx[ctx]`. If `true`, mode = New.
   2. If `false`: `is_nonzero` → `mode_is_nonzero_ctx[ctx]`. If `false`, mode = Zero.
   3. If `is_nonzero`: `is_near` → `mode_is_near_ctx[ctx]`. If `true`, mode = Near.
   4. If not Near: `is_temporal` → `mode_is_temporal_ctx[ctx]`. If `true`, mode = Temporal.
   5. If not Temporal: `is_top_right` → `mode_is_top_right_ctx[ctx]`. If `true`, mode = TopRight.
   6. If not TopRight: `is_top_left` → `mode_is_top_left_ctx[ctx]`. If `true`, mode = TopLeft. Otherwise mode = Nearest.
3. **NEW base selector** – if the block is `New`, encode via a 4-level binary tree:
   1. `is_nearest` → `new_base_is_nearest_ctx[ctx]`. If `true`, base = nearest (0).
   2. If `false`: `is_near` → `new_base_is_near_ctx[ctx]`. If `true`, base = near (1).
   3. If `false`: `is_top_right` → `new_base_is_top_right_ctx[ctx]`. If `true`, base = top-right (2).
   4. If `false`: `is_top_left` → `new_base_is_top_left_ctx[ctx]`. If `true`, base = top-left (3). Otherwise base = temporal (4).
4. **NEW deltas** – only present for `New` blocks:
   - **Candidate history check** (only when candidate list is non-empty):
     - Compute `hit_ctx_idx = (left_hit as usize) + (top_hit as usize)` where left/top hits refer to whether the immediately preceding blocks in those directions were candidate hits.
     - Emit `is_candidate_hit` via `candidate_hit_ctx[hit_ctx_idx]`.
     - If hit: emit candidate index via binary tree using `candidate_idx_ctx`. The delta and fractional offsets are recovered from the candidate entry.
     - If miss: encode `delta_x` and `delta_y` explicitly as below, then encode fractional flags.
   - If no candidates exist, always encode deltas explicitly.
   - **Explicit delta encoding** (if miss or no candidates):
     - Each component `(delta_x, delta_y)` is encoded independently:
     - Compute the magnitude class (`0`..`MV_MAX_CLASS=10`) via binary search tree using `class_ctx_*` (depth 4).
     - If the class is `0`, the component is zero and no further symbols are written.
     - Otherwise emit the sign bit via `sign_ctx_*`, then the remaining magnitude bits (`mv_class_bits(class)` = `class - 1` for class ≥ 2, 0 for class ≤ 1) via the corresponding `mag_bit_ctx_*` slots (LSB first).
     - Magnitude reconstruction: `value = mv_class_base(class) + offset`, where `mv_class_base(class) = 2^(class-1)` for class > 0 (0 for class 0).
5. **Fractional offsets** – if not recovered from candidate history:
   - Encode `has_frac_x` via `sub_x_has_ctx`. If present, subpel = PlusHalf (+0.5); if absent, subpel = Zero.
   - Encode `has_frac_y` via `sub_y_has_ctx`. Same convention.
   - **Note**: only `PlusHalf` can be explicitly coded. `MinusHalf` values only arise through predictor propagation (inherited from neighbor/temporal modes), never from explicit NEW encoding.

**Candidate history maintenance**: after each frame, both encoder and decoder update their candidate lists identically:
- Decay all existing counts by factor 7/8 (integer: `count = (count * 7) / 8`).
- Increment counts for all NEW blocks in the frame (excluding those with MinusHalf subpel).
- Remove entries with zero count.
- If total NEW history < 100, clear the candidate list entirely.
- Otherwise, sort by count (descending), filter to entries with count > total/100, take top `CANDIDATE_SIZE = 16`.

### 4.6 Reconstructing absolute motion vectors and skip mask

Given:

- `mv_blocks` from `MvRansDecoder::decode_frame(blocks_w, blocks_h)`
- Block grid dimensions
- Frame-level bias `global_mv = (bias_dx, bias_dy)`
- `prev_mvs` (previous frame's reconstructed MV list, for temporal prediction)

Process every block in raster order:

1. Derive the `nearest`/`near`/`temporal` predictor triplet and raw `top_right`/`top_left` neighbors from blocks that have already been reconstructed, using `derive_mv_predictors` and `gather_mv_neighbor_set` (same logic as §4.4).
2. Apply NEW deltas with bias removal:
   - For NEW blocks: `delta_x = (block.delta_x as i16 + bias_dx).clamp(-128, 127) as i8`, same for Y.
   - For all other modes: delta is implicitly zero.
3. Determine the base vector (including subpixel components):
   - `Zero` → `(0, 0, 0, 0)`
   - `Nearest` → `predictors.nearest`
   - `Near` → `predictors.near`
   - `TopRight` → `neighbors.top_right`, falling back to `predictors.nearest` if unavailable
   - `TopLeft` → `neighbors.top_left`, falling back to `predictors.nearest` if unavailable
   - `Temporal` → `predictors.temporal`
   - `New` → base selected by `new_base`: 0=nearest, 1=near, 2=top_right, 3=top_left, 4=temporal
4. Compute final integer MV: `dx = (base_dx + delta_x).clamp(-128, 127)`, same for dy.
5. Determine subpixel offsets:
   - For `New` blocks: use the decoded `subpel_x`/`subpel_y` directly (from candidate or explicit coding).
   - For all other modes: inherit subpixel from the base predictor.
6. Assemble `MotionVector { dx, dy, subpixel_x, subpixel_y, skip }` and append to the block list.

Because both encoder and decoder run the same predictor derivation and mode tree, the absolute motion field and authoritative skip mask are reproduced exactly even though only NEW blocks transmit explicit residuals.

### 4.7 Skip mode (authoritative semantics)

A block is in **skip mode** if `block.skip` is true (decoded from the RANS skip flag).

In skip mode:

- The decoder **applies no residual** to that block (i.e. `current_block = predicted_block`).
- Motion vector components `(dx, dy, subpixel_x, subpixel_y)` are still fully decoded and used for neighbor prediction.
- `skip_mask[block]` is built from the decoded skip flags and is treated as **authoritative**:
  - The encoder may additionally mark blocks as skipped if their residual quantizes to all zeros.
  - In that case it updates the skip flag in the MvCodedBlock to reflect the optimized skip mask.

---

## 5. Residual data payload (`residual_yuv420`)

### 5.1 Overview

The residual data encodes the difference between the current frame and the motion-compensated predicted frame. All operations occur in **YUV420** color space. The encoder and decoder maintain reference frames as YUV420.

### 5.2 Residual computation

**Intra frames**: the residual is computed as `pixel - 128` for each YUV channel, where 128 is the implicit DC predictor (mid-gray).

**Inter frames**: the residual is computed as `current_yuv - predicted_yuv` (signed i16, range [-255, 255]) directly on YUV420 planes. The predicted frame is built by motion-compensating the previous reconstructed YUV420 frame.

For each motion block (16×16 pixels):
- Y plane: 16×16 block
- U plane: 8×8 block (YUV420 halves both dimensions)
- V plane: 8×8 block

### 5.3 DCT encoding and quantization

#### 5.3.1 Base quantization step

The quality parameter (1-100) maps to a base quantization step:

- Quality 100: `quant_step = 1.0` (minimal quantization)
- Quality 1: `quant_step = 112.0` (maximum quantization)
- Linear interpolation: `quant_step = 1.0 + (100 - quality) / 99.0 * 111.0`

#### 5.3.2 Intra quantization (JPEG-style perceptual matrices)

Intra frames use **JPEG-style perceptual quantization matrices** instead of uniform quantization:

- Standard JPEG luminance 8×8 table for Y plane
- Standard JPEG chrominance 8×8 table for U/V planes
- Each table entry is normalized by its table's DC value and scaled by `quant_step`: `table[i] = round(JPEG[i] / JPEG[0] * quant_step).max(1)`. This makes DC quantization always equal to `quant_step` regardless of table (luma DC=16, chroma DC=17).
- For 16×16 Y blocks: the 8×8 luma table is bilinearly upsampled to 16×16 before applying the scaling above
- DC prediction is enabled: the DC coefficient is delta-coded from the previous block's DC (per plane)

#### 5.3.3 Inter quantization (Adaptive Quantization)

Inter frames use **adaptive quantization (AQ)** with x264-style variance weighting:

- Per-block variance is computed on the **predicted frame's Y plane** (available to both encoder and decoder)
- AQ formula: `block_qs = base_qs × 2^(strength × (ln(var) - ln(avg_var)) / 6)`
  - `strength`: AQ strength, compile-time constant = 0.8 (0.0 would disable AQ)
  - `var`: block variance (floored at 4.0 to avoid log(0))
  - `avg_var`: geometric mean of all block variances
- Effect: busy/textured blocks get coarser quantization; smooth blocks get finer quantization
- The same per-block quant steps are computed identically by both encoder and decoder from the predicted frame
- No DC prediction for inter frames

#### 5.3.4 Inter dead zone

Inter frames use a **wider dead zone** than standard rounding:

- Threshold: `dead_zone = 0.75` (default; configurable via `--inter-dead-zone`)
- A coefficient quantizes to **zero** if `|coeff / quant_step| < dead_zone`; otherwise it rounds to the nearest integer as usual
- Standard rounding corresponds to `dead_zone = 0.5` (H.264-style inter coding uses ~0.75)
- This only applies to inter (`_aq`) quantization. Intra (`_matrix`) quantization always uses standard rounding (dead zone = 0.5)
- The dead zone threshold is a pure encoder parameter — it is **not transmitted** in the bitstream. The decoder dequantizes identically regardless of which dead zone was used during encoding.

#### 5.3.5 DCT transform

1. **2D DCT**: Apply 2D Discrete Cosine Transform to the residual block (16×16 for Y, 8×8 for U/V)
2. **Quantization**: Divide DCT coefficients by the appropriate quantization value, apply dead zone, then round
3. **Zigzag scan**: Reorder quantized coefficients from 2D (row-major) to 1D zigzag order

### 5.4 RANS encoding

After zigzag scanning, quantized DCT coefficients are encoded using **RANS (Range Asymmetric Numeral Systems)** entropy coding with class-based magnitude coding and band-specific contexts.

#### Context inventory

- **Per-position significance contexts**: 256 for luma (Y), 64 for chroma (U/V)
- **EOB contexts** (class-based, separate for Y and UV):
  - Luma: 9 class contexts (max EOB value 255 → class 8)
  - Chroma: 7 class contexts (max EOB value 63 → class 6)
- **Magnitude contexts** (class-based, per frequency band):
  - Luma: 4 bands × 13 class contexts
    - Band 0: DC (position 0)
    - Band 1: low AC (positions 1–15)
    - Band 2: mid AC (positions 16–63)
    - Band 3: high AC (positions 64–255)
  - Chroma: 3 bands × 13 class contexts
    - Band 0: DC (position 0)
    - Band 1: low AC (positions 1–7)
    - Band 2: high AC (positions 8–63)
- **DC prediction state** (intra only): previous DC value per plane [Y, U, V], initialized to 0

#### Class-based coding scheme

Both EOB and magnitude values use the same log2-based class mapping:
- Class 0: value = 0
- Class 1: value = 1
- Class k (k ≥ 2): values in range `[2^(k-1), 2^k - 1]`

Encoding: truncated unary for the class index (emit `true` to continue, `false` to stop, or implicit stop at max class), then `(class - 1)` bypass-coded offset bits (MSB first) for class ≥ 2.

Decoding mirrors: read unary class, then offset bits.

#### Bitstream layout

- Residual coefficients for the entire frame share **one** `RansWriter32` output buffer. There is no per-block header inside the stream; decoders must follow the exact symbol order below.
- For each macroblock in raster order:
  1. If the block is skipped, no symbols are emitted for Y/U/V and the decoder implicitly fills the residual with zeros.
  2. Otherwise, emit plane data in Y, then U, then V order. Each plane performs:
     - **EOB** (class-based coding): encode the index of the last non-zero coefficient using class-based coding with plane-specific EOB contexts (luma or chroma). For intra with DC prediction, the DC delta is considered when determining the last non-zero position.
     - For every coefficient index from 0 through EOB (inclusive):
       * **Significance bit**: boolean stating whether the coefficient is non-zero. Context index equals the coefficient index (0-255 for Y, 0-63 for U/V). For intra DC prediction, index 0 encodes the DC delta rather than the raw DC value.
       * **Sign bit** (if significant): bypass-coded boolean (0 = positive, 1 = negative).
       * **Magnitude** (if significant): encode `abs(value) - 1` using class-based coding with band-specific magnitude contexts. The band is determined by the coefficient position within the plane.
- RANS contexts (significance, EOB, magnitude) adapt across all blocks within the frame. **Intra frames** create a fresh RANS encoder/decoder with DC prediction enabled. **Inter frames** create a fresh RANS encoder/decoder without DC prediction.

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

**Intra**:

1. Create RANS decoder **with DC prediction enabled** and consume `residual_yuv420` bytes.
2. For each block in raster order (no skip mask for intra):
   - Decode Y (256 coefficients), U (64), V (64) via RANS. DC prediction reconstructs the raw DC from the delta + previous block's DC.
3. Build JPEG-style perceptual quantization tables from `quality`.
4. For each plane: reverse zigzag → dequantize using perceptual matrix → inverse 2D DCT.
5. Reconstruct pixel values: `pixel = residual + 128`, clamped to [0, 255].

**Inter**:

1. If `skip_residuals` flag is set, return the predicted frame directly.
2. Create RANS decoder **without DC prediction** and consume `residual_yuv420` bytes.
3. For each block in raster order:
   - If `skip_mask[block]`: skip (residual stays zero).
   - Otherwise: decode Y (256), U (64), V (64) via RANS.
4. Compute adaptive per-block quant steps from the predicted frame's Y plane (same formula as encoder, see §5.3.3).
5. For each plane: reverse zigzag → dequantize using per-block AQ quant steps → inverse 2D DCT.
6. Apply residuals: `recon = predicted + residual` per YUV channel, clamped to [0, 255].

---

## 6. Padding and cropping rules

### 6.1 Encoder padding (display → storage)

Input frames arrive as display-sized RGB24.

The encoder pads to storage size via edge replication:

- **Right pad**: repeat the last pixel in each row.
- **Bottom pad**: repeat the last (already padded) row.

Both intra and inter residual coding operate on the **storage-sized** YUV420 planes derived from this padded RGB24 image.

### 6.2 Decoder cropping (storage → display)

Decoder reconstructs the full storage frame as YUV420, converts to RGB24, then outputs only the top-left `display_width × display_height` region.

---

## 7. Encoder algorithm (normative)

The encoder maintains:

- `prev_recon_yuv`: previous reconstructed storage frame as YUV420, i.e. what the decoder would have.
- `prev_mvs`: previous frame's motion vector list (for temporal prediction).
- `mv_rans_encoder`: MV RANS encoder with contexts persisting across inter frames.

### 7.1 I-frame

1. Pad current display frame → `curr_storage_rgb`.
2. Convert `curr_storage_rgb` to storage-sized **YUV420** (`curr_yuv`).
3. Compute residual: `res = curr_yuv - 128` (mid-gray reference).
4. DCT encode all planes with **JPEG-style perceptual quantization matrices** and **DC prediction**.
5. RANS-encode coefficients with DC prediction enabled.
6. Reconstruct: IDCT + dequantize, then `pixel = residual + 128`, clamped to [0, 255].
7. **Deblock** `recon_yuv` in place (§14), level derived from `quality`, filtering **all** interior block edges (every intra block is coded).
8. Set `prev_recon_yuv = recon_yuv`. Clear `prev_mvs`. Reset MV RANS contexts.
9. Write an Intra record with `(quality, res_size, residual_yuv420)`.

### 7.2 P-frame (motion-compensated inter)

Given `prev_recon_yuv` and `curr_storage_yuv`:

1. **Motion estimation** per 16×16 block using hex search with lambda-weighted SAD cost and half-pixel refinement (see §11).
2. **Motion prediction**: derive VP9-style predictors, select best mode from {Zero, Nearest, Near, TopRight, TopLeft, Temporal, New} using the mode cost model in §11.2.
3. Build `predicted_yuv` from `prev_recon_yuv` using absolute MVs with half-pixel precision, operating directly on YUV420 planes.
4. **Skip decision** via RDO (see §12): for each block, compute SATD against `predicted_yuv` and apply the lambda-weighted cost model to decide skip vs. code. Blocks that pass the skip decision are marked in `skip_mask`; others are residual-encoded with **adaptive quantization** (no DC prediction).
5. **Bias computation**: collect all raw NEW-mode deltas and compute per-axis `global_mv` bias (see §3.2).
6. **Motion vector encoding**: construct `MvCodedBlock` list (subtracting bias from NEW deltas), update skip flags with the optimized skip mask (blocks whose residual quantized to all-zeros are additionally marked skip), encode via `MvRansEncoder`.
7. **Reconstruct reference**: IDCT + dequantize residuals, `recon_yuv = predicted_yuv + residual`.
8. **Deblock** `recon_yuv` in place (§14), level derived from `inter_quality`, with the per-block filter mask built from the **optimized** skip mask and the **reconstructed** MV list (§14.3).
9. Set `prev_recon_yuv = recon_yuv`. Set `prev_mvs` = the **reconstructed** absolute MV list (one entry per block, derived by applying mode + predictors during step 2 — identical to what the decoder would reconstruct). Do **not** use raw motion-search results: when a predictor carries a `MinusHalf` subpixel component (propagated from an earlier frame), the reconstructed MV's component representation can differ from the canonical search output even at the same half-pixel position, and using the wrong list breaks `Temporal`-mode decoding in the next inter frame.
10. Write an Inter record containing `(quality, global_mv, mv_deflate, residual_yuv420)`.

---

## 8. Decoder algorithm (normative)

Decoder maintains:

- `prev_recon_yuv`: previous reconstructed storage frame as YUV420.
- `prev_mvs`: previous frame's motion vector list (for temporal prediction). `None` after intra or before first inter.
- `mv_rans_decoder`: MV RANS decoder with contexts persisting across inter frames.

### 8.1 I-frame

1. From the Intra record, read `quality` and `residual_yuv420`.
2. Create RANS decoder with DC prediction, decode all blocks (no skip mask).
3. Build JPEG-style quant tables from `quality`, IDCT + dequantize, reconstruct as `pixel = residual + 128`.
4. **Deblock** `recon_yuv` in place (§14) exactly as the encoder did.
5. Set `prev_recon_yuv = recon_yuv`. Clear `prev_mvs`. Reset MV RANS contexts.
6. Convert `recon_yuv` to RGB24 and crop to display size for output.

### 8.2 P-frame

1. Compute `blocks_w = storage_width/16`, `blocks_h = storage_height/16`.
2. Decode motion vectors:
   - `mv_rans_decoder.consume_frame(mv_deflate)`, then `decode_frame(blocks_w, blocks_h)` → list of `MvCodedBlock`.
   - Reconstruct absolute MVs and `skip_mask` from these blocks using the predictor logic (§4.4/§4.6), applying global MV bias to NEW deltas.
3. Build `predicted_yuv` from `prev_recon_yuv` using absolute MVs with half-pixel precision on YUV420 planes.
4. Decode residuals:
   - Create RANS decoder (no DC prediction), consume `residual_yuv420`.
   - For each block in raster order: if skipped, leave residual zero; otherwise decode Y/U/V coefficients.
   - Compute adaptive per-block quant steps from `predicted_yuv` (same AQ formula as encoder).
   - IDCT + dequantize using per-block AQ quant steps.
5. Apply residuals: `recon_yuv = predicted_yuv + residual`, clamped to [0, 255].
6. **Deblock** `recon_yuv` in place (§14) with the filter mask from the decoded skip flags and reconstructed MVs.
7. Set `prev_recon_yuv = recon_yuv`. Store MVs as `prev_mvs`.
8. Convert `recon_yuv` to RGB24 and crop to display size for output.

---

## 9. CLI knobs

`ri-cli encode` exposes:

- `--intra-quality <1..=100>`: Intra-frame residual quality (default: 90)
- `--inter-quality <1..=100>`: Quality parameter for residual DCT quantization in P-frames (default: 35)
- `--keyframe-interval <N>`: I-frame every N frames (default: 30). Special values: `0` = all-intra (every frame is an I-frame). `Automatic` exists as a config variant but is currently unimplemented and falls back to Fixed(30); scene-change detection is not yet implemented.
- `--search-range <0..=127>`: integer motion search radius (±N pixels); must fit in i8 (default: 12)
- `--me-zero-mv-threshold <i64>`: if SAD(0,0) ≤ threshold, skip integer search entirely for that block (0 = disabled, default: 0). See §11.
- `--me-predictor-threshold <i64>`: if best-candidate SAD after seed evaluation ≤ threshold, skip hex refinement (0 = disabled, default: 0). See §11.
- `--skip-threshold <0..=255>`: per-block SATD-based hard guard for skip decisions — if block SATD > `threshold × block_area_bytes`, residuals are always coded regardless of RDO outcome (0 = disable all skips). (default: 3). See §12.
- `--rdo-lambda-mult <f64>`: RDO lambda multiplier (see §12). Scales the lambda used in skip cost comparisons (default: 0.49)
- `--inter-dead-zone <f32>`: dead zone threshold for inter quantization (0.5 = standard rounding, 0.75 = H.264-style wider dead zone) (default: 0.75)
- `--max-frames <u64>`: encode at most this many frames (0 = no limit)

**Note on `mv_deflate_level`**: the `EncoderConfig` struct retains a `mv_deflate_level` field (0..=10) for historical reasons. It has no effect — the motion vector stream uses pure RANS and no DEFLATE layer. The field is vestigial and may be removed in a future version.

`ri-cli decode` exposes:

- `--skip-residuals`: if set, decodes motion vectors only and outputs motion‑predicted frames (no residuals applied)

Additional subcommands:

- `extract-frame`: extract a single decoded frame from a `.riv` file into a directory on disk
- `roundtrip`: encode to `.riv` in memory and immediately re‑encode to a conventional video file for visual inspection

---

## 10. Notes / caveats

- This is an early format; it is **not designed for backward compatibility** yet.
- Motion search cost grows quickly with `search_range`; keeping it small is recommended.
- Motion search uses **hexagonal search** with lambda-weighted cost and half-pixel refinement; see §11 for the full algorithm.
- **Optional parallelism**: the encoder can use Rayon for per-row-strip parallel motion estimation when compiled with the `threads` feature. SIMD-accelerated SAD/half-pixel kernels are available when compiled with the `simd` feature; the scalar fallback is always present.
- Motion vector prediction uses a **7-mode tree** (`Zero`, `Nearest`, `Near`, `TopRight`, `TopLeft`, `Temporal`, `New`) extending the VP9-style approach with direct top-right and top-left neighbor modes.
- Predictor tuples include all four components `(dx, dy, subpixel_x, subpixel_y)`, not just the integer part.
- Motion vectors use **half-pixel precision** with bilinear interpolation on YUV420 planes.
- **Intra frames** use JPEG-style perceptual quantization matrices and DC prediction for better compression of low-frequency content.
- **Inter frames** use adaptive quantization (x264-style variance weighting) computed from the predicted frame, ensuring encoder/decoder agreement without transmitting per-block quant steps. A wider dead zone (default 0.75 vs standard 0.5) is applied during inter quantization to aggressively zero near-threshold coefficients that carry little signal.
- Residual RANS uses **class-based** magnitude and EOB coding (log2-based classes with truncated unary + bypass bits), not simple unary. Magnitude contexts are **band-specific** (4 luma bands, 3 chroma bands × 13 classes each).
- Fractional offsets for NEW blocks only encode `PlusHalf` (+0.5); `MinusHalf` (-0.5) only propagates through predictor modes. This avoids needing a sign bit in the fractional coding.
- All prediction and reconstruction operates in **YUV420 space**. RGB24 is only the external input/output format.
- DCT coefficient values are stored as i16 (range -32768 to 32767) after quantization.

---

## 11. Motion estimation algorithm (encoder)

Motion estimation runs per 16×16 block in raster order. It outputs one integer-pixel MV per block plus a half-pixel refinement. The search operates on **luma (Y) only** — SAD/SATD is computed on the Y plane. Chroma inherits the chosen MV.

### 11.1 Candidate seeding

For each block `(bx, by)`:

1. Collect spatial/temporal motion predictors in order: **left, top, top-right, top-left, temporal**. Integer components `(dx, dy)` only; subpixel is ignored here.
2. Deduplicate predictors; pad with `(0, 0)` if fewer than two unique values.
3. Build a search candidate list: `(0, 0)` first, then each unique predictor — all clamped to `[−search_range, search_range]`. This list seeds the integer search.
4. The first two unique predictors are called `nearest` and `near`. The **median** of left/top/top-right integer components is also computed as `median_x = median3(left.dx, top.dx, topright.dx)`, same for Y.

### 11.2 Lambda-weighted cost model

Every candidate MV is evaluated using a combined cost:

```
total_cost(dx, dy) = SAD(dx, dy) + lambda * scale * mode_bits(dx, dy)
```

where `scale = 2.0` and `lambda` comes from the RDO context (§12). `mode_bits` is an approximation of how many bits the mode + delta would cost:

| Condition | Estimated bits |
|---|---|
| `(dx, dy) = (0, 0)` | 0.5 |
| `(dx, dy) = nearest` or `near` or `median` | 1.5 |
| New mode (vs nearest): delta `(dx−nearest.dx, dy−nearest.dy)` | `6.0 + log2_bits(Δx) + log2_bits(Δy)` |
| New mode (vs near): same formula using `near` | same |

The cheapest estimate across all applicable modes is used. `log2_bits(d) = 1.0` if `d=0`, else `2*log2(|d|) + 2.0`.

The SAD kernel uses **limit-based early exit**: evaluation aborts once the accumulated partial SAD exceeds `state.cost − mv_cost`, avoiding useless work.

### 11.3 Integer search phases

**Phase 0 – Zero-MV early termination** (if `me_zero_mv_threshold > 0`):
- Compute `SAD(0, 0)`. If `SAD ≤ me_zero_mv_threshold`, skip all further integer search and use MV `(0, 0)`.

**Phase 1 – Seed candidate evaluation**:
- Evaluate each non-zero candidate from §11.1 using the cost model. Update the best state `(dx, dy, cost)`.
- If `me_predictor_threshold > 0` and the best SAD after this phase ≤ `me_predictor_threshold`, skip phases 2 and 3.

**Phase 2 – Large Hex Search (LHS)**:
- Iteratively check 6 hex offsets `{(±2,0), (±1,±2)}` around the current best position.
- Each candidate uses limit-based early exit.
- Repeat until no offset improves the cost (convergence).

**Phase 3 – Small Diamond Search (SDS)**:
- Check the 4 cardinal neighbors `{(±1,0), (0,±1)}` around the converged position. One final pass (no loop).

### 11.4 Half-pixel refinement

After integer search, check all 8 sub-pixel candidates `{(±1, ±1)}` in half-pixel units around the best integer MV. Sub-pixel SAD uses bilinear interpolation. A fractional penalty is added: `penalty = lambda × 2.0 × 1.5` bits if either axis is fractional. The integer-aligned position (0 fractional offset) is also re-evaluated for comparison.

The half-pixel offset convention: a `dx_hp / 2` integer part with `dx_hp % 2` fractional part. `MinusHalf` (−0.5) results from a `−1` fractional unit on a `0` integer base; per the canonicalization rule (§4.5 note), such values only arise through predictor propagation, not from the search.

### 11.5 SIMD and threading

When compiled with the `simd` feature, SAD and half-pixel kernels use SIMD intrinsics. The `threads` feature enables Rayon-based parallelism: blocks are divided into horizontal strips (4× the thread count for load balancing), and strips are processed in parallel. Within each strip, rows are processed sequentially to allow use of top-neighbor predictors.

---

## 12. Rate-distortion optimization (encoder skip decisions)

The encoder uses a lambda-weighted heuristic to decide, per block, whether to transmit residuals (coded) or skip them entirely.

### 12.1 Lambda derivation

From `inter_quality` (1–100) and `rdo_lambda_mult`:

```
qp = round((100 − quality) × 0.6).clamp(0, 60)
lambda = rdo_lambda_mult × 2^((qp − 12) / 3)   // x264-style
```

`q` = `quant_step_from_quality(quality)` (the base quantization step, see §5.3.1). Derived parameters:

| Parameter | Formula |
|---|---|
| `distortion_retention` | `0.15 + ((q − 1) / 111) × 0.55` |
| `residual_rate_slope` | `0.5 / q` |
| `residual_rate_intercept` | `5.0 + 0.56 × q` |
| `skip_flag_bits` | `0.55 − (quality/100) × 0.15` |
| `keep_flag_bits` | `1.20 + (quality/100) × 0.35` |

### 12.2 Per-block cost model

Given a block's **SAD** (Sum of Absolute Differences on the luma plane, reused from motion estimation):

```
skip_cost  = SAD + lambda × skip_flag_bits
coded_cost = (SAD × distortion_retention) + lambda × (keep_flag_bits + residual_rate_intercept + residual_rate_slope × SAD)
```

The block is skipped if `skip_cost ≤ coded_cost` — unless a guard overrides.

**Note**: the encoder reuses the SAD score returned by motion estimation rather than computing a separate SATD on the post-MC residual. The variable is named `satd` internally but holds a plain SAD value.

### 12.3 Skip guards (override conditions)

Three conditions force residuals to be coded regardless of the cost comparison:

1. **Disabled** (`skip_threshold = 0`): all blocks are always coded. `skip_cost` / `coded_cost` are computed but ignored.
2. **FractionalMv**: the block's best MV has a sub-pixel offset (not integer-aligned). Sub-pixel blocks are always coded because the prediction is approximate and the residual carries real signal.
3. **ThresholdExceeded**: `SAD > block_area_bytes × skip_threshold`. Busy blocks are always coded. `block_area_bytes = 16 × 16 × 3` (matches the encoder's RGB-equivalent area used for this guard; the SAD itself is luma-only).

When a guard fires, `skip = false` regardless of cost. The guard reason is tracked in per-frame telemetry but is not transmitted in the bitstream.

### 12.4 Final skip mask

The RDO skip mask is an initial estimate. After residual encoding, any block whose quantized DCT coefficients are all zero is additionally marked as skip in the MV coded-block flags. The decoder's authoritative skip mask therefore reflects both the RDO decision and quantization-induced zeros.

---

## 13. Entropy coder specification

All RANS streams in this format use two primitives: a binary probability context (`BinProb`) and a 32-bit word-aligned rANS engine. This section is the normative description for both. Any conforming implementation must reproduce these exactly — the bitstream is sensitive to every detail.

### 13.1 rANS engine

The rANS engine is a word-aligned (u16) adaptation of the public-domain ryg_rans byte-aligned design by Fabian Giesen. The only structural change is substituting u16 word emission for byte emission, which shifts the normalization constants accordingly.

**Parameters**

| Name | Value | Meaning |
|---|---|---|
| `RANS_L` | `1 << 16` = 65536 | Lower bound of the normalization interval |
| `SCALE_BITS` | 8 | `log2(M)`; frequency table has M = 256 entries |
| State type | `u32` | Single 32-bit unsigned integer |

The normalization interval is `[RANS_L, RANS_L * M)` = `[2^16, 2^24)`.

**Encoder init**: state = `RANS_L`.

**Encode one symbol** `(start: u32, freq: NonZeroU32)` — frequencies must sum to M = 256:
1. Renormalize: let `x_max = ((RANS_L >> SCALE_BITS) << 16) * freq`. If `state >= x_max`, emit `state & 0xffff` as a LE u16 word, then `state >>= 16`.
2. Encode: `state = (state / freq) << SCALE_BITS | (state % freq) + start`.

**Encoder flush**: emit `state & 0xffff` as LE u16, then `(state >> 16) & 0xffff` as LE u16.

**Decoder init**: read two LE u16 words `lo`, `hi`; state = `lo | (hi << 16)`.

**Decode — get current slot**: `slot = state & ((1 << SCALE_BITS) - 1)`. Compare against cumulative frequencies to identify the symbol.

**Decode advance** `(start: u32, freq: NonZeroU32)` — after identifying the symbol:
1. `state = freq * (state >> SCALE_BITS) + (state & (M-1)) - start`.
2. Renormalize: if `state < RANS_L`, read one LE u16 word `w`; `state = (state << 16) | w`.

**Dual-state interleaving**: all streams use two independent states `r0`, `r1` that alternate per symbol (r0 for even-indexed symbols, r1 for odd). This enables instruction-level parallelism on superscalar CPUs. The encoder buffers a full batch of `BATCH = 16386` symbols before draining; the decoder reloads both states every 16386 symbols. After encoding, `r0` is flushed first then `r1`; the decoder reads `r0` first then `r1` — note: because words are prepended to the output deque, what the decoder reads as `r0` at init time is the final state of the encoder's `r1`, and vice versa. This cross-mapping is intentional and must be preserved.

**Bypass coding** (equiprobable bits): `start = state & 0x80`, `freq = 128`. Bit is 1 if `start != 0`.

### 13.2 BinProb — binary probability context

`BinProb` is an adaptive binary context that tracks the probability of observing a 0-bit. The model is the saturating-counter approach described in the VP8 bitstream specification (RFC 6386 §7.3), adapted here as a self-contained normative description.

**State**: a pair of saturating 8-bit counters `(z, o)` where `z` counts zero-observations and `o` counts one-observations. Packed into a `u16` as `(z << 8) | o`. Both counters are always in `[1, 255]`.

**Initial state**: `z = 1, o = 1` (balanced, no prior observations).

**Probability of zero**: `p_zero = floor(z * 256 / (z + o))`, clamped to `[1, 255]`. This is the value passed to the rANS engine as `freq` when coding a 0-bit (`start = 0, freq = p_zero`); a 1-bit uses `start = p_zero, freq = 256 - p_zero`.

**Update after observing bit `b`** — the observed counter is incremented; on overflow, both counters are halved to prevent unbounded growth while preserving the ratio:

Let `observed = if b == 0 { z } else { o }` and `other = if b == 0 { o } else { z }`.

1. If `observed < 255`: increment `observed` by 1. Done.
2. If `observed == 255` (overflow):
   - **Special case** — if `other == 1` (the opposite symbol has barely been seen), preserve the extreme probability: set `observed = 255`, `other = 1` unchanged. Done.
   - **Normal case**: set `observed = 129` (fixed — the implementation forces the observed byte to `0x81 = 129` via bitmask), then `other = other >> 1` (truncating, i.e. floor division). Examples: `(255, 2) → (129, 1)`, `(255, 100) → (129, 50)`, `(255, 254) → (129, 127)`.

The special case prevents the model from forgetting a strong prior when the opposite symbol has appeared only once. The normal halving keeps counters from saturating and allows the model to track drifting sources.

**Note on implementation**: the exact numerical values produced by the update rule above must match to the bit. In particular, the `other == 1` guard uses the value of `other` *before* any modification.

---

## 14. In-loop deblocking filter (normative, v5+)

Authoritative code: `reitero_video/reitero_video_common/src/deblock.rs`.

Applied in place to every reconstructed storage frame **before** it is stored as `prev_recon_yuv` (and, in the decoder, before RGB conversion/output). Encoder and decoder MUST apply it with identical inputs or the prediction references drift. Nothing is transmitted: the filter level derives from the frame's `quality` byte and the filter mask from data both sides already reconstruct.

### 14.1 Filter level and thresholds

```
level = clamp(round(quant_step(quality) * 0.6), 0, 63)
```

where `quant_step` is the §5.3.1 mapping and `quality` is the frame record's quality byte (`intra_quality` for I-frames, `inter_quality` for P-frames — AQ per-block steps are NOT used). `level = 0` disables the filter for the frame. The `0.6` scale was tuned on the bench footage set (akiyo/bus/foreman/mobile CIF + park_joy 1080p, three quality points each) for minimum bytes × DSSIM.

From `level`, derive the RFC 6386 §15.1 thresholds for macroblock edges with `sharpness = 0`:

```
interior_limit = max(level, 1)
edge_limit     = (level + 2) * 2 + interior_limit
hev_threshold  = I-frame:  level >= 40 → 2, level >= 15 → 1, else 0
                 P-frame:  level >= 40 → 3, level >= 20 → 2, level >= 15 → 1, else 0
```

### 14.2 Filter algorithm

The filter is the VP8 "normal loop filter" for **macroblock edges** (RFC 6386 §15.2/§15.3) applied to interior block-grid edges only (never the frame border): the **16-pixel** grid on the Y plane and the corresponding **8-pixel** grid on the U and V planes (both grids map 1:1 onto the macroblock grid, so one mask entry per 16×16 macroblock covers all three planes). RIV has no interior transform edges (one 16×16 luma / 8×8 chroma DCT per macroblock), so the subblock-edge filter variant does not exist here.

Per plane, first all **vertical** edges are processed (each block column boundary `x = grid, 2*grid, …`, every row top-to-bottom), then all **horizontal** edges (each block row boundary `y = grid, 2*grid, …`, every column left-to-right). The second pass reads the output of the first.

Each edge-crossing line examines eight pixels `p3 p2 p1 p0 | q0 q1 q2 q3` (four each side, perpendicular to the edge):

1. **Filter mask** (unbiased 0..255 values, integer arithmetic, truncating division) — filter only if the step across the edge is small AND both sides are locally smooth:

```
  2*|p0 - q0| + |q1 - p1|/2 <= edge_limit
  and |p3-p2| <= interior_limit and |p2-p1| <= interior_limit and |p1-p0| <= interior_limit
  and |q1-q0| <= interior_limit and |q2-q1| <= interior_limit and |q3-q2| <= interior_limit
```

2. **High edge variance (hev)**: `hev = |p1-p0| > hev_threshold or |q1-q0| > hev_threshold`.

3. **Adjustment** on signed-biased values (`v - 128`; all intermediate values clamped to `[-128, 127]`, written `c(...)`; `>>` is arithmetic shift), with `w = c(c(p1 - q1) + 3*(q0 - p0))`:

   - If `hev` (locally sharp — narrow correction only):

     ```
     F1 = c(w + 4) >> 3;  F2 = c(w + 3) >> 3
     q0' = q0 - F1;  p0' = p0 + F2
     ```

   - Else (locally smooth — wide taper across three pixels each side):

     ```
     a = c((27*w + 63) >> 7);  q0' = c(q0 - a);  p0' = c(p0 + a)
     a = c((18*w + 63) >> 7);  q1' = c(q1 - a);  p1' = c(p1 + a)
     a = c(( 9*w + 63) >> 7);  q2' = c(q2 - a);  p2' = c(p2 + a)
     ```

   Results are re-biased (`+ 128`) and clamped to `[0, 255]`. `p3`/`q3` are never modified.

### 14.3 Filter mask (P-frames)

Blocks that are pure copies of the previous frame already contain filtered pixels; re-filtering them every frame would progressively blur static areas. Therefore each macroblock gets a flag:

```
filter[b] = !skip[b] || dx[b] != 0 || dy[b] != 0 || subpel_x[b] != 0 || subpel_y[b] != 0
```

using the **optimized** (authoritative, §4.7) skip flag and the **reconstructed** absolute MV of the block (the same list stored as `prev_mvs`). An edge is filtered iff **either** adjacent block's flag is set. I-frames filter all edges unconditionally (every block is coded).
