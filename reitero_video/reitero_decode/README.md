# reitero_decode

Decoder for the [Reitero video format](https://github.com/xhighway999/riv).

## Usage

```toml
[dependencies]
reitero_decode = "0.1"
```

Implement [`VideoReader`] for your input source, then feed it to [`Decoder`]:

```rust
use reitero_decode::{Decoder, DecodedFrame, DecodeError, Result, VideoReader};

struct SliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl VideoReader for SliceReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let n = buf.len().min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
    fn position(&mut self) -> u64 { self.pos as u64 }
    fn seek(&mut self, pos: u64) -> Result<()> {
        self.pos = pos as usize;
        Ok(())
    }
}

let reader = SliceReader { data: &riv_bytes, pos: 0 };
let mut decoder = Decoder::new(reader)?;

println!("{}x{} @ {} fps, {} frames",
    decoder.header().display_width,
    decoder.header().display_height,
    decoder.header().fps,
    decoder.header().frame_count,
);

while decoder.has_more_frames() {
    let frame: DecodedFrame = decoder.decode_frame()?;
    // frame.data — RGB24, row-major, frame.width * frame.height * 3 bytes
    // frame.timestamp — milliseconds
    // frame.frame_type — Intra or Inter
}
```

## Performance

Always build with `--release`. For maximum throughput on your own machine, enable
`target-cpu=native` via the workspace `full-opt` profile:

```sh
cargo build --profile full-opt
```

The `simd` feature (enabled by default) uses SIMD-accelerated residual decoding;
disabling it will reduce decode speed.

## Output

[`DecodedFrame`] contains:
- `data: Vec<u8>` — RGB24, cropped to display dimensions
- `width`, `height` — display size in pixels
- `timestamp` — milliseconds as encoded in the stream
- `frame_index` — zero-based frame number
- `frame_type` — [`FrameType::Intra`] or [`FrameType::Inter`]
