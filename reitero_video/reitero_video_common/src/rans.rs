//! Adaptive rANS entropy coding for the Reitero video codec.
//!
//! Two main components:
//! - [`BinProb`]: adaptive binary probability context. Tracks a saturating
//!   (false, true) counter pair and derives a symbol probability on demand.
//!   Update rule matches the VP8/WebM probability model (BSD-licensed by
//!   Google) — the algorithm is well-documented in the VP8 bitstream spec.
//! - [`RansEncoder`] / [`RansDecoder`]: 32-bit rANS (range Asymmetric
//!   Numeral Systems, Duda 2013) with dual interleaved states.  Symbols are
//!   buffered and encoded in reverse order so the bitstream can be written
//!   sequentially.  Scale is fixed at 8 bits (256-entry frequency table).

use std::{
    collections::VecDeque,
    io::{Read, Result, Write},
    num::{NonZeroU32, NonZeroU8},
};

// ---------------------------------------------------------------------------
// BinProb — adaptive binary probability context
// ---------------------------------------------------------------------------
//
// Internal layout: a u16 holding two saturating 8-bit counters.
//   bits [15:8] = count of 0-observations
//   bits  [7:0] = count of 1-observations
//
// Probability of observing 0 = zeros * 256 / (zeros + ones), clamped ≥ 1.
//

const fn build_zero_prob_table() -> [NonZeroU8; 65536] {
    let mut t = [NonZeroU8::MIN; 65536];
    let mut i = 1i32;
    while i < 65536 {
        let zeros = i >> 8;
        let ones = i & 0xff;
        if let Some(p) = NonZeroU8::new(((zeros << 8) / (zeros + ones)) as u8) {
            t[i as usize] = p;
        }
        i += 1;
    }
    t
}

static ZERO_PROB: [NonZeroU8; 65536] = build_zero_prob_table();

/// Binary probability context: adapts to the observed symbol frequency.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BinProb {
    packed: u16,
}

impl Default for BinProb {
    #[inline]
    fn default() -> Self {
        BinProb { packed: 0x0101 } // start balanced: 1 zero seen, 1 one seen
    }
}

impl BinProb {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Probability that the next bit is 0, in [1, 255].
    #[inline(always)]
    pub fn prob_zero(&self) -> NonZeroU8 {
        ZERO_PROB[self.packed as usize]
    }

    /// Return an updated context after observing `bit`.
    #[inline(always)]
    pub fn observe(self, bit: bool) -> Self {
        // Rotate so the counter for the observed symbol lands in the high byte,
        // then increment that byte.  On overflow, halve both counters to keep
        // the ratio while limiting unbounded growth.
        let rotated = self.packed.rotate_left(bit as u32 * 8);
        let (mut updated, overflow) = rotated.overflowing_add(0x100);
        if overflow {
            let mask = if rotated == 0xff01 { 0xff00u16 } else { 0x8100u16 };
            updated = ((rotated.wrapping_add(0x101)) >> 1) | mask;
        }
        BinProb { packed: updated.rotate_left(bit as u32 * 8) }
    }
}

// ---------------------------------------------------------------------------
// rANS core
// ---------------------------------------------------------------------------
//
// Word-aligned (u16) rANS, directly adapted from ryg_rans by Fabian 'ryg'
// Giesen (public domain, 2014).  Original: rans_byte.h in ryg/ryg_rans.
//
// The only structural difference from the byte variant: emission granularity
// is one u16 word instead of one byte.  Every comment below that says "byte"
// in the original has been updated to "word" accordingly.
//
// Scale is fixed at SCALE_BITS=8 (M=256), matching our binary probability
// model.  The normalization interval is [RANS_L, RANS_L * M) = [2^16, 2^24).
//
// Because rANS encodes in reverse (last symbol first), we buffer symbols on a
// stack and drain in one pass.  Two independent states are interleaved so a
// superscalar CPU can execute them in parallel (noted as a valid technique in
// the ryg_rans header comments).

// Lower bound of the normalization interval.
// ryg uses RANS_BYTE_L = 1<<23 (byte version); we use 1<<16 (word version).
const RANS_L: u32 = 1 << 16;
const SCALE_BITS: u32 = 8;

// rANS state — a single u32, exactly as in ryg_rans.
type RansState = u32;

// RansEncInit
#[inline]
fn rans_enc_init() -> RansState { RANS_L }

// RansDecInit — reads two LE u16 words that form a u32 state.
// (ryg reads four LE bytes; we read two LE u16s — same bit layout.)
#[inline]
fn rans_dec_init(src: &mut impl Read) -> Result<RansState> {
    let lo = read_u16_le(src)? as u32;
    let hi = read_u16_le(src)? as u32;
    Ok(lo | (hi << 16))
}

// RansEncFlush — push the two u16 halves of the state onto the output deque.
// (ryg writes four bytes to a backward pointer; we push_front to a deque so
// the words appear in the right order when iterated forward.)
#[inline]
fn rans_enc_flush(r: RansState, out: &mut VecDeque<u16>) {
    out.push_front((r >> 16) as u16);
    out.push_front((r & 0xffff) as u16);
}

// RansDecGet — returns the current cumulative frequency slot.
#[inline]
fn rans_dec_get(r: RansState) -> u32 {
    r & ((1 << SCALE_BITS) - 1)
}

// RansEncRenorm — renormalize before encoding a symbol with the given freq.
// Emits one u16 word if x is out of range (at most one for our parameters).
#[inline]
fn rans_enc_renorm(mut r: RansState, out: &mut VecDeque<u16>, freq: NonZeroU32) -> RansState {
    // x_max: exclusive upper bound of the pre-normalization interval.
    // ryg: x_max = ((RANS_BYTE_L >> scale_bits) << 8) * freq
    // word: x_max = ((RANS_L     >> SCALE_BITS) << 16) * freq
    let x_max = ((RANS_L >> SCALE_BITS) << 16) * freq.get();
    if r >= x_max {
        out.push_front(r as u16); // emit low word
        r >>= 16;
        debug_assert!(r < x_max);
    }
    r
}

// RansEncPut — encode one symbol (start, freq) into state r.
// ryg: *r = ((x / freq) << scale_bits) + (x % freq) + start
#[inline]
fn rans_enc_put(r: &mut RansState, out: &mut VecDeque<u16>, start: u32, freq: NonZeroU32) {
    *r = rans_enc_renorm(*r, out, freq);
    *r = (*r / freq) << SCALE_BITS | (*r % freq) + start;
}

// RansDecAdvance — decode one symbol and renormalize.
// ryg: x = freq*(x>>scale_bits) + (x&mask) - start; refill if x < RANS_L
#[inline]
fn rans_dec_advance(r: &mut RansState, src: &mut impl Read, start: u32, freq: NonZeroU32) -> Result<()> {
    let mask = (1 << SCALE_BITS) - 1;
    *r = freq.get() * (*r >> SCALE_BITS) + (*r & mask) - start;
    if *r < RANS_L {
        *r = (*r << 16) | read_u16_le(src)? as u32;
        debug_assert!(*r >= RANS_L);
    }
    Ok(())
}

#[inline]
fn read_u16_le(r: &mut impl Read) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

/// Map (bit, p_zero) → (CDF start, frequency) in [0, 256).
#[inline]
fn to_range(bit: bool, p_zero: NonZeroU8) -> (u32, NonZeroU32) {
    if bit {
        (p_zero.get() as u32, NonZeroU32::new(256 - p_zero.get() as u32).unwrap())
    } else {
        (0, NonZeroU32::from(p_zero))
    }
}

// ---------------------------------------------------------------------------
// RansEncoder
// ---------------------------------------------------------------------------

// Number of symbols buffered per drain.  Decoder must use the same value.
const BATCH: usize = 16386;

#[derive(Clone, Copy, Default)]
struct Pending {
    bit: bool,
    p: u8, // p_zero snapshot captured at coding time
}

/// Streaming adaptive rANS encoder.
///
/// Symbols are pushed onto a stack (last in, first encoded — rANS is a LIFO
/// coder).  Every [`BATCH`] symbols the stack is drained: two interleaved
/// states encode the batch in reverse and the resulting u16 words are written
/// to the sink in forward order.
pub struct RansEncoder<W> {
    sink: W,
    stack: Box<[Pending; BATCH]>,
    sp: usize, // stack pointer, counts down from BATCH
}

impl<W: Write> RansEncoder<W> {
    pub fn new(writer: W) -> Self {
        RansEncoder {
            sink: writer,
            stack: Box::new([Pending::default(); BATCH]),
            sp: BATCH,
        }
    }

    #[cold]
    fn drain(&mut self) -> Result<()> {
        let mut r0: RansState = rans_enc_init();
        let mut r1: RansState = rans_enc_init();
        let mut out: VecDeque<u16> = VecDeque::new();

        debug_assert!(self.sp < BATCH);

        // Walk the stack from bottom to top (reverse symbol order).
        // If the batch is odd-sized, encode the first symbol alone then swap
        // states so the remaining pairs stay aligned between r0 and r1.
        let mut i = self.sp;
        if i & 1 != 0 {
            let s = self.stack[i];
            let pz = NonZeroU8::new(s.p).unwrap_or(NonZeroU8::MIN);
            let (start, freq) = to_range(s.bit, pz);
            rans_enc_put(&mut r0, &mut out, start, freq);
            i += 1;
            std::mem::swap(&mut r0, &mut r1);
        }
        while i < BATCH {
            let s0 = self.stack[i];
            let s1 = self.stack[i + 1];
            let pz0 = NonZeroU8::new(s0.p).unwrap_or(NonZeroU8::MIN);
            let pz1 = NonZeroU8::new(s1.p).unwrap_or(NonZeroU8::MIN);
            let (st0, fr0) = to_range(s0.bit, pz0);
            let (st1, fr1) = to_range(s1.bit, pz1);
            rans_enc_put(&mut r0, &mut out, st0, fr0);
            rans_enc_put(&mut r1, &mut out, st1, fr1);
            i += 2;
        }
        rans_enc_flush(r0, &mut out);
        rans_enc_flush(r1, &mut out);

        for word in &out {
            self.sink.write_all(&word.to_le_bytes())?;
        }
        self.sp = BATCH;
        Ok(())
    }

    #[inline]
    pub fn put(&mut self, bit: bool, ctx: &mut BinProb) -> Result<()> {
        let p = ctx.prob_zero();
        *ctx = ctx.observe(bit);
        if self.sp == 0 { self.drain()?; }
        self.sp -= 1;
        self.stack[self.sp] = Pending { bit, p: p.get() };
        Ok(())
    }

    #[inline]
    pub fn put_bypass(&mut self, bit: bool) -> Result<()> {
        if self.sp == 0 { self.drain()?; }
        self.sp -= 1;
        self.stack[self.sp] = Pending { bit, p: 128 };
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.drain()
    }
}

// ---------------------------------------------------------------------------
// RansDecoder
// ---------------------------------------------------------------------------

/// Streaming adaptive rANS decoder that mirrors [`RansEncoder`].
pub struct RansDecoder<R> {
    src: R,
    r0: RansState,
    r1: RansState,
    count: usize,
}

impl<R: Read> RansDecoder<R> {
    pub fn new(mut reader: R) -> Result<Self> {
        let r0 = rans_dec_init(&mut reader)?;
        let r1 = rans_dec_init(&mut reader)?;
        Ok(RansDecoder { src: reader, r0, r1, count: 0 })
    }

    /// Reload both states after every BATCH symbols (mirrors encoder drain cadence).
    #[inline]
    fn maybe_reload(&mut self) -> Result<()> {
        if self.count == BATCH {
            self.count = 0;
            self.r0 = rans_dec_init(&mut self.src)?;
            self.r1 = rans_dec_init(&mut self.src)?;
        }
        self.count += 1;
        Ok(())
    }

    #[inline]
    pub fn get(&mut self, ctx: &mut BinProb) -> Result<bool> {
        self.maybe_reload()?;
        // Alternate r0/r1 each symbol to mirror the interleaved encoder.
        let mut r = self.r0;
        self.r0 = self.r1;

        let p = ctx.prob_zero();
        let bit = rans_dec_get(r) >= p.get() as u32;
        *ctx = ctx.observe(bit);

        let (start, freq) = to_range(bit, p);
        rans_dec_advance(&mut r, &mut self.src, start, freq)?;
        self.r1 = r;
        Ok(bit)
    }

    #[inline]
    pub fn get_bypass(&mut self) -> Result<bool> {
        self.maybe_reload()?;
        let mut r = self.r0;
        self.r0 = self.r1;

        // Bypass: equiprobable, split at 128 (half of M=256).
        let start = r & 0x80;
        rans_dec_advance(&mut r, &mut self.src, start, NonZeroU32::new(128).unwrap())?;
        self.r1 = r;
        Ok(start != 0)
    }
}

