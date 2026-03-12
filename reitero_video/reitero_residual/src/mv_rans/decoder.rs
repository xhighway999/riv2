use std::io::Cursor;

use reitero_video_common::rans::{RansDecoder, BinProb};

use crate::mv_predictor::{mv_mode_context, MvMode};

use super::{
    MvCodedBlock, Subpel, CANDIDATE_IDX_DEPTH, CANDIDATE_SIZE, MODE_CTXS, MV_CLASS_TREE_DEPTH,
    MV_MAX_MAG_BITS, mv_class_base, mv_class_bits,
};

fn decode_class_symbol(
    reader: &mut RansDecoder<Cursor<Vec<u8>>>,
    ctx: &mut [BinProb; MV_CLASS_TREE_DEPTH],
) -> u8 {
    let mut lo = 0i32;
    let mut hi = super::MV_MAX_CLASS as i32;
    let mut depth = 0usize;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let go_high = reader
            .get(&mut ctx[depth])
            .expect("RANS reader failure on class tree");
        if go_high {
            lo = mid + 1;
        } else {
            hi = mid;
        }
        depth += 1;
    }
    lo as u8
}

fn decode_candidate_index(
    reader: &mut RansDecoder<Cursor<Vec<u8>>>,
    ctx: &mut [BinProb; CANDIDATE_IDX_DEPTH],
) -> u8 {
    let mut lo = 0i32;
    let mut hi = CANDIDATE_SIZE as i32;
    let mut depth = 0usize;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        let go_high = reader
            .get(&mut ctx[depth])
            .expect("RANS reader failure on candidate tree");
        if go_high {
            lo = mid;
        } else {
            hi = mid;
        }
        depth += 1;
    }
    lo as u8
}

/// RANS decoder for motion vectors
pub struct MvRansDecoder {
    reader: Option<RansDecoder<Cursor<Vec<u8>>>>,
    // Contexts for decoding components
    skip_ctx: BinProb,
    mode_is_new_ctx: [BinProb; MODE_CTXS],
    mode_is_nonzero_ctx: [BinProb; MODE_CTXS],
    mode_is_near_ctx: [BinProb; MODE_CTXS],
    mode_is_temporal_ctx: [BinProb; MODE_CTXS],
    mode_is_top_right_ctx: [BinProb; MODE_CTXS],
    mode_is_top_left_ctx: [BinProb; MODE_CTXS],
    new_base_is_nearest_ctx: [BinProb; MODE_CTXS],
    new_base_is_near_ctx: [BinProb; MODE_CTXS],
    new_base_is_top_right_ctx: [BinProb; MODE_CTXS],
    new_base_is_top_left_ctx: [BinProb; MODE_CTXS],
    class_ctx_x: [BinProb; MV_CLASS_TREE_DEPTH],
    class_ctx_y: [BinProb; MV_CLASS_TREE_DEPTH],
    sign_ctx_x: BinProb,
    sign_ctx_y: BinProb,
    mag_bit_ctx_x: [BinProb; MV_MAX_MAG_BITS],
    mag_bit_ctx_y: [BinProb; MV_MAX_MAG_BITS],
    sub_x_has_ctx: BinProb,
    sub_y_has_ctx: BinProb,
    candidate_hit_ctx: [BinProb; 3],
    candidate_idx_ctx: [BinProb; CANDIDATE_IDX_DEPTH],
    candidates: Vec<(i8, i8, Subpel, Subpel)>,
    candidate_counts: std::collections::HashMap<(i8, i8, Subpel, Subpel), u32>,
    total_new_history: u32,
}

impl MvRansDecoder {
    pub fn new() -> Self {
        Self {
            reader: None,
            skip_ctx: BinProb::default(),
            mode_is_new_ctx: [BinProb::default(); MODE_CTXS],
            mode_is_nonzero_ctx: [BinProb::default(); MODE_CTXS],
            mode_is_near_ctx: [BinProb::default(); MODE_CTXS],
            mode_is_temporal_ctx: [BinProb::default(); MODE_CTXS],
            mode_is_top_right_ctx: [BinProb::default(); MODE_CTXS],
            mode_is_top_left_ctx: [BinProb::default(); MODE_CTXS],
            new_base_is_nearest_ctx: [BinProb::default(); MODE_CTXS],
            new_base_is_near_ctx: [BinProb::default(); MODE_CTXS],
            new_base_is_top_right_ctx: [BinProb::default(); MODE_CTXS],
            new_base_is_top_left_ctx: [BinProb::default(); MODE_CTXS],
            class_ctx_x: [BinProb::default(); MV_CLASS_TREE_DEPTH],
            class_ctx_y: [BinProb::default(); MV_CLASS_TREE_DEPTH],
            sign_ctx_x: BinProb::default(),
            sign_ctx_y: BinProb::default(),
            mag_bit_ctx_x: [BinProb::default(); MV_MAX_MAG_BITS],
            mag_bit_ctx_y: [BinProb::default(); MV_MAX_MAG_BITS],
            sub_x_has_ctx: BinProb::default(),
            sub_y_has_ctx: BinProb::default(),
            candidate_hit_ctx: [BinProb::default(); 3],
            candidate_idx_ctx: [BinProb::default(); CANDIDATE_IDX_DEPTH],
            candidates: Vec::new(),
            candidate_counts: std::collections::HashMap::new(),
            total_new_history: 0,
        }
    }

    pub fn reset_contexts(&mut self) {
        self.skip_ctx = BinProb::default();
        self.mode_is_new_ctx = [BinProb::default(); MODE_CTXS];
        self.mode_is_nonzero_ctx = [BinProb::default(); MODE_CTXS];
        self.mode_is_near_ctx = [BinProb::default(); MODE_CTXS];
        self.mode_is_temporal_ctx = [BinProb::default(); MODE_CTXS];
        self.mode_is_top_right_ctx = [BinProb::default(); MODE_CTXS];
        self.mode_is_top_left_ctx = [BinProb::default(); MODE_CTXS];
        self.new_base_is_nearest_ctx = [BinProb::default(); MODE_CTXS];
        self.new_base_is_near_ctx = [BinProb::default(); MODE_CTXS];
        self.new_base_is_top_right_ctx = [BinProb::default(); MODE_CTXS];
        self.new_base_is_top_left_ctx = [BinProb::default(); MODE_CTXS];
        self.class_ctx_x = [BinProb::default(); MV_CLASS_TREE_DEPTH];
        self.class_ctx_y = [BinProb::default(); MV_CLASS_TREE_DEPTH];
        self.sign_ctx_x = BinProb::default();
        self.sign_ctx_y = BinProb::default();
        self.mag_bit_ctx_x = [BinProb::default(); MV_MAX_MAG_BITS];
        self.mag_bit_ctx_y = [BinProb::default(); MV_MAX_MAG_BITS];
        self.sub_x_has_ctx = BinProb::default();
        self.sub_y_has_ctx = BinProb::default();
        self.candidate_hit_ctx = [BinProb::default(); 3];
        self.candidate_idx_ctx = [BinProb::default(); CANDIDATE_IDX_DEPTH];
        self.candidates.clear();
        self.candidate_counts.clear();
        self.total_new_history = 0;
    }

    pub fn consume_frame(&mut self, bytes: &[u8]) {
        let cursor = Cursor::new(bytes.to_vec());
        self.reader = Some(RansDecoder::new(cursor).expect("RANS Init Failed"));
    }

    pub fn decode_frame(&mut self, blocks_w: usize, blocks_h: usize) -> Vec<MvCodedBlock> {
        let num_blocks = blocks_w * blocks_h;
        let mut reader = self
            .reader
            .take()
            .expect("Decoder called before consume_frame()");

        let mut result = Vec::with_capacity(num_blocks);
        let mut modes_so_far: Vec<MvMode> = Vec::with_capacity(num_blocks);
        let mut hits_so_far: Vec<bool> = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let bx = block_idx % blocks_w;
            let by = block_idx / blocks_w;

            let skip = reader
                .get(&mut self.skip_ctx)
                .expect("Failed to read skip flag");
            let mut subpel_x = Subpel::Zero;
            let mut subpel_y = Subpel::Zero;

            let ctx = mv_mode_context(&modes_so_far, blocks_w, bx, by);
            let mode = self.decode_mode(&mut reader, ctx);

            let mut new_base = 0u8;
            let mut delta_x = 0i8;
            let mut delta_y = 0i8;
            let mut decoded_fractional_via_candidate = false;

            if mode == MvMode::New {
                let is_nearest = reader
                    .get(&mut self.new_base_is_nearest_ctx[ctx])
                    .expect("Failed to read NEW base split (nearest)");
                if is_nearest {
                    new_base = 0;
                } else {
                    let is_near = reader
                        .get(&mut self.new_base_is_near_ctx[ctx])
                        .expect("Failed to read NEW base split (near)");
                    if is_near {
                        new_base = 1;
                    } else {
                        let is_top_right = reader
                            .get(&mut self.new_base_is_top_right_ctx[ctx])
                            .expect("Failed to read NEW base split (top-right)");
                        if is_top_right {
                            new_base = 2;
                        } else {
                            let is_top_left = reader
                                .get(&mut self.new_base_is_top_left_ctx[ctx])
                                .expect("Failed to read NEW base split (top-left)");
                            new_base = if is_top_left { 3 } else { 4 };
                        }
                    }
                }

                if !self.candidates.is_empty() {
                    let left_hit = if bx > 0 { hits_so_far[block_idx - 1] } else { false };
                    let top_hit = if by > 0 { hits_so_far[block_idx - blocks_w] } else { false };
                    let hit_ctx_idx = (left_hit as usize) + (top_hit as usize);

                    let hit = reader
                        .get(&mut self.candidate_hit_ctx[hit_ctx_idx])
                        .expect("Failed to read candidate hit");
                    if hit {
                        let idx = decode_candidate_index(&mut reader, &mut self.candidate_idx_ctx);
                        if (idx as usize) < self.candidates.len() {
                            let (dx, dy, spx, spy) = self.candidates[idx as usize];
                            delta_x = dx;
                            delta_y = dy;
                            subpel_x = spx;
                            subpel_y = spy;
                            decoded_fractional_via_candidate = true;
                        } else {
                            delta_x = 0;
                            delta_y = 0;
                        }
                    } else {
                        delta_x = Self::decode_component(
                            &mut reader,
                            &mut self.class_ctx_x,
                            &mut self.sign_ctx_x,
                            &mut self.mag_bit_ctx_x,
                        );
                        delta_y = Self::decode_component(
                            &mut reader,
                            &mut self.class_ctx_y,
                            &mut self.sign_ctx_y,
                            &mut self.mag_bit_ctx_y,
                        );
                    }
                    hits_so_far.push(hit);
                } else {
                    delta_x = Self::decode_component(
                        &mut reader,
                        &mut self.class_ctx_x,
                        &mut self.sign_ctx_x,
                        &mut self.mag_bit_ctx_x,
                    );
                    delta_y = Self::decode_component(
                        &mut reader,
                        &mut self.class_ctx_y,
                        &mut self.sign_ctx_y,
                        &mut self.mag_bit_ctx_y,
                    );
                    hits_so_far.push(false);
                }

                if !decoded_fractional_via_candidate {
                    let (spx, spy) = self.decode_fractional(&mut reader);
                    subpel_x = spx;
                    subpel_y = spy;
                }
            } else {
                hits_so_far.push(false);
            }

            result.push(MvCodedBlock {
                mode,
                new_base,
                delta_x,
                delta_y,
                subpel_x,
                subpel_y,
                skip,
            });
            modes_so_far.push(mode);
        }

        self.reader = Some(reader);

        self.update_candidates(&result);

        result
    }

    fn update_candidates(&mut self, blocks: &[MvCodedBlock]) {
        for count in self.candidate_counts.values_mut() {
            *count = (*count * 7) / 8;
        }
        self.total_new_history = (self.total_new_history * 7) / 8;

        for block in blocks {
            if block.mode == MvMode::New {
                // Disallow seeding candidates with -0.5 fractional values from decoded frames
                if matches!(block.subpel_x, Subpel::MinusHalf) || matches!(block.subpel_y, Subpel::MinusHalf) {
                    continue;
                }
                *self
                    .candidate_counts
                    .entry((block.delta_x, block.delta_y, block.subpel_x, block.subpel_y))
                    .or_insert(0) += 1;
                self.total_new_history += 1;
            }
        }

        self.candidate_counts.retain(|_, &mut c| c > 0);

        if self.total_new_history < 100 {
            self.candidates.clear();
            return;
        }

        let mut sorted: Vec<_> = self
            .candidate_counts
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let threshold = self.total_new_history / 100;

        self.candidates = sorted
            .into_iter()
            .filter(|&(_, count)| count > threshold)
            .take(CANDIDATE_SIZE)
            .map(|(k, _)| k)
            .collect();
    }

    fn decode_mode(&mut self, reader: &mut RansDecoder<Cursor<Vec<u8>>>, ctx: usize) -> MvMode {
        let is_new = reader
            .get(&mut self.mode_is_new_ctx[ctx])
            .expect("Failed to read mode split 0");
        if is_new {
            return MvMode::New;
        }

        let non_zero = reader
            .get(&mut self.mode_is_nonzero_ctx[ctx])
            .expect("Failed to read mode split 1");
        if !non_zero {
            return MvMode::Zero;
        }

        let is_near = reader
            .get(&mut self.mode_is_near_ctx[ctx])
            .expect("Failed to read mode split 2");
        if is_near {
            return MvMode::Near;
        }

        let is_temporal = reader
            .get(&mut self.mode_is_temporal_ctx[ctx])
            .expect("Failed to read mode split 3");
        if is_temporal {
            return MvMode::Temporal;
        }

        let is_top_right = reader
            .get(&mut self.mode_is_top_right_ctx[ctx])
            .expect("Failed to read mode split 4");
        if is_top_right {
            return MvMode::TopRight;
        }

        let is_top_left = reader
            .get(&mut self.mode_is_top_left_ctx[ctx])
            .expect("Failed to read mode split 5");
        if is_top_left {
            return MvMode::TopLeft;
        }

        MvMode::Nearest
    }

    fn decode_component(
        reader: &mut RansDecoder<Cursor<Vec<u8>>>,
        class_ctx: &mut [BinProb; MV_CLASS_TREE_DEPTH],
        sign_ctx: &mut BinProb,
        mag_ctx: &mut [BinProb; MV_MAX_MAG_BITS],
    ) -> i8 {
        let class = decode_class_symbol(reader, class_ctx);
        if class == 0 {
            return 0;
        }

        let sign = reader.get(sign_ctx).expect("Failed to read component sign");

        let base = mv_class_base(class);
        let bits = mv_class_bits(class);
        let mut offset = 0u16;
        for bit in 0..bits {
            let bit_val = reader
                .get(&mut mag_ctx[bit as usize])
                .expect("Failed to read component mag bit");
            if bit_val {
                offset |= 1 << bit;
            }
        }
        let mag = base + offset;
        let value = mag as i16;
        let signed = if sign { -(value as i16) } else { value as i16 };
        signed.clamp(-128, 127) as i8
    }

    fn decode_fractional(&mut self, reader: &mut RansDecoder<Cursor<Vec<u8>>>) -> (Subpel, Subpel) {
        let mut subpel_x = Subpel::Zero;
        let has_frac_x = reader
            .get(&mut self.sub_x_has_ctx)
            .expect("Failed to read frac-x flag");
        if has_frac_x {
            // No longer read sign; presence implies +0.5
            subpel_x = Subpel::PlusHalf;
        }

        let mut subpel_y = Subpel::Zero;
        let has_frac_y = reader
            .get(&mut self.sub_y_has_ctx)
            .expect("Failed to read frac-y flag");
        if has_frac_y {
            // No longer read sign; presence implies +0.5
            subpel_y = Subpel::PlusHalf;
        }

        (subpel_x, subpel_y)
    }
}
