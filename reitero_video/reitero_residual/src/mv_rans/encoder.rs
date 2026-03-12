use std::{cell::RefCell, rc::Rc};

use reitero_video_common::rans::{RansEncoder, BinProb};

use crate::mv_predictor::{mv_mode_context, MvMode};

use super::{
    MvCodedBlock, Subpel, CANDIDATE_IDX_DEPTH, CANDIDATE_SIZE, MODE_CTXS, MV_CLASS_TREE_DEPTH,
    MV_MAX_MAG_BITS, mv_class_base, mv_class_bits, mv_class_from_magnitude,
};

// Local helpers specific to encoding
fn encode_class_symbol(
    writer: &mut RansEncoder<super::SharedBuffer>,
    ctx: &mut [BinProb; MV_CLASS_TREE_DEPTH],
    class: u8,
) {
    let mut lo = 0i32;
    let mut hi = super::MV_MAX_CLASS as i32;
    let mut depth = 0usize;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let go_high = class as i32 > mid;
        writer
            .put(go_high, &mut ctx[depth])
            .expect("RANS writer failure on class tree");
        if go_high {
            lo = mid + 1;
        } else {
            hi = mid;
        }
        depth += 1;
    }
}

fn encode_candidate_index(
    writer: &mut RansEncoder<super::SharedBuffer>,
    ctx: &mut [BinProb; CANDIDATE_IDX_DEPTH],
    index: u8,
) {
    let mut lo = 0i32;
    let mut hi = CANDIDATE_SIZE as i32;
    let mut depth = 0usize;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        let go_high = (index as i32) >= mid;
        writer
            .put(go_high, &mut ctx[depth])
            .expect("RANS writer failure on candidate tree");
        if go_high {
            lo = mid;
        } else {
            hi = mid;
        }
        depth += 1;
    }
}

/// RANS encoder for motion vectors
pub struct MvRansEncoder {
    skip_ctx: BinProb,
    mode_is_new_ctx: [BinProb; MODE_CTXS],
    mode_is_nonzero_ctx: [BinProb; MODE_CTXS],
    mode_is_near_ctx: [BinProb; MODE_CTXS],
    mode_is_temporal_ctx: [BinProb; MODE_CTXS],
    mode_is_top_right_ctx: [BinProb; MODE_CTXS],
    mode_is_top_left_ctx: [BinProb; MODE_CTXS],
    // NEW base selector decision tree (5-way): nearest/near/top-right/top-left/temporal.
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

impl MvRansEncoder {
    pub fn new() -> Self {
        Self {
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

    fn encode_frame_to_writer(
        &mut self,
        writer: &mut RansEncoder<super::SharedBuffer>,
        blocks: &[MvCodedBlock],
        blocks_w: usize,
        blocks_h: usize,
    ) {
        assert_eq!(blocks.len(), blocks_w * blocks_h);
        let mut modes_so_far: Vec<MvMode> = Vec::with_capacity(blocks.len());
        let mut hits_so_far: Vec<bool> = Vec::with_capacity(blocks.len());

        for (i, block) in blocks.iter().enumerate() {
            let bx = i % blocks_w;
            let by = i / blocks_w;
            let skip = block.skip;
            writer
                .put(skip, &mut self.skip_ctx)
                .expect("RANS writer failure on skip flag");

            let ctx = mv_mode_context(&modes_so_far, blocks_w, bx, by);
            self.encode_mode(writer, block.mode, ctx);

            if block.mode == MvMode::New {
                let b = block.new_base;
                let is_nearest = b == 0;
                writer
                    .put(is_nearest, &mut self.new_base_is_nearest_ctx[ctx])
                    .expect("RANS writer failure on NEW base split (nearest)");
                if !is_nearest {
                    let is_near = b == 1;
                    writer
                        .put(is_near, &mut self.new_base_is_near_ctx[ctx])
                        .expect("RANS writer failure on NEW base split (near)");
                    if !is_near {
                        let is_top_right = b == 2;
                        writer
                            .put(is_top_right, &mut self.new_base_is_top_right_ctx[ctx])
                            .expect("RANS writer failure on NEW base split (top-right)");
                        if !is_top_right {
                            let is_top_left = b == 3;
                            writer
                                .put(is_top_left, &mut self.new_base_is_top_left_ctx[ctx])
                                .expect("RANS writer failure on NEW base split (top-left)");
                        }
                    }
                }

                let mut encoded_fractional_via_candidate = false;
                if !self.candidates.is_empty() {
                    let candidate_idx = self
                        .candidates
                        .iter()
                        .position(|&p| p == (block.delta_x, block.delta_y, block.subpel_x, block.subpel_y));
                    let hit = candidate_idx.is_some();

                    let left_hit = if bx > 0 { hits_so_far[i - 1] } else { false };
                    let top_hit = if by > 0 { hits_so_far[i - blocks_w] } else { false };
                    let hit_ctx_idx = (left_hit as usize) + (top_hit as usize);

                    writer
                        .put(hit, &mut self.candidate_hit_ctx[hit_ctx_idx])
                        .expect("RANS writer failure on candidate hit");

                    if let Some(idx) = candidate_idx {
                        encode_candidate_index(writer, &mut self.candidate_idx_ctx, idx as u8);
                        encoded_fractional_via_candidate = true;
                    } else {
                        Self::encode_component(
                            writer,
                            block.delta_x,
                            &mut self.class_ctx_x,
                            &mut self.sign_ctx_x,
                            &mut self.mag_bit_ctx_x,
                        );
                        Self::encode_component(
                            writer,
                            block.delta_y,
                            &mut self.class_ctx_y,
                            &mut self.sign_ctx_y,
                            &mut self.mag_bit_ctx_y,
                        );
                    }
                    hits_so_far.push(hit);
                } else {
                    Self::encode_component(
                        writer,
                        block.delta_x,
                        &mut self.class_ctx_x,
                        &mut self.sign_ctx_x,
                        &mut self.mag_bit_ctx_x,
                    );
                    Self::encode_component(
                        writer,
                        block.delta_y,
                        &mut self.class_ctx_y,
                        &mut self.sign_ctx_y,
                        &mut self.mag_bit_ctx_y,
                    );
                    hits_so_far.push(false);
                }

                if !encoded_fractional_via_candidate {
                    self.encode_fractional(writer, block.subpel_x, block.subpel_y);
                }
            } else {
                hits_so_far.push(false);
            }

            modes_so_far.push(block.mode);
        }
    }

    fn encode_mode(&mut self, writer: &mut RansEncoder<super::SharedBuffer>, mode: MvMode, ctx: usize) {
        let is_new = matches!(mode, MvMode::New);
        writer
            .put(is_new, &mut self.mode_is_new_ctx[ctx])
            .expect("RANS writer failure on mode split 0");
        if is_new {
            return;
        }

        let non_zero = !matches!(mode, MvMode::Zero);
        writer
            .put(non_zero, &mut self.mode_is_nonzero_ctx[ctx])
            .expect("RANS writer failure on mode split 1");
        if !non_zero {
            return;
        }

        let is_near = matches!(mode, MvMode::Near);
        writer
            .put(is_near, &mut self.mode_is_near_ctx[ctx])
            .expect("RANS writer failure on mode split 2");
        if is_near {
            return;
        }

        let is_temporal = matches!(mode, MvMode::Temporal);
        writer
            .put(is_temporal, &mut self.mode_is_temporal_ctx[ctx])
            .expect("RANS writer failure on mode split 3");
        if is_temporal {
            return;
        }

        let is_top_right = matches!(mode, MvMode::TopRight);
        writer
            .put(is_top_right, &mut self.mode_is_top_right_ctx[ctx])
            .expect("RANS writer failure on mode split 4");
        if is_top_right {
            return;
        }

        let is_top_left = matches!(mode, MvMode::TopLeft);
        writer
            .put(is_top_left, &mut self.mode_is_top_left_ctx[ctx])
            .expect("RANS writer failure on mode split 5");
    }

    fn encode_component(
        writer: &mut RansEncoder<super::SharedBuffer>,
        delta: i8,
        class_ctx: &mut [BinProb; MV_CLASS_TREE_DEPTH],
        sign_ctx: &mut BinProb,
        mag_ctx: &mut [BinProb; MV_MAX_MAG_BITS],
    ) {
        let mag = (i16::from(delta).abs()) as u16;
        let class = mv_class_from_magnitude(mag);
        encode_class_symbol(writer, class_ctx, class);
        if class == 0 {
            return;
        }

        let sign = delta < 0;
        writer
            .put(sign, sign_ctx)
            .expect("RANS writer failure on component sign");

        let base = mv_class_base(class);
        let bits = mv_class_bits(class);
        let offset = mag - base;
        for bit in 0..bits {
            let bit_val = (offset >> bit) & 1 != 0;
            writer
                .put(bit_val, &mut mag_ctx[bit as usize])
                .expect("RANS writer failure on component mag bit");
        }
    }

    fn encode_fractional(&mut self, writer: &mut RansEncoder<super::SharedBuffer>, sub_x: Subpel, sub_y: Subpel) {
        if matches!(sub_x, Subpel::MinusHalf) || matches!(sub_y, Subpel::MinusHalf) {
            panic!("raw -0.5 subpixel is disallowed");
        }
        let has_frac_x = !matches!(sub_x, Subpel::Zero);
        writer
            .put(has_frac_x, &mut self.sub_x_has_ctx)
            .expect("RANS writer failure on frac-x flag");
        // No longer store sign; presence implies +0.5 during decode

        let has_frac_y = !matches!(sub_y, Subpel::Zero);
        writer
            .put(has_frac_y, &mut self.sub_y_has_ctx)
            .expect("RANS writer failure on frac-y flag");
        // No longer store sign; presence implies +0.5 during decode
    }

    pub fn encode_frame_and_get_data(
        &mut self,
        blocks: &[MvCodedBlock],
        blocks_w: usize,
        blocks_h: usize,
    ) -> Vec<u8> {
        let buf = Rc::new(RefCell::new(Vec::new()));
        let writer_handle = super::SharedBuffer(Rc::clone(&buf));
        let mut writer = RansEncoder::new(writer_handle);

        self.encode_frame_to_writer(&mut writer, blocks, blocks_w, blocks_h);

        writer.finish().unwrap();

        self.update_candidates(blocks);

        buf.borrow().clone()
    }

    fn update_candidates(&mut self, blocks: &[MvCodedBlock]) {
        for count in self.candidate_counts.values_mut() {
            *count = (*count * 7) / 8;
        }
        self.total_new_history = (self.total_new_history * 7) / 8;

        for block in blocks {
            if block.mode == MvMode::New {
                // Disallow seeding candidates with raw -0.5 fractional values
                if matches!(block.subpel_x, Subpel::MinusHalf) || matches!(block.subpel_y, Subpel::MinusHalf) {
                    panic!("internal invariant violated: -0.5 subpixel should never occur");
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
}
