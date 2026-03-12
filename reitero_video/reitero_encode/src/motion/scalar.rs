use super::backend::{sad_block_halfpel_limit_luma, sad_block_int_limit};
use super::{BLOCK_SIZE, LumaPlane, MotionVector};

#[cfg(feature = "threads")]
use rayon::prelude::*;

const CARDINAL_OFFSETS: &[(i32, i32)] = &[(1, 0), (-1, 0), (0, 1), (0, -1)];
const HEX_OFFSETS: &[(i32, i32)] = &[(2, 0), (1, 2), (-1, 2), (-2, 0), (-1, -2), (1, -2)];

#[derive(Clone, Copy)]
struct IntSearchState {
    dx: i32,
    dy: i32,
    cost: i64,
}

impl IntSearchState {
    fn new(cost: i64) -> Self {
        Self { dx: 0, dy: 0, cost }
    }
}

#[inline]
fn median3(a: i32, b: i32, c: i32) -> i32 {
    // Return median of three values
    if a > b {
        if b > c {
            b
        } else if a > c {
            c
        } else {
            a
        }
    } else {
        if a > c {
            a
        } else if b > c {
            c
        } else {
            b
        }
    }
}

#[inline]
fn push_candidate(candidates: &mut Vec<(i32, i32)>, dx: i32, dy: i32, sr: i32) {
    if dx.abs() > sr || dy.abs() > sr {
        return;
    }
    if !candidates.iter().any(|&(cx, cy)| cx == dx && cy == dy) {
        candidates.push((dx, dy));
    }
}

#[cfg(not(feature = "threads"))]
fn gather_local_motion_candidates<'a>(
    mvs: &[MotionVector],
    prev_mvs: Option<&[MotionVector]>,
    blocks_w: usize,
    bx: usize,
    by: usize,
    sr: i32,
    scratch: &'a mut Vec<(i32, i32)>,
) -> ((i32, i32), (i32, i32), (i32, i32)) {
    // Gather predictors in order: Left, Top, Top-Right, Top-Left, Temporal
    let mut predictors: Vec<(i8, i8)> = Vec::with_capacity(5);
    let idx = by * blocks_w + bx;

    // Left
    if bx > 0 {
        let mv = mvs[idx - 1];
        predictors.push((mv.dx(), mv.dy()));
    }

    // Top
    if by > 0 {
        let mv = mvs[(by - 1) * blocks_w + bx];
        predictors.push((mv.dx(), mv.dy()));

        // Top-Right
        if bx + 1 < blocks_w {
            let mv = mvs[(by - 1) * blocks_w + (bx + 1)];
            predictors.push((mv.dx(), mv.dy()));
        }

        // Top-Left
        if bx > 0 {
            let mv = mvs[(by - 1) * blocks_w + (bx - 1)];
            predictors.push((mv.dx(), mv.dy()));
        }
    }

    // Temporal
    if let Some(prev) = prev_mvs {
        if prev.len() == mvs.len() {
            if let Some(mv) = prev.get(idx) {
                predictors.push((mv.dx(), mv.dy()));
            }
        }
    }

    // Deduplicate for Nearest/Near
    let mut unique_predictors = Vec::with_capacity(5);
    for &p in &predictors {
        if !unique_predictors.contains(&p) {
            unique_predictors.push(p);
        }
    }
    if unique_predictors.is_empty() {
        unique_predictors.push((0, 0));
    }
    if unique_predictors.len() == 1 {
        unique_predictors.push((0, 0));
    }

    let nearest = (unique_predictors[0].0 as i32, unique_predictors[0].1 as i32);
    let near = (unique_predictors[1].0 as i32, unique_predictors[1].1 as i32);

    // Calculate Median
    let left = if bx > 0 {
        let mv = mvs[idx - 1];
        (mv.dx() as i32, mv.dy() as i32)
    } else {
        (0, 0)
    };

    let top = if by > 0 {
        let mv = mvs[(by - 1) * blocks_w + bx];
        (mv.dx() as i32, mv.dy() as i32)
    } else {
        (0, 0)
    };

    let top_right = if by > 0 && bx + 1 < blocks_w {
        let mv = mvs[(by - 1) * blocks_w + (bx + 1)];
        (mv.dx() as i32, mv.dy() as i32)
    } else if by > 0 && bx > 0 {
        let mv = mvs[(by - 1) * blocks_w + (bx - 1)];
        (mv.dx() as i32, mv.dy() as i32)
    } else {
        (0, 0)
    };

    let median_x = median3(left.0, top.0, top_right.0);
    let median_y = median3(left.1, top.1, top_right.1);
    let median = (median_x, median_y);

    // Populate scratch (search candidates)
    scratch.clear();
    push_candidate(scratch, 0, 0, sr);
    for p in &predictors {
        push_candidate(scratch, p.0 as i32, p.1 as i32, sr);
    }

    (nearest, near, median)
}

#[inline]
fn block_origin(block_idx: usize, block_size: usize, limit: usize) -> i32 {
    let origin = block_idx * block_size;
    let clamped = if origin + block_size <= limit {
        origin
    } else {
        limit.saturating_sub(block_size)
    };
    clamped as i32
}

fn block_is_identical(
    curr: &LumaPlane,
    prev: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    if x0 < 0 || y0 < 0 {
        return false;
    }
    let x0 = x0 as usize;
    let y0 = y0 as usize;
    if x0 >= width || y0 >= height {
        return false;
    }
    if x0 + BLOCK_SIZE > width || y0 + BLOCK_SIZE > height {
        return false;
    }

    let curr_slice = curr.as_slice();
    let prev_slice = prev.as_slice();
    for row in 0..BLOCK_SIZE {
        let offset = (y0 + row) * width + x0;
        if curr_slice[offset..offset + BLOCK_SIZE]
            != prev_slice[offset..offset + BLOCK_SIZE]
        {
            return false;
        }
    }
    true
}

/// Compute motion vectors for the current frame against the previous one using
/// Hexagonal Search (integer) followed by half-pixel refinement (±0.5 in each
/// axis) around the best integer MV. The integer search now always evaluates
/// SAD so the hot loop remains simple and cheap.
pub fn hex_search_sad_with_scores_luma(
    width: usize,
    height: usize,
    search_range: u8,
    prev_mvs: Option<&[MotionVector]>,
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    zero_mv_threshold: i64,
    predictor_threshold: i64,
    lambda: f64,
) -> (Vec<MotionVector>, Vec<i64>) {
    hex_search_impl(
        width,
        height,
        search_range,
        prev_mvs,
        prev_luma,
        curr_luma,
        zero_mv_threshold,
        predictor_threshold,
        lambda,
    )
}

fn hex_search_impl(
    width: usize,
    height: usize,
    search_range: u8,
    prev_mvs: Option<&[MotionVector]>,
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    zero_mv_threshold: i64,
    predictor_threshold: i64,
    lambda: f64,
) -> (Vec<MotionVector>, Vec<i64>) {
    let blocks_w = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let blocks_h = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut mvs = vec![MotionVector::new(0, 0, 0, 0, false); blocks_w * blocks_h];
    let mut best_sads = vec![0i64; blocks_w * blocks_h];
    let sr = search_range as i32;

    #[cfg(feature = "threads")]
    {
        let num_threads = rayon::current_num_threads();

        // Create 4x more strips than threads to improve load balancing
        // This increases the number of boundary rows (slightly less efficiency) but keeps cores busy
        let target_strips = num_threads * 4;
        let rows_per_strip = ((blocks_h + target_strips - 1) / target_strips).max(1);
        let chunk_size = blocks_w * rows_per_strip;

        mvs.par_chunks_mut(chunk_size)
            .zip(best_sads.par_chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(strip_idx, (strip_mvs, strip_sads))| {
                let mut candidate_buf: Vec<(i32, i32)> = Vec::with_capacity(12);
                let strip_start_by = strip_idx * rows_per_strip;
                let strip_rows = strip_mvs.len() / blocks_w;

                // We iterate rows within the strip sequentially to allow using Top predictors
                for local_by in 0..strip_rows {
                    let global_by = strip_start_by + local_by;

                    // Split the strip slice to separate "processed rows" from "current row"
                    // This allows us to safely read predictors from the previous row
                    let (prev_rows, current_row_rest) = strip_mvs.split_at_mut(local_by * blocks_w);
                    let (current_row_mvs, _) = current_row_rest.split_at_mut(blocks_w);
                    let (current_row_sads, _) =
                        strip_sads[local_by * blocks_w..].split_at_mut(blocks_w);

                    for bx in 0..blocks_w {
                        let idx = global_by * blocks_w + bx;

                        // Gather predictors for cost estimation (and search candidates)
                        // Order must match mv_predictor.rs: Left, Top, Top-Right, Top-Left, Temporal
                        let mut predictors: Vec<(i8, i8)> = Vec::with_capacity(5);

                        // Left
                        if bx > 0 {
                            let mv = current_row_mvs[bx - 1];
                            predictors.push((mv.dx(), mv.dy()));
                        }

                        // Top, Top-Right, Top-Left
                        if local_by > 0 {
                            let prev_row_start = (local_by - 1) * blocks_w;
                            let prev_row = &prev_rows[prev_row_start..prev_row_start + blocks_w];

                            // Top
                            let mv = prev_row[bx];
                            predictors.push((mv.dx(), mv.dy()));

                            // Top-Right
                            if bx + 1 < blocks_w {
                                let mv = prev_row[bx + 1];
                                predictors.push((mv.dx(), mv.dy()));
                            }

                            // Top-Left
                            if bx > 0 {
                                let mv = prev_row[bx - 1];
                                predictors.push((mv.dx(), mv.dy()));
                            }
                        }

                        // Temporal
                        if let Some(prev) = prev_mvs {
                            if let Some(mv) = prev.get(idx) {
                                predictors.push((mv.dx(), mv.dy()));
                            }
                        }

                        // Deduplicate predictors to find Nearest/Near
                        let mut unique_predictors = Vec::with_capacity(5);
                        for &p in &predictors {
                            if !unique_predictors.contains(&p) {
                                unique_predictors.push(p);
                            }
                        }

                        // Pad with (0,0) if needed
                        if unique_predictors.is_empty() {
                            unique_predictors.push((0, 0));
                        }
                        if unique_predictors.len() == 1 {
                            unique_predictors.push((0, 0));
                        }

                        let nearest_i8 = unique_predictors[0];
                        let near_i8 = unique_predictors[1];
                        let nearest = (nearest_i8.0 as i32, nearest_i8.1 as i32);
                        let near = (near_i8.0 as i32, near_i8.1 as i32);

                        // Calculate Median
                        let left = if bx > 0 {
                            let mv = current_row_mvs[bx - 1];
                            (mv.dx() as i32, mv.dy() as i32)
                        } else {
                            (0, 0)
                        };

                        let top = if local_by > 0 {
                            let prev_row_start = (local_by - 1) * blocks_w;
                            let prev_row = &prev_rows[prev_row_start..prev_row_start + blocks_w];
                            let mv = prev_row[bx];
                            (mv.dx() as i32, mv.dy() as i32)
                        } else {
                            (0, 0)
                        };

                        let top_right = if local_by > 0 && bx + 1 < blocks_w {
                            let prev_row_start = (local_by - 1) * blocks_w;
                            let prev_row = &prev_rows[prev_row_start..prev_row_start + blocks_w];
                            let mv = prev_row[bx + 1];
                            (mv.dx() as i32, mv.dy() as i32)
                        } else if local_by > 0 && bx > 0 {
                            let prev_row_start = (local_by - 1) * blocks_w;
                            let prev_row = &prev_rows[prev_row_start..prev_row_start + blocks_w];
                            let mv = prev_row[bx - 1];
                            (mv.dx() as i32, mv.dy() as i32)
                        } else {
                            (0, 0)
                        };

                        let median_x = median3(left.0, top.0, top_right.0);
                        let median_y = median3(left.1, top.1, top_right.1);
                        let median = (median_x, median_y);

                        // Populate search candidates
                        candidate_buf.clear();
                        // Always add (0,0) first for search stability/bias
                        push_candidate(&mut candidate_buf, 0, 0, sr);
                        // Add all predictors
                        for p in &predictors {
                            push_candidate(&mut candidate_buf, p.0 as i32, p.1 as i32, sr);
                        }

                        let (mv, sad) = process_block_hex(
                            bx,
                            global_by,
                            width,
                            height,
                            sr,
                            &candidate_buf,
                            prev_luma,
                            curr_luma,
                            zero_mv_threshold,
                            predictor_threshold,
                            lambda,
                            nearest,
                            near,
                            median,
                        );
                        current_row_mvs[bx] = mv;
                        current_row_sads[bx] = sad;
                    }
                }
            });
    }

    #[cfg(not(feature = "threads"))]
    {
        let mut candidate_buf: Vec<(i32, i32)> = Vec::with_capacity(12);
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                let idx = by * blocks_w + bx;

                let (nearest, near, median) = gather_local_motion_candidates(
                    &mvs,
                    prev_mvs,
                    blocks_w,
                    bx,
                    by,
                    sr,
                    &mut candidate_buf,
                );

                let (mv, sad) = process_block_hex(
                    bx,
                    by,
                    width,
                    height,
                    sr,
                    &candidate_buf,
                    prev_luma,
                    curr_luma,
                    zero_mv_threshold,
                    predictor_threshold,
                    lambda,
                    nearest,
                    near,
                    median,
                );
                mvs[idx] = mv;
                best_sads[idx] = sad;
            }
        }
    }

    (mvs, best_sads)
}

// Helper for cost estimation
#[inline]
fn estimate_mv_cost(
    dx: i32,
    dy: i32,
    nearest: (i32, i32),
    near: (i32, i32),
    median: (i32, i32),
    lambda: f64,
) -> i64 {
    if lambda == 0.0 {
        return 0;
    }

    // Lambda is derived from QP but is often too small relative to the SAD scale (especially RGB SAD).
    // We apply a scaling factor to make bit costs meaningful enough to override small SAD noise.
    // A factor of 2.0 means 1 bit is worth ~5 SAD units at lambda=0.5.
    let scale = 2.0;
    let mut best_cost = i64::MAX;

    // Check Nearest
    if dx == nearest.0 && dy == nearest.1 {
        let cost = (lambda * scale * 1.5) as i64;
        if cost < best_cost {
            best_cost = cost;
        }
    }

    // Check Near
    if dx == near.0 && dy == near.1 {
        let cost = (lambda * scale * 1.5) as i64;
        if cost < best_cost {
            best_cost = cost;
        }
    }

    // Check Median
    if dx == median.0 && dy == median.1 {
        let cost = (lambda * scale * 1.5) as i64;
        if cost < best_cost {
            best_cost = cost;
        }
    }

    // Check Zero (explicit mode)
    if dx == 0 && dy == 0 {
        let cost = (lambda * scale * 0.5) as i64;
        if cost < best_cost {
            best_cost = cost;
        }
    }

    // Check New (ref Nearest)
    {
        let diff_x = (dx - nearest.0).abs();
        let diff_y = (dy - nearest.1).abs();
        let bits_x = if diff_x == 0 {
            1.0
        } else {
            2.0 * (diff_x as f64).log2() + 2.0
        };
        let bits_y = if diff_y == 0 {
            1.0
        } else {
            2.0 * (diff_y as f64).log2() + 2.0
        };
        let cost = (lambda * scale * (6.0 + bits_x + bits_y)) as i64;
        if cost < best_cost {
            best_cost = cost;
        }
    }

    // Check New (ref Near)
    {
        let diff_x = (dx - near.0).abs();
        let diff_y = (dy - near.1).abs();
        let bits_x = if diff_x == 0 {
            1.0
        } else {
            2.0 * (diff_x as f64).log2() + 2.0
        };
        let bits_y = if diff_y == 0 {
            1.0
        } else {
            2.0 * (diff_y as f64).log2() + 2.0
        };
        let cost = (lambda * scale * (6.0 + bits_x + bits_y)) as i64;
        if cost < best_cost {
            best_cost = cost;
        }
    }

    best_cost
}

fn process_block_hex(
    bx: usize,
    by: usize,
    width: usize,
    height: usize,
    sr: i32,
    candidates: &[(i32, i32)],
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    zero_mv_threshold: i64,
    predictor_threshold: i64,
    lambda: f64,
    nearest: (i32, i32),
    near: (i32, i32),
    median: (i32, i32),
) -> (MotionVector, i64) {
    let x0 = block_origin(bx, BLOCK_SIZE, width);
    let y0 = block_origin(by, BLOCK_SIZE, height);

    let zero_sad = if block_is_identical(curr_luma, prev_luma, width, height, x0, y0) {
        0
    } else {
        sad_block_int_limit(prev_luma, curr_luma, width, height, x0, y0, 0, 0, i64::MAX)
    };

    // Initial cost includes MV cost for (0,0)
    let zero_mv_cost = estimate_mv_cost(0, 0, nearest, near, median, lambda);
    let mut state = IntSearchState::new(zero_sad + zero_mv_cost);
    // Store pure SAD for threshold checks? No, thresholds are usually on SAD.
    // But if we use cost, we should compare cost.
    // However, zero_mv_threshold is likely calibrated for SAD.
    // Let's keep using SAD for thresholds, but Cost for search.
    // Wait, state.cost is used for search.
    // So state.cost should be Total Cost.

    // If we use Total Cost in state.cost, we need to adjust thresholds.
    // Or we can store SAD separately.
    // IntSearchState only has one cost field.
    // Let's assume thresholds are loose enough or we just check SAD derived from cost.
    // Derived SAD = state.cost - mv_cost.

    let mut skip_integer_search = false;
    if zero_mv_threshold > 0 && zero_sad <= zero_mv_threshold {
        skip_integer_search = true;
    }

    if !skip_integer_search {
        evaluate_seed_candidates(
            candidates, &mut state, sr, prev_luma, curr_luma, width, height, x0, y0, lambda,
            nearest, near, median,
        );

        let current_sad =
            state.cost - estimate_mv_cost(state.dx, state.dy, nearest, near, median, lambda);
        if predictor_threshold > 0 && current_sad <= predictor_threshold {
            skip_integer_search = true;
        }
    }

    if !skip_integer_search {
        refine_integer_motion(
            &mut state, sr, prev_luma, curr_luma, width, height, x0, y0, lambda, nearest, near,
            median,
        );
    }

    // Half-pel refinement
    // We should also include cost in half-pel.
    // But half-pel adds 2 bits for flags.
    // Let's just refine SAD for now, or add simple penalty.
    let (best_dx_hp, best_dy_hp, best_sub_sad) =
        refine_halfpel(&state, prev_luma, curr_luma, width, height, x0, y0, lambda);

    let dx_hp = best_dx_hp.clamp(-255, 255);
    let dy_hp = best_dy_hp.clamp(-255, 255);
    let dx_px = dx_hp / 2;
    let dy_px = dy_hp / 2;
    let dx_frac = dx_hp % 2;
    let dy_frac = dy_hp % 2;
    (
        MotionVector::new(
            dx_px.clamp(-128, 127) as i8,
            dy_px.clamp(-128, 127) as i8,
            dx_frac as i8,
            dy_frac as i8,
            false,
        )
        .as_canonicalized(),
        best_sub_sad,
    )
}

fn evaluate_seed_candidates(
    candidates: &[(i32, i32)],
    state: &mut IntSearchState,
    sr: i32,
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    lambda: f64,
    nearest: (i32, i32),
    near: (i32, i32),
    median: (i32, i32),
) {
    for &(dx, dy) in candidates {
        if (dx == 0 && dy == 0) || dx.abs() > sr || dy.abs() > sr {
            continue;
        }

        let mv_cost = estimate_mv_cost(dx, dy, nearest, near, median, lambda);
        // We want total_cost < state.cost
        // total_cost = sad + mv_cost
        // sad < state.cost - mv_cost
        let limit = state.cost.saturating_sub(mv_cost);

        // If limit is negative, this candidate is already more expensive than current best just from bits
        if limit < 0 {
            continue;
        }

        let sad = sad_block_int_limit(prev_luma, curr_luma, width, height, x0, y0, dx, dy, limit);

        let total_cost = sad + mv_cost;
        if total_cost < state.cost {
            state.dx = dx;
            state.dy = dy;
            state.cost = total_cost;
        }
    }
}

fn refine_integer_motion(
    state: &mut IntSearchState,
    sr: i32,
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    lambda: f64,
    nearest: (i32, i32),
    near: (i32, i32),
    median: (i32, i32),
) {
    // 1. Large Hexagon Search (LHS)
    // We repeat the hexagonal pattern until the best point is the center.
    loop {
        let mut improved = false;
        let center_dx = state.dx;
        let center_dy = state.dy;

        for &(ox, oy) in HEX_OFFSETS {
            let dx = center_dx + ox;
            let dy = center_dy + oy;
            if dx.abs() > sr || dy.abs() > sr {
                continue;
            }

            let mv_cost = estimate_mv_cost(dx, dy, nearest, near, median, lambda);
            let limit = state.cost.saturating_sub(mv_cost);
            if limit < 0 {
                continue;
            }

            let sad =
                sad_block_int_limit(prev_luma, curr_luma, width, height, x0, y0, dx, dy, limit);

            let total_cost = sad + mv_cost;
            if total_cost < state.cost {
                state.dx = dx;
                state.dy = dy;
                state.cost = total_cost;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // 2. Small Diamond Search (SDS) refinement
    // Once LHS converges, we check the 4 immediate neighbors to fine-tune.
    let center_dx = state.dx;
    let center_dy = state.dy;
    for &(ox, oy) in CARDINAL_OFFSETS {
        let dx = center_dx + ox;
        let dy = center_dy + oy;
        if dx.abs() > sr || dy.abs() > sr {
            continue;
        }

        let mv_cost = estimate_mv_cost(dx, dy, nearest, near, median, lambda);
        let limit = state.cost.saturating_sub(mv_cost);
        if limit < 0 {
            continue;
        }

        let sad = sad_block_int_limit(prev_luma, curr_luma, width, height, x0, y0, dx, dy, limit);

        let total_cost = sad + mv_cost;
        if total_cost < state.cost {
            state.dx = dx;
            state.dy = dy;
            state.cost = total_cost;
        }
    }
}

fn refine_halfpel(
    state: &IntSearchState,
    prev_luma: &LumaPlane,
    curr_luma: &LumaPlane,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    lambda: f64,
) -> (i32, i32, i64) {
    let base_dx_hp = state.dx * 2;
    let base_dy_hp = state.dy * 2;
    let mut best_dx_hp = base_dx_hp;
    let mut best_dy_hp = base_dy_hp;

    // Calculate initial cost (integer aligned)
    // We only consider the *additional* cost of fractional parts here,
    // because the integer part cost is common to all candidates in this refinement.
    // Integer aligned (0,0 offset) has 0 additional cost.
    let mut best_total_cost = sad_block_halfpel_limit_luma(
        prev_luma,
        curr_luma,
        width,
        height,
        x0,
        y0,
        base_dx_hp,
        base_dy_hp,
        i64::MAX,
    );
    // Store the pure SAD for the return value
    let mut best_sad = best_total_cost;

    for sy in [-1, 0, 1] {
        for sx in [-1, 0, 1] {
            if sx == 0 && sy == 0 {
                continue;
            }

            let dx_hp = base_dx_hp + sx;
            let dy_hp = base_dy_hp + sy;

            // Skip boundary combo: integer -128 with -0.5 fractional step.
            // This would produce dx_hp/dy_hp = -257 from base -256 (beyond our clamp window),
            // so don't evaluate it in refinement.
            if (state.dx == -128 && sx == -1) || (state.dy == -128 && sy == -1) {
                continue;
            }

            // Fractional penalty: if either component is fractional, we pay for flags.
            // We approximate this as 2 bits if any fraction exists (simple model).
            // Or maybe 1 bit per axis? Let's say 1.5 bits total for non-zero fraction.
            let scale = 2.0; // same scale as in estimate_mv_cost
            let is_fractional = (dx_hp % 2 != 0) || (dy_hp % 2 != 0);
            let penalty = if is_fractional {
                (lambda * scale * 1.5) as i64
            } else {
                0
            };

            let limit = best_total_cost.saturating_sub(penalty);
            if limit < 0 {
                continue;
            }

            let sad = sad_block_halfpel_limit_luma(
                prev_luma, curr_luma, width, height, x0, y0, dx_hp, dy_hp, limit,
            );

            let total_cost = sad + penalty;
            if total_cost < best_total_cost {
                best_total_cost = total_cost;
                best_sad = sad;
                best_dx_hp = dx_hp;
                best_dy_hp = dy_hp;
            }
        }
    }

    (best_dx_hp, best_dy_hp, best_sad)
}

