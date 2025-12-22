use reitero_video_common::MotionVector;

/// Motion-vector prediction modes following the VP9-style neighbor scan.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MvMode {
    Zero,
    Nearest,
    Near,
    TopRight,
    TopLeft,
    New,
    Temporal,
}

impl Default for MvMode {
    fn default() -> Self {
        MvMode::Zero
    }
}

impl MvMode {
    pub fn as_u8(self) -> u8 {
        match self {
            MvMode::Zero => 0,
            MvMode::Nearest => 1,
            MvMode::Near => 2,
            MvMode::TopRight => 3,
            MvMode::TopLeft => 4,
            MvMode::New => 5,
            MvMode::Temporal => 6,
        }
    }
}

/// The reference predictors used for inter modes.
#[derive(Copy, Clone, Debug, Default)]
pub struct MvPredictors {
    pub nearest: (i8, i8, i8, i8),
    pub near: (i8, i8, i8, i8),
    pub temporal: (i8, i8, i8, i8),
}

/// Raw (non-deduped) neighbor vectors that may inform prediction.
///
/// These are the exact neighbors referenced by the VP9-style scan:
/// left, top, top-right, top-left, and temporal (same block previous frame).
#[derive(Copy, Clone, Debug, Default)]
pub struct MvNeighborSet {
    pub left: Option<(i8, i8, i8, i8)>,
    pub top: Option<(i8, i8, i8, i8)>,
    pub top_right: Option<(i8, i8, i8, i8)>,
    pub top_left: Option<(i8, i8, i8, i8)>,
    pub temporal: Option<(i8, i8, i8, i8)>,
}

/// Returns the raw neighbor MVs for a block.
///
/// Note: `current_mvs` is expected to contain already-coded / reconstructed MVs in raster order.
pub fn gather_mv_neighbor_set(
    current_mvs: &[MotionVector],
    prev_mvs: Option<&[MotionVector]>,
    blocks_w: usize,
    blocks_h: usize,
    bx: usize,
    by: usize,
) -> MvNeighborSet {
    let idx = by * blocks_w + bx;
    let max_idx = current_mvs.len();
    let mut set = MvNeighborSet::default();

    if bx > 0 {
        let left_idx = idx - 1;
        if left_idx < max_idx {
            let mv = current_mvs[left_idx];
            set.left = Some((mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y()));
        }
    }

    if by > 0 {
        let top_idx = idx - blocks_w;
        if top_idx < max_idx {
            let mv = current_mvs[top_idx];
            set.top = Some((mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y()));
        }

        if bx + 1 < blocks_w {
            let top_right_idx = top_idx + 1;
            if top_right_idx < max_idx {
                let mv = current_mvs[top_right_idx];
                set.top_right = Some((mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y()));
            }
        }

        if bx > 0 {
            let top_left_idx = top_idx - 1;
            if top_left_idx < max_idx {
                let mv = current_mvs[top_left_idx];
                set.top_left = Some((mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y()));
            }
        }
    }

    if let Some(prev) = prev_mvs {
        let total_blocks = blocks_w * blocks_h;
        if prev.len() == total_blocks {
            if let Some(prev_mv) = prev.get(idx) {
                set.temporal = Some((
                    prev_mv.dx(),
                    prev_mv.dy(),
                    prev_mv.subpixel_x(),
                    prev_mv.subpixel_y(),
                ));
            }
        }
    }

    set
}

/// Derive VP9-style predictor candidates from already coded neighbors and the previous frame.
/// The scan order roughly matches VP9: left, top, top-right, top-left, temporal.
/// Duplicates are removed and the list is padded with zero vectors as needed.
pub fn derive_mv_predictors(
    current_mvs: &[MotionVector],
    prev_mvs: Option<&[MotionVector]>,
    blocks_w: usize,
    blocks_h: usize,
    bx: usize,
    by: usize,
) -> MvPredictors {
    derive_mv_predictors_with_stats(current_mvs, prev_mvs, blocks_w, blocks_h, bx, by).0
}

/// Same as [`derive_mv_predictors`] but also returns how many unique candidates were examined.
pub fn derive_mv_predictors_with_stats(
    current_mvs: &[MotionVector],
    prev_mvs: Option<&[MotionVector]>,
    blocks_w: usize,
    blocks_h: usize,
    bx: usize,
    by: usize,
) -> (MvPredictors, usize) {
    let mut candidates = gather_mv_candidates(current_mvs, prev_mvs, blocks_w, blocks_h, bx, by);
    if candidates.is_empty() {
        candidates.push((0, 0, 0, 0));
    }
    let unique = candidates.len();
    if candidates.len() == 1 {
        candidates.push((0, 0, 0, 0));
    }

    // Calculate Temporal predictor
    let temporal = calculate_temporal_predictor(prev_mvs, blocks_w, bx, by);

    (
        MvPredictors {
            nearest: candidates[0],
            near: candidates[1],
            temporal,
        },
        unique,
    )
}

fn calculate_temporal_predictor(
    prev_mvs: Option<&[MotionVector]>,
    blocks_w: usize,
    bx: usize,
    by: usize,
) -> (i8, i8, i8, i8) {
    if let Some(prev) = prev_mvs {
        let idx = by * blocks_w + bx;
        if idx < prev.len() {
            let mv = prev[idx];
            return (mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y());
        }
    }
    (0, 0, 0, 0)
}

fn gather_mv_candidates(
    current_mvs: &[MotionVector],
    prev_mvs: Option<&[MotionVector]>,
    blocks_w: usize,
    blocks_h: usize,
    bx: usize,
    by: usize,
) -> Vec<(i8, i8, i8, i8)> {
    let idx = by * blocks_w + bx;
    let mut candidates: Vec<(i8, i8, i8, i8)> = Vec::with_capacity(8);
    let mut spatial_mvs: Vec<MotionVector> = Vec::with_capacity(4);
    let max_idx = current_mvs.len();

    // Helper to add candidate if unique
    let add_candidate = |list: &mut Vec<(i8, i8, i8, i8)>, cand: (i8, i8, i8, i8)| {
        if !list.iter().any(|&existing| existing == cand) {
            list.push(cand);
        }
    };

    // Left (already coded in raster order)
    if bx > 0 {
        let left_idx = idx - 1;
        if left_idx < max_idx {
            let mv = current_mvs[left_idx];
            spatial_mvs.push(mv);
            add_candidate(
                &mut candidates,
                (mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y()),
            );
        }
    }

    // Top
    if by > 0 {
        let top_idx = idx - blocks_w;
        if top_idx < max_idx {
            let mv = current_mvs[top_idx];
            spatial_mvs.push(mv);
            add_candidate(
                &mut candidates,
                (mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y()),
            );
        }

        // Top-right
        if bx + 1 < blocks_w {
            let top_right_idx = top_idx + 1;
            if top_right_idx < max_idx {
                let mv = current_mvs[top_right_idx];
                spatial_mvs.push(mv);
                add_candidate(
                    &mut candidates,
                    (mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y()),
                );
            }
        }

        // Top-left as fallback (especially at row edges)
        if bx > 0 && top_idx > 0 {
            let top_left_idx = top_idx - 1;
            if top_left_idx < max_idx {
                let mv = current_mvs[top_left_idx];
                spatial_mvs.push(mv);
                add_candidate(
                    &mut candidates,
                    (mv.dx(), mv.dy(), mv.subpixel_x(), mv.subpixel_y()),
                );
            }
        }
    }

    // Temporal neighbor: same block from previous frame
    if let Some(prev) = prev_mvs {
        let total_blocks = blocks_w * blocks_h;
        if prev.len() == total_blocks {
            if let Some(prev_mv) = prev.get(idx) {
                add_candidate(
                    &mut candidates,
                    (
                        prev_mv.dx(),
                        prev_mv.dy(),
                        prev_mv.subpixel_x(),
                        prev_mv.subpixel_y(),
                    ),
                );
            }
        }
    }

    candidates
}

/// Compute a small context index for MV mode coding based on left/top modes.
pub fn mv_mode_context(modes: &[MvMode], blocks_w: usize, bx: usize, by: usize) -> usize {
    const MAX_CTX: usize = 4;
    let idx = by * blocks_w + bx;

    let left_mode = if bx > 0 {
        let left_idx = idx - 1;
        modes.get(left_idx).copied()
    } else {
        None
    };
    let top_mode = if by > 0 && idx >= blocks_w {
        let top_idx = idx - blocks_w;
        modes.get(top_idx).copied()
    } else {
        None
    };

    let mut score = 0usize;
    for mode in [left_mode, top_mode] {
        if let Some(m) = mode {
            score += match m {
                MvMode::Zero => 2,
                MvMode::Nearest
                | MvMode::Near
                | MvMode::TopRight
                | MvMode::TopLeft
                | MvMode::Temporal => 1,
                MvMode::New => 0,
            };
        }
    }

    score.min(MAX_CTX - 1)
}
