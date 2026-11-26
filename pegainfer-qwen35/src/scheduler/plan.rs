use crate::Qwen35SchedulerPolicy;

/// Decode cohort 标识：单 GPU 双 cohort 轮流调度 A/B，TP 与单 cohort 仅使用 A。
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum CohortId {
    A = 0,
    B = 1,
}

impl CohortId {
    pub(super) const fn index(self) -> usize {
        self as usize
    }

    pub(super) const fn other(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

/// 准入时为新请求分配目标 cohort：优先填满 A，确保 `c <= decode_width` 时只跑 A、
/// B 保持空闲；超过单 cohort 宽度后再溢出到 B。
pub(super) fn assign_cohort(cohort_free: [usize; 2]) -> Option<CohortId> {
    if cohort_free[CohortId::A.index()] > 0 {
        Some(CohortId::A)
    } else if cohort_free[CohortId::B.index()] > 0 {
        Some(CohortId::B)
    } else {
        None
    }
}

/// 选择本步要 decode 的 cohort 以及下一步的轮转起点：
/// - 只有 A 活跃时只跑 A（行为与单 cohort 一致）；
/// - 只有 B 活跃时只跑 B；
/// - A/B 均活跃时按 `next_turn` 严格交替（A -> B -> A -> B）。
pub(super) fn next_decode_cohort(
    active_counts: [usize; 2],
    next_turn: CohortId,
) -> Option<(CohortId, CohortId)> {
    let a = active_counts[CohortId::A.index()] > 0;
    let b = active_counts[CohortId::B.index()] > 0;
    match (a, b) {
        (false, false) => None,
        (true, false) => Some((CohortId::A, CohortId::A)),
        (false, true) => Some((CohortId::B, CohortId::B)),
        (true, true) => Some((next_turn, next_turn.other())),
    }
}

pub(super) enum ExecutionPlan<T> {
    Prefill { pending: Vec<T> },
    Decode,
    Unified { pending: Vec<T> },
}

pub(super) struct AdmissionOutcome<T> {
    pub(super) pending: Vec<T>,
    pub(super) deferred: Vec<T>,
    pub(super) rejected: Vec<(T, RejectReason)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ActiveKvBudget {
    pub(super) prompt_len: usize,
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ActiveDecodeState {
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
}

impl ActiveDecodeState {
    fn remaining_tokens(self) -> usize {
        self.max_tokens.saturating_sub(self.generated_count)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PrefillQueueState {
    pub(super) remaining_tokens: usize,
}

const DECODE_FINISH_WINDOW_TOKENS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RejectReason {
    ContextLength { limit: usize },
    KvBudget,
}

pub(super) fn build_next_plan<T>(have_active: bool, pending: Vec<T>) -> Option<ExecutionPlan<T>> {
    if !pending.is_empty() && have_active {
        Some(ExecutionPlan::Unified { pending })
    } else if !pending.is_empty() {
        Some(ExecutionPlan::Prefill { pending })
    } else if have_active {
        Some(ExecutionPlan::Decode)
    } else {
        None
    }
}

pub(super) fn choose_prefill_budget(
    policy: Qwen35SchedulerPolicy,
    base_budget: usize,
    active: &[ActiveDecodeState],
    prefilling: &[PrefillQueueState],
) -> usize {
    assert!(
        base_budget > 0,
        "Qwen3.5 adaptive scheduler requires a positive base prefill budget"
    );
    if policy == Qwen35SchedulerPolicy::Off || active.is_empty() || prefilling.is_empty() {
        return base_budget;
    }

    if active
        .iter()
        .any(|req| req.remaining_tokens() <= DECODE_FINISH_WINDOW_TOKENS)
    {
        return 0;
    }

    prefilling[0].remaining_tokens.min(base_budget)
}

pub(super) fn admit_pending_requests<T>(
    pending: Vec<T>,
    active: &[ActiveKvBudget],
    max_batch: usize,
    page_size: usize,
    available_pages: usize,
    max_request_pages: usize,
    max_context_tokens: usize,
    mut prompt_len: impl FnMut(&T) -> usize,
    mut max_tokens: impl FnMut(&T) -> usize,
) -> AdmissionOutcome<T> {
    assert!(page_size > 0, "Qwen3.5 KV page size must be non-zero");

    let mut page_budget = available_pages.saturating_sub(active_future_pages(active, page_size));
    let slot_budget = max_batch.saturating_sub(active.len());
    let mut admitted = Vec::new();
    let mut still_deferred = Vec::new();
    let mut rejected = Vec::new();
    let mut blocked = false;

    for req in pending {
        let prompt_len = prompt_len(&req);
        let max_tokens = max_tokens(&req);
        if prompt_len.saturating_add(max_tokens) > max_context_tokens {
            rejected.push((
                req,
                RejectReason::ContextLength {
                    limit: max_context_tokens,
                },
            ));
            continue;
        }

        let request_pages = pages_needed(max_kv_tokens(prompt_len, max_tokens), page_size);
        if request_pages > max_request_pages {
            rejected.push((req, RejectReason::KvBudget));
            continue;
        }

        if blocked || admitted.len() >= slot_budget || request_pages > page_budget {
            blocked = true;
            still_deferred.push(req);
            continue;
        }

        page_budget -= request_pages;
        admitted.push(req);
    }

    AdmissionOutcome {
        pending: admitted,
        deferred: still_deferred,
        rejected,
    }
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

// Prefill samples the first output token but does not append it to KV. A
// generated token occupies KV only when it is fed as the next decode input.
// Therefore N returned completion tokens occupy at most N - 1 generated-token
// KV slots.
pub(super) fn max_kv_tokens(prompt_len: usize, max_tokens: usize) -> usize {
    prompt_len.saturating_add(max_tokens.saturating_sub(1))
}

fn current_active_tokens(req: ActiveKvBudget) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

fn active_future_pages(active: &[ActiveKvBudget], page_size: usize) -> usize {
    active
        .iter()
        .map(|req| {
            let max_pages = pages_needed(max_kv_tokens(req.prompt_len, req.max_tokens), page_size);
            let current_pages = pages_needed(current_active_tokens(*req), page_size);
            assert!(
                current_pages <= max_pages,
                "active Qwen3.5 request exceeded its admitted KV lifetime budget"
            );
            max_pages.saturating_sub(current_pages)
        })
        .sum()
}

/// KV-lifetime budget of a request for chunked prefill
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PrefillKvBudget {
    pub(super) current_tokens: usize,
    pub(super) prompt_len: usize,
    pub(super) max_tokens: usize,
}

/// Pages an in-flight chunked prefill will still allocate before it finishes
pub(super) fn prefilling_future_pages(prefilling: &[PrefillKvBudget], page_size: usize) -> usize {
    prefilling
        .iter()
        .map(|req| {
            let max_pages = pages_needed(max_kv_tokens(req.prompt_len, req.max_tokens), page_size);
            let current_pages = pages_needed(req.current_tokens, page_size);
            assert!(
                current_pages <= max_pages,
                "Qwen3.5 chunked prefill exceeded its admitted KV lifetime budget"
            );
            max_pages.saturating_sub(current_pages)
        })
        .sum()
}

/// Decide how many prompt tokens each FIFO-front prefilling request prefills this step
/// return FIFO order with token budget and per-step request count cap (`max_reqs`, matching decode sampling scratch).
pub(super) fn plan_prefill_chunks(
    remaining: &[usize],
    budget: usize,
    max_reqs: usize,
) -> Vec<usize> {
    let mut chunks = Vec::new();
    let mut left = budget;
    for &rem in remaining {
        if left == 0 || rem == 0 || chunks.len() >= max_reqs {
            break;
        }
        let take = rem.min(left);
        chunks.push(take);
        left -= take;
        if take < rem {
            break;
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct Pending {
        id: u32,
        prompt_len: usize,
        max_tokens: usize,
    }

    fn pending(id: u32, prompt_len: usize) -> Pending {
        Pending {
            id,
            prompt_len,
            max_tokens: 1,
        }
    }

    fn pending_with_max(id: u32, prompt_len: usize, max_tokens: usize) -> Pending {
        Pending {
            id,
            prompt_len,
            max_tokens,
        }
    }

    fn ids(reqs: &[Pending]) -> Vec<u32> {
        reqs.iter().map(|req| req.id).collect()
    }

    fn rejected_ids(reqs: &[(Pending, RejectReason)]) -> Vec<u32> {
        reqs.iter().map(|(req, _)| req.id).collect()
    }

    #[test]
    fn plan_selection_follows_active_and_pending_state() {
        assert!(
            build_next_plan::<Pending>(false, vec![]).is_none(),
            "idle scheduler produces no execution plan"
        );
        assert!(
            matches!(
                build_next_plan::<Pending>(true, vec![]),
                Some(ExecutionPlan::Decode)
            ),
            "active-only scheduler tick decodes the active batch"
        );
        assert!(
            matches!(
                build_next_plan(false, vec![pending(1, 8)]),
                Some(ExecutionPlan::Prefill { pending }) if ids(&pending) == vec![1]
            ),
            "pending-only scheduler tick prefills new requests"
        );
        assert!(
            matches!(
                build_next_plan(true, vec![pending(1, 8)]),
                Some(ExecutionPlan::Unified { pending }) if ids(&pending) == vec![1]
            ),
            "active + pending scheduler tick runs the unified path"
        );
    }

    #[test]
    fn adaptive_prefill_budget_preserves_off_policy() {
        let active = [ActiveDecodeState {
            generated_count: 16,
            max_tokens: 256,
        }];
        let prefilling = [PrefillQueueState {
            remaining_tokens: 4096,
        }];

        assert_eq!(
            choose_prefill_budget(Qwen35SchedulerPolicy::Off, 1024, &active, &prefilling),
            1024,
            "off keeps the fixed chunk budget"
        );
    }

    #[test]
    fn adaptive_prefill_budget_never_exceeds_configured_cap() {
        let active = [ActiveDecodeState {
            generated_count: 16,
            max_tokens: 4096,
        }];
        let prefilling = [PrefillQueueState {
            remaining_tokens: 4096,
        }];

        assert_eq!(
            choose_prefill_budget(Qwen35SchedulerPolicy::Auto, 1024, &active, &prefilling),
            1024,
            "auto preserves --max-prefill-tokens as a hard per-step cap"
        );
    }

    #[test]
    fn adaptive_prefill_budget_keeps_standard_long_output_cells_chunked() {
        let active = [ActiveDecodeState {
            generated_count: 16,
            max_tokens: 256,
        }];
        let prefilling = [PrefillQueueState {
            remaining_tokens: 4096,
        }];

        assert_eq!(
            choose_prefill_budget(Qwen35SchedulerPolicy::Auto, 1024, &active, &prefilling),
            1024,
            "standard serving cells with long outputs keep the fixed chunk path"
        );
    }

    #[test]
    fn adaptive_prefill_budget_trims_final_chunk_to_remaining_tokens() {
        let active = [ActiveDecodeState {
            generated_count: 16,
            max_tokens: 4096,
        }];
        let prefilling = [PrefillQueueState {
            remaining_tokens: 512,
        }];

        assert_eq!(
            choose_prefill_budget(Qwen35SchedulerPolicy::Auto, 1024, &active, &prefilling),
            512,
            "auto may shrink the final chunk but never expands beyond the configured cap"
        );
    }

    #[test]
    fn adaptive_prefill_budget_prioritizes_decode_when_active_request_is_finishing() {
        let active = [
            ActiveDecodeState {
                generated_count: 252,
                max_tokens: 256,
            },
            ActiveDecodeState {
                generated_count: 16,
                max_tokens: 4096,
            },
        ];
        let prefilling = [PrefillQueueState {
            remaining_tokens: 4096,
        }];

        assert_eq!(
            choose_prefill_budget(Qwen35SchedulerPolicy::Auto, 1024, &active, &prefilling),
            0,
            "a near-finished active request gets a decode-priority tick before a long prefill"
        );
        assert!(
            matches!(
                build_next_plan::<Pending>(true, vec![]),
                Some(ExecutionPlan::Decode)
            ),
            "zero prefill budget turns the scheduler tick into decode-only work"
        );
    }

    #[test]
    fn adaptive_prefill_budget_prioritizes_decode_before_final_prefill_chunk() {
        let active = [ActiveDecodeState {
            generated_count: 253,
            max_tokens: 256,
        }];
        let prefilling = [PrefillQueueState {
            remaining_tokens: 512,
        }];

        assert_eq!(
            choose_prefill_budget(Qwen35SchedulerPolicy::Auto, 1024, &active, &prefilling),
            0,
            "decode-priority applies before even a final prefill chunk when an active request is finishing"
        );
    }

    #[test]
    fn admission_respects_slot_capacity_and_active_decode_reserve() {
        let active = [
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 18,
            },
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 1,
            },
        ];
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 16), pending(3, 16)],
            &active,
            4,
            16,
            6,
            6,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(
            ids(&outcome.pending),
            vec![1, 2],
            "active future KV growth is reserved before admitting new requests"
        );
        assert_eq!(
            ids(&outcome.deferred),
            vec![3],
            "requests beyond the remaining slot/page budget stay deferred"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_is_fcfs_and_keeps_later_requests_deferred_after_first_miss() {
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 33), pending(3, 16)],
            &[],
            8,
            16,
            3,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2, 3],
            "a later smaller request must not jump ahead of an earlier budget miss"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_keeps_order_when_first_pending_request_misses_budget() {
        let outcome = admit_pending_requests(
            vec![pending(1, 33), pending(2, 16)],
            &[],
            8,
            16,
            2,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(
            ids(&outcome.deferred),
            vec![1, 2],
            "a later smaller request must not bypass the first deferred request"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_uses_ceil_div_at_page_boundaries() {
        let outcome = admit_pending_requests(
            vec![pending(1, 15), pending(2, 16), pending(3, 17)],
            &[],
            8,
            16,
            3,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(
            ids(&outcome.pending),
            vec![1, 2],
            "15 and 16 tokens each use one page"
        );
        assert_eq!(
            ids(&outcome.deferred),
            vec![3],
            "17 tokens needs two pages and waits when only one page remains"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_rejects_requests_past_context_window() {
        let outcome = admit_pending_requests(
            vec![
                pending_with_max(1, 16, 16), // 32 context tokens: admitted.
                pending_with_max(2, 16, 17), // 33 context tokens: rejected.
                pending_with_max(3, 40, 1),  // prompt alone exceeds the window.
            ],
            &[],
            8,
            16,
            1000,
            1000,
            32,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert!(outcome.deferred.is_empty());
        assert_eq!(rejected_ids(&outcome.rejected), vec![2, 3]);
        for (_, reason) in &outcome.rejected {
            assert!(
                matches!(reason, RejectReason::ContextLength { limit: 32 }),
                "over-window requests must be rejected on context length"
            );
        }
    }

    #[test]
    fn context_window_rejection_takes_precedence_over_kv_budget() {
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 40, 40)], // exceeds both 32-token context and 1 page.
            &[],
            8,
            16,
            1000,
            1,
            32,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert!(outcome.pending.is_empty());
        assert!(outcome.deferred.is_empty());
        assert_eq!(rejected_ids(&outcome.rejected), vec![1]);
        assert!(
            matches!(
                outcome.rejected[0].1,
                RejectReason::ContextLength { limit: 32 }
            ),
            "a static context-window violation should not be reported as KV pressure"
        );
    }

    #[test]
    fn admission_returns_all_pending_when_active_batch_is_at_slot_capacity() {
        let outcome = admit_pending_requests(
            vec![pending(1, 1), pending(2, 1)],
            &[
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
            ],
            4,
            16,
            10,
            10,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(ids(&outcome.deferred), vec![1, 2]);
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn active_scheduler_decodes_when_no_pending_request_is_admitted() {
        let outcome = admit_pending_requests(
            vec![pending(1, 16)],
            &[ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 17,
            }],
            4,
            16,
            1,
            4,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(ids(&outcome.deferred), vec![1]);
        assert!(outcome.rejected.is_empty());
        assert!(
            matches!(
                build_next_plan(true, outcome.pending),
                Some(ExecutionPlan::Decode)
            ),
            "active requests should keep decoding when pending requests are all deferred"
        );
    }

    #[test]
    fn admission_counts_pending_generation_budget() {
        let outcome = admit_pending_requests(
            vec![
                pending_with_max(1, 16, 17), // 32 KV tokens -> 2 pages
                pending(2, 16),              // 16 KV tokens -> 1 page
            ],
            &[],
            8,
            16,
            2,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2],
            "pending request 1 reserves its future decode KV page"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn active_future_pages_counts_only_remaining_growth() {
        let active = [
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1, // current 16 tokens -> 1 page
                max_tokens: 33,     // max 48 KV tokens -> 3 pages
            },
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 17, // current 32 tokens -> 2 pages
                max_tokens: 17,      // max 32 KV tokens -> 2 pages
            },
            ActiveKvBudget {
                prompt_len: 9,
                generated_count: 8, // current 16 tokens -> 1 page
                max_tokens: 24,     // max 32 KV tokens -> 2 pages
            },
        ];

        assert_eq!(
            active_future_pages(&active, 16),
            3,
            "active admission reserves only future page growth, not pages already held"
        );
    }

    #[test]
    fn active_future_reservation_can_defer_pending_by_page_budget() {
        let active = [ActiveKvBudget {
            prompt_len: 16,
            generated_count: 1, // current 16 tokens -> 1 page
            max_tokens: 49,     // max 64 KV tokens -> 4 pages; future growth = 3 pages
        }];
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 16)],
            &active,
            8,
            16,
            4,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2],
            "slot budget is available, but active future KV growth leaves only one page"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_rejects_impossible_request_without_blocking_later_fit() {
        let outcome = admit_pending_requests(
            vec![
                pending_with_max(1, 16, 65), // 80 KV tokens -> 5 pages
                pending(2, 16),
            ],
            &[],
            8,
            16,
            4,
            4,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![2]);
        assert!(outcome.deferred.is_empty());
        assert_eq!(rejected_ids(&outcome.rejected), vec![1]);
        assert!(
            matches!(outcome.rejected[0].1, RejectReason::KvBudget),
            "over-budget requests should keep the existing KV rejection reason"
        );
    }

    #[test]
    fn admission_allows_request_at_single_request_page_cap() {
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 16, 49)], // 64 KV tokens -> 4 pages
            &[],
            1,
            16,
            4,
            4,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert!(outcome.deferred.is_empty());
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn one_token_completion_on_page_boundary_uses_only_prompt_page() {
        assert_eq!(max_kv_tokens(16, 1), 16);
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 16, 1)],
            &[],
            1,
            16,
            1,
            1,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert!(outcome.deferred.is_empty());
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn prefill_chunks_slice_long_prompt_and_stop_packing() {
        // One long prompt, budget smaller than its remaining work: take a
        // partial chunk and stop so it stays at the front next step.
        assert_eq!(plan_prefill_chunks(&[5000], 512, 64), vec![512]);
        assert_eq!(plan_prefill_chunks(&[5000, 8], 512, 64), vec![512]);
    }

    #[test]
    fn prefill_chunks_pack_short_prompts_up_to_budget() {
        // Several short prompts pack into one step until the budget runs out.
        assert_eq!(
            plan_prefill_chunks(&[100, 100, 100], 512, 64),
            vec![100, 100, 100]
        );
        // Third prompt only partially fits: 200 + 200 = 400 used, 112 left, the
        // 300-token prompt takes 112 and stops.
        assert_eq!(
            plan_prefill_chunks(&[200, 200, 300], 512, 64),
            vec![200, 200, 112]
        );
    }

    #[test]
    fn prefill_chunks_complete_then_continue_to_next() {
        // A prompt that exactly fits the budget completes; packing continues.
        assert_eq!(plan_prefill_chunks(&[512, 8], 1024, 64), vec![512, 8]);
    }

    #[test]
    fn prefill_chunks_huge_budget_prefills_everything_in_one_step() {
        // The pre-#375 behaviour: a budget above total remaining work prefills
        // every prompt in a single step (up to max_reqs).
        assert_eq!(
            plan_prefill_chunks(&[1000, 2000, 3000], usize::MAX, 64),
            vec![1000, 2000, 3000]
        );
    }

    #[test]
    fn prefill_chunks_cap_request_count_to_decode_width() {
        let remaining = vec![1usize; 128];
        let chunks = plan_prefill_chunks(&remaining, 1024, 64);
        assert_eq!(chunks.len(), 64);
    }

    #[test]
    fn prefill_chunks_empty_queue_schedules_nothing() {
        assert!(plan_prefill_chunks(&[], 512, 64).is_empty());
    }

    #[test]
    fn cohort_assignment_fills_a_before_spilling_to_b() {
        assert_eq!(assign_cohort([64, 64]), Some(CohortId::A));
        assert_eq!(assign_cohort([1, 64]), Some(CohortId::A));
        assert_eq!(assign_cohort([0, 64]), Some(CohortId::B));
        assert_eq!(assign_cohort([0, 1]), Some(CohortId::B));
        assert_eq!(assign_cohort([0, 0]), None);
    }

    #[test]
    fn decode_cohort_selection_stays_on_a_under_64_and_alternates_above_64() {
        assert_eq!(next_decode_cohort([0, 0], CohortId::A), None);
        // c <= 64：只跑 A，next_turn 保持 A。
        assert_eq!(
            next_decode_cohort([64, 0], CohortId::A),
            Some((CohortId::A, CohortId::A))
        );
        assert_eq!(
            next_decode_cohort([32, 0], CohortId::B),
            Some((CohortId::A, CohortId::A))
        );
        // 仅 B 活跃时只跑 B。
        assert_eq!(
            next_decode_cohort([0, 12], CohortId::A),
            Some((CohortId::B, CohortId::B))
        );
        // c > 64：A/B 严格交替。
        let (first, turn1) = next_decode_cohort([64, 32], CohortId::A).unwrap();
        let (second, turn2) = next_decode_cohort([64, 32], turn1).unwrap();
        let (third, turn3) = next_decode_cohort([64, 32], turn2).unwrap();
        assert_eq!(
            (first, second, third),
            (CohortId::A, CohortId::B, CohortId::A)
        );
        assert_eq!(turn3, CohortId::B);
    }

    #[test]
    fn admission_admits_128_resident_requests_and_defers_129th() {
        let pending: Vec<Pending> = (0..129).map(|i| pending_with_max(i, 16, 16)).collect();
        let outcome = admit_pending_requests(
            pending,
            &[],
            128,
            16,
            10_000,
            10_000,
            262_144,
            |req| req.prompt_len,
            |req| req.max_tokens,
        );
        assert_eq!(outcome.pending.len(), 128);
        assert_eq!(ids(&outcome.deferred), vec![128]);
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn prefilling_future_pages_reserves_only_remaining_growth() {
        // current 16 tokens -> 1 page; max 48 KV tokens -> 3 pages; future = 2.
        let prefilling = [PrefillKvBudget {
            current_tokens: 16,
            prompt_len: 16,
            max_tokens: 33,
        }];
        assert_eq!(prefilling_future_pages(&prefilling, 16), 2);
    }

    #[test]
    fn prefilling_future_pages_sums_across_requests() {
        let prefilling = [
            PrefillKvBudget {
                current_tokens: 0, // just admitted, nothing in KV yet -> 0 pages
                prompt_len: 16,
                max_tokens: 17, // max 32 KV tokens -> 2 pages; future = 2
            },
            PrefillKvBudget {
                current_tokens: 32, // 2 pages held
                prompt_len: 40,
                max_tokens: 9, // max 48 KV tokens -> 3 pages; future = 1
            },
        ];
        assert_eq!(prefilling_future_pages(&prefilling, 16), 3);
    }
}
