//! Coordinator for the checkpoint-native MTP draft lane.

use anyhow::Context as _;

use super::RankSlots;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::model::Glm52StepShape;
use crate::runner::Glm52MtpAppend;
use crate::runner::Glm52MtpRound;
use crate::runner::Glm52Worker;

/// Native MTP is an EP collective, unlike DSpark, and its collective chain
/// is FIXED: every rank runs the full round (one context forward + the
/// proposal iterations) every step, sized by its OWN appends/proposals —
/// a rank with no work enters with padding rows. No round-kind negotiation,
/// no cross-rank bucket agreement: the per-step collective count is a
/// constant of the code, never a function of fleet state (the free-running
/// fixed-chain discipline, `docs/models/glm52/free-running-dp.md` §4).
pub(super) fn run_mtp_round(
    workers: &[Glm52Worker],
    slots: &mut [RankSlots],
    shapes: &[Glm52StepShape],
    pending_resets: &mut [Vec<usize>],
    rank_appends: Vec<Vec<Glm52MtpAppend>>,
    rank_proposals: Vec<Vec<(usize, u32, usize)>>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        workers.len() == slots.len()
            && shapes.len() == workers.len()
            && pending_resets.len() == workers.len()
            && rank_appends.len() == workers.len()
            && rank_proposals.len() == workers.len(),
        "GLM5.2 native MTP requires one logical rank per local EP worker"
    );
    let pick_bucket = |rows: usize| {
        GLM52_DECODE_BUCKETS
            .into_iter()
            .find(|&bucket| bucket >= rows.max(1))
            .with_context(|| format!("GLM5.2 native MTP row count {rows} exceeds bucket capacity"))
    };
    #[cfg(test)]
    let probe = crate::freerun_probe::enabled().then(|| {
        (
            rank_appends
                .iter()
                .zip(&rank_proposals)
                .map(|(appends, proposals)| appends.is_empty() && proposals.is_empty())
                .collect::<Vec<bool>>(),
            std::time::Instant::now(),
        )
    });

    let mut joins = Vec::with_capacity(workers.len());
    let mut proposal_slots = Vec::with_capacity(workers.len());
    let mut rank_errors: Vec<Option<anyhow::Error>> = (0..workers.len()).map(|_| None).collect();
    for (rank, ((worker, appends), proposals)) in workers
        .iter()
        .zip(rank_appends)
        .zip(rank_proposals)
        .enumerate()
    {
        let slots_for_rank = proposals
            .iter()
            .map(|&(slot, _, _)| slot)
            .collect::<Vec<_>>();
        let resets = std::mem::take(&mut pending_resets[rank]);
        let round = Glm52MtpRound {
            source_bucket: shapes[rank].bucket,
            context_bucket: pick_bucket(appends.len())?,
            draft_bucket: pick_bucket(slots_for_rank.len())?,
            resets,
            appends,
            proposal_slots: slots_for_rank.clone(),
        };
        let response = match worker.mtp_draft_async(round) {
            Ok(response) => Some(response),
            Err(err) => {
                let err = err.context(format!("GLM5.2 rank {rank} MTP draft submission"));
                log::error!("GLM5.2 rank {rank} MTP draft submission failed: {err:#}");
                rank_errors[rank] = Some(err);
                None
            }
        };
        joins.push(response);
        proposal_slots.push(slots_for_rank);
    }

    // Join every rank before returning an error. The first rank received can
    // be blocked inside DeepEP and report only its device timeout; a later
    // response may contain the pre-collective invariant failure that caused
    // it. Preserve every error in the log and return the first in rank order.
    let mut rank_spans = Vec::with_capacity(joins.len());
    for (rank, (rx, expected_slots)) in joins.iter().zip(&proposal_slots).enumerate() {
        let Some(rx) = rx else {
            rank_spans.push(Vec::new());
            continue;
        };
        let result = rx
            .recv()
            .map_err(|_| anyhow::anyhow!("dropped its response"))
            .and_then(|result| result)
            .and_then(|spans| {
                anyhow::ensure!(
                    spans.len() == expected_slots.len(),
                    "returned {} spans for {} proposals",
                    spans.len(),
                    expected_slots.len()
                );
                Ok(spans)
            });
        match result {
            Ok(spans) => rank_spans.push(spans),
            Err(err) => {
                let err = err.context(format!("GLM5.2 rank {rank} MTP draft"));
                log::error!("GLM5.2 rank {rank} MTP draft failed: {err:#}");
                rank_errors[rank] = Some(err);
                rank_spans.push(Vec::new());
            }
        }
    }
    if let Some(err) = rank_errors.into_iter().flatten().next() {
        return Err(err);
    }
    #[cfg(test)]
    if let Some((rank_empty, started)) = probe {
        crate::freerun_probe::record_mtp_round(rank_empty, started.elapsed());
    }

    for (rank, (spans, proposal_slots)) in rank_spans.into_iter().zip(proposal_slots).enumerate() {
        for (slot_id, span) in proposal_slots.into_iter().zip(spans) {
            if let Some(active) = slots[rank][slot_id].as_mut() {
                #[cfg(test)]
                super::slot::record_mtp_proposal(active.req.request_id.as_deref(), &span);
                active
                    .state
                    .set_drafts(span.to_vec(), crate::mtp::GLM52_MTP_DRAFTS);
            }
        }
    }
    Ok(())
}
