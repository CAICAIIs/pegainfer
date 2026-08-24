//! Tensor-parallel `responses` concern, split out of tp_executor.rs.
//! Reaches the rest of the TP executor via `use super::*;`.

use super::*;

#[derive(Debug)]
pub(super) enum TpWorkerReply {
    Ack,
    DropAck {
        existed: bool,
    },
    Prefill(PrefillResult),
    Decode(DecodeResult),
    Unified(TpUnifiedResult),
    #[cfg(test)]
    Snapshot(WorkerStateSnapshot),
}
#[derive(Debug)]
pub(super) struct TpWorkerResponse {
    pub(super) rank: usize,
    pub(super) result: Result<TpWorkerReply>,
}
pub(super) fn recv_runtime_responses(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    expected: usize,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<Vec<TpWorkerResponse>> {
    collect_runtime_responses(expected, operation, poison, || {
        recv_runtime_response(responses, operation, poison)
    })
}

pub(super) fn collect_runtime_responses(
    expected: usize,
    operation: &'static str,
    poison: &TpRuntimePoison,
    mut recv_next: impl FnMut() -> Result<TpWorkerResponse>,
) -> Result<Vec<TpWorkerResponse>> {
    let mut collected = Vec::with_capacity(expected);
    for _ in 0..expected {
        let response = recv_next()?;
        if let Err(err) = &response.result {
            // A failed rank may leave peers blocked in a collective, so response-set
            // completeness is no longer recoverable or useful.
            let reason = poison.poison(format!(
                "rank {} failed during {operation}: {err:#}",
                response.rank
            ));
            return Err(anyhow::anyhow!(reason));
        }
        collected.push(response);
    }
    Ok(collected)
}

pub(super) fn validate_dispatched_responses<T>(
    result: Result<T>,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<T> {
    result.map_err(|err| {
        let reason = poison.poison(format!(
            "invalid Qwen3.5 TP {operation} response set: {err:#}"
        ));
        anyhow::anyhow!(reason)
    })
}

pub(super) fn validate_exact_rank_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    operation: &'static str,
) -> Result<Vec<(usize, TpWorkerReply)>> {
    anyhow::ensure!(
        responses.len() == world_size,
        "{operation} expected {world_size} responses, got {}",
        responses.len()
    );
    let mut seen_ranks = HashSet::with_capacity(world_size);
    let mut replies = Vec::with_capacity(world_size);
    for response in responses {
        anyhow::ensure!(
            response.rank < world_size,
            "{operation} returned out-of-range rank {} for world size {world_size}",
            response.rank
        );
        anyhow::ensure!(
            seen_ranks.insert(response.rank),
            "{operation} returned duplicate rank {}",
            response.rank
        );
        replies.push((response.rank, response.result?));
    }
    anyhow::ensure!(
        (0..world_size).all(|rank| seen_ranks.contains(&rank)),
        "{operation} response set did not contain every rank"
    );
    replies.sort_unstable_by_key(|(rank, _)| *rank);
    Ok(replies)
}

#[cfg(test)]
pub(super) fn validate_ack_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    operation: &'static str,
) -> Result<()> {
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, operation)? {
        anyhow::ensure!(
            matches!(reply, TpWorkerReply::Ack),
            "{operation} rank {rank} returned {} instead of acknowledgement",
            reply_name(&reply)
        );
    }
    Ok(())
}

pub(super) fn validate_drop_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    expectation: DropExpectation,
) -> Result<()> {
    let mut existence = Vec::with_capacity(world_size);
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, "drop request")? {
        let TpWorkerReply::DropAck { existed } = reply else {
            anyhow::bail!(
                "drop request rank {rank} returned {} instead of drop acknowledgement",
                reply_name(&reply)
            );
        };
        existence.push((rank, existed));
    }
    let expected = expectation == DropExpectation::MustExist;
    anyhow::ensure!(
        existence.iter().all(|(_, existed)| *existed == expected),
        "drop request expected {expectation:?}, got rank existence {existence:?}"
    );
    Ok(())
}

pub(super) fn validate_prefill_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<PrefillResult> {
    let mut primary = None;
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, "prefill")? {
        match (rank, reply) {
            (0, TpWorkerReply::Prefill(result)) => primary = Some(result),
            (0, reply) => anyhow::bail!(
                "prefill rank 0 returned {} instead of primary prefill result",
                reply_name(&reply)
            ),
            (_, TpWorkerReply::Ack) => {}
            (rank, reply) => anyhow::bail!(
                "prefill non-primary rank {rank} returned {} instead of acknowledgement",
                reply_name(&reply)
            ),
        }
    }
    primary.ok_or_else(|| anyhow::anyhow!("prefill returned no primary result"))
}

pub(super) fn validate_decode_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<DecodeResult> {
    let mut primary = None;
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, "decode")? {
        match (rank, reply) {
            (0, TpWorkerReply::Decode(result)) => primary = Some(result),
            (0, reply) => anyhow::bail!(
                "decode rank 0 returned {} instead of primary decode result",
                reply_name(&reply)
            ),
            (_, TpWorkerReply::Ack) => {}
            (rank, reply) => anyhow::bail!(
                "decode non-primary rank {rank} returned {} instead of acknowledgement",
                reply_name(&reply)
            ),
        }
    }
    primary.ok_or_else(|| anyhow::anyhow!("decode returned no primary result"))
}

pub(super) fn validate_unified_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<TpUnifiedResult> {
    let mut primary = None;
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, "unified step")? {
        match (rank, reply) {
            (0, TpWorkerReply::Unified(result)) => primary = Some(result),
            (0, reply) => anyhow::bail!(
                "unified step rank 0 returned {} instead of primary unified result",
                reply_name(&reply)
            ),
            (_, TpWorkerReply::Ack) => {}
            (rank, reply) => anyhow::bail!(
                "unified step non-primary rank {rank} returned {} instead of acknowledgement",
                reply_name(&reply)
            ),
        }
    }
    primary.ok_or_else(|| anyhow::anyhow!("unified step returned no primary result"))
}

pub(super) fn reply_name(reply: &TpWorkerReply) -> &'static str {
    match reply {
        TpWorkerReply::Ack => "acknowledgement",
        TpWorkerReply::DropAck { .. } => "drop acknowledgement",
        TpWorkerReply::Prefill(_) => "prefill result",
        TpWorkerReply::Decode(_) => "decode result",
        TpWorkerReply::Unified(_) => "unified result",
        #[cfg(test)]
        TpWorkerReply::Snapshot(_) => "worker snapshot",
    }
}
pub(super) fn recv_runtime_response(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<TpWorkerResponse> {
    match responses.recv_timeout(TP_RUNTIME_STEP_TIMEOUT) {
        Ok(response) => Ok(response),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let reason = poison.poison(format!("response channel disconnected during {operation}"));
            Err(anyhow::anyhow!(reason))
        }
        Err(mpsc::RecvTimeoutError::Timeout) => fatal_tp_abort(&format!(
            "Qwen3.5 TP {operation} did not complete within {}s",
            TP_RUNTIME_STEP_TIMEOUT.as_secs()
        )),
    }
}
