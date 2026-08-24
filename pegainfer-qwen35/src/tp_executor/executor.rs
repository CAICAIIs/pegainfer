//! Tensor-parallel executor: the TP orchestration (Qwen35TpExecutor) + its step
//! items + plan validators. Split out of tp_executor.rs; reaches worker/responses
//! via `use super::*;`.

use super::*;

/// TP executor. Rank 0 is the primary worker and returns scheduler-visible
/// artifacts; every rank runs the same ordered state-mutating commands.

pub struct Qwen35TpExecutor {
    pub(super) workers: Vec<TpWorker>,
    pub(super) poison: Arc<TpRuntimePoison>,
    pub(super) world_size: usize,
    pub(super) max_batch: usize,
    pub(super) page_size: usize,
    pub(super) capacity_pages_for_requests: usize,
    pub(super) max_position_embeddings: usize,
    pub(super) eos_token_id: u32,
}

#[derive(Clone)]
pub(crate) struct TpPrefillChunkItem {
    pub(super) request_id: RequestId,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) logprobs: usize,
    pub(super) sampling_params: SamplingParams,
    pub(super) finish_prefill: bool,
}

impl TpPrefillChunkItem {
    pub(super) fn new(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        logprobs: usize,
        finish_prefill: bool,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            logprobs,
            sampling_params: SamplingParams::default(),
            finish_prefill,
        }
    }

    pub(crate) fn new_with_sampling(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        logprobs: usize,
        sampling_params: SamplingParams,
        finish_prefill: bool,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            logprobs,
            sampling_params,
            finish_prefill,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TpDecodeStepItem {
    pub(super) request_id: RequestId,
    pub(super) token_id: u32,
    pub(super) logprobs: usize,
    pub(super) sampling_params: SamplingParams,
}

impl TpDecodeStepItem {
    pub(crate) fn new(
        request_id: RequestId,
        token_id: u32,
        logprobs: usize,
        sampling_params: SamplingParams,
    ) -> Self {
        Self {
            request_id,
            token_id,
            logprobs,
            sampling_params,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TpUnifiedPlan {
    pub(crate) prefill: Vec<TpPrefillChunkItem>,
    pub(crate) decode: Vec<TpDecodeStepItem>,
    pub(crate) prefill_sample_seed: u64,
    pub(crate) decode_sample_seed: u64,
}

#[derive(Debug)]
pub(crate) struct TpUnifiedResult {
    pub(crate) prefill: PrefillResult,
    pub(crate) decode: DecodeResult,
}

impl Qwen35TpExecutor {
    pub fn from_runtime_with_capacity(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        max_batch: usize,
    ) -> Result<Self> {
        Self::from_runtime_with_limits(
            model_path,
            enable_cuda_graph,
            device_ordinals,
            max_batch,
            PREFILL_CHUNK_LEN,
        )
    }

    pub(crate) fn from_runtime_with_limits(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        max_batch: usize,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        validate_cuda_ordinals(device_ordinals)?;
        anyhow::ensure!(
            device_ordinals.len() > 1,
            "Qwen3.5 TP executor requires at least two CUDA devices, got {}",
            device_ordinals.len()
        );
        anyhow::ensure!(
            !enable_cuda_graph,
            "Qwen3.5 TP Phase 1 supports eager execution only; disable CUDA Graph"
        );
        anyhow::ensure!(
            max_prefill_tokens > 0,
            "Qwen3.5 TP max_prefill_tokens must be positive"
        );

        let world_size = device_ordinals.len();
        let mut models = Vec::with_capacity(world_size);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            models.push(Qwen35Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph: false,
                    tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                    device_ordinal,
                },
            )?);
        }
        let first = models
            .first()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 TP executor loaded no models"))?;
        let page_size = first.kv_pool().layout().page_size;
        let mut min_capacity_pages = usize::MAX;
        for (rank, model) in models.iter().enumerate() {
            let rank_page_size = model.kv_pool().layout().page_size;
            anyhow::ensure!(
                rank_page_size == page_size,
                "Qwen3.5 TP rank {rank} KV page size {rank_page_size} does not match rank 0 page size {page_size}"
            );
            min_capacity_pages = min_capacity_pages.min(model.kv_pool().capacity_pages());
        }
        let capacity_pages_for_requests = min_capacity_pages.saturating_sub(1);
        let max_position_embeddings = first.config().max_position_embeddings;
        let eos_token_id = first.config().eos_token_id;

        let nccl_id = cudarc::nccl::safe::Id::new()
            .map_err(|e| anyhow::anyhow!("failed to create Qwen3.5 TP NCCL id: {e:?}"))?;
        let startup_gate = Arc::new(TpStartupGate::default());
        let effective_max_batch = Arc::new(AtomicUsize::new(0));
        let poison = Arc::new(TpRuntimePoison::default());
        let mut workers = Vec::with_capacity(world_size);
        let mut preflights = Vec::with_capacity(world_size);
        let mut startups = Vec::with_capacity(world_size);
        for (rank, model) in models.into_iter().enumerate() {
            match TpWorker::spawn(
                rank,
                world_size,
                model,
                max_batch,
                max_prefill_tokens,
                nccl_id,
                Arc::clone(&startup_gate),
                Arc::clone(&effective_max_batch),
                Arc::clone(&poison),
            ) {
                Ok((worker, preflight, startup)) => {
                    workers.push(worker);
                    preflights.push(preflight);
                    startups.push(startup);
                }
                Err(err) => {
                    startup_gate.cancel();
                    return Err(err);
                }
            }
        }
        let mut min_rank_max_batch = max_batch;
        for (rank, preflight) in preflights.into_iter().enumerate() {
            match preflight.recv() {
                Ok(Ok(rank_max_batch)) => {
                    min_rank_max_batch = min_rank_max_batch.min(rank_max_batch);
                }
                Ok(Err(err)) => {
                    startup_gate.cancel();
                    return Err(err);
                }
                Err(_) => {
                    startup_gate.cancel();
                    return Err(anyhow::anyhow!(
                        "Qwen3.5 TP worker {rank} exited during pre-NCCL startup"
                    ));
                }
            }
        }
        anyhow::ensure!(
            min_rank_max_batch > 0,
            "Qwen3.5 TP has no memory capacity for one recurrent request state"
        );
        effective_max_batch.store(min_rank_max_batch, Ordering::Release);
        if min_rank_max_batch < max_batch {
            log::warn!(
                "Qwen3.5 TP max_batch reduced from {max_batch} to {min_rank_max_batch} by rank-local recurrent-state memory capacity"
            );
        }
        let (watchdog_done, watchdog) = match spawn_nccl_startup_watchdog() {
            Ok(watchdog) => watchdog,
            Err(err) => {
                startup_gate.cancel();
                return Err(err);
            }
        };
        startup_gate.connect();
        let startup_result = startups
            .into_iter()
            .enumerate()
            .try_for_each(|(rank, startup)| {
                startup.recv().map_err(|_| {
                    anyhow::anyhow!("Qwen3.5 TP worker {rank} exited during startup")
                })?
            });
        if let Err(err) = startup_result {
            drop(workers);
            disarm_nccl_startup_watchdog(watchdog_done, watchdog)?;
            return Err(err);
        }
        disarm_nccl_startup_watchdog(watchdog_done, watchdog)?;

        Ok(Self {
            workers,
            poison,
            world_size,
            max_batch: min_rank_max_batch,
            page_size,
            capacity_pages_for_requests,
            max_position_embeddings,
            eos_token_id,
        })
    }

    #[cfg(test)]
    pub(super) fn world_size(&self) -> usize {
        self.world_size
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub(crate) fn page_size(&self) -> usize {
        self.page_size
    }

    pub(crate) fn capacity_pages_for_requests(&self) -> usize {
        self.capacity_pages_for_requests
    }

    pub(crate) fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    pub(crate) fn is_stop_token(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
    }

    #[cfg(test)]
    pub(super) fn ping_all(&self) -> Result<()> {
        self.poison.ensure_healthy()?;
        let (resp_tx, resp_rx) = mpsc::channel();
        for worker in &self.workers {
            self.send_or_poison(
                worker,
                TpWorkerCommand::Ping {
                    resp: resp_tx.clone(),
                },
            )?;
        }
        drop(resp_tx);
        let responses = recv_runtime_responses(&resp_rx, self.world_size, "ping", &self.poison)?;
        validate_dispatched_responses(
            validate_ack_responses(responses, self.world_size, "ping"),
            "ping",
            &self.poison,
        )
    }

    pub fn execute_prefill(&self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 TP prefill plan requires at least one request"
        );
        let chunks: Vec<TpPrefillChunkItem> = plan
            .requests
            .iter()
            .cloned()
            .map(TpPrefillChunkItem::from)
            .collect();
        self.execute_prefill_chunks(&chunks)
    }

    pub(super) fn execute_prefill_chunks(
        &self,
        chunks: &[TpPrefillChunkItem],
    ) -> Result<PrefillResult> {
        self.execute_prefill_chunks_with_seed(chunks, 0)
    }

    pub(crate) fn execute_prefill_chunks_with_seed(
        &self,
        chunks: &[TpPrefillChunkItem],
        sample_seed: u64,
    ) -> Result<PrefillResult> {
        self.poison.ensure_healthy()?;
        anyhow::ensure!(
            !chunks.is_empty(),
            "Qwen3.5 TP prefill chunk command requires at least one chunk"
        );
        validate_prefill_chunks(chunks)?;
        let chunks = chunks.to_vec();
        let resp_rx = self.dispatch_mutating("prefill chunks", |start, resp| {
            TpWorkerCommand::RunPrefillChunks {
                chunks: chunks.clone(),
                sample_seed,
                start,
                resp,
            }
        })?;
        let responses =
            recv_runtime_responses(&resp_rx, self.world_size, "prefill chunks", &self.poison)?;
        validate_dispatched_responses(
            validate_prefill_responses(responses, self.world_size),
            "prefill chunks",
            &self.poison,
        )
    }

    pub fn execute_decode(&self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 TP decode plan requires at least one request"
        );
        let requests: Vec<TpDecodeStepItem> = plan
            .requests
            .iter()
            .map(|request| {
                TpDecodeStepItem::new(
                    request.request_id,
                    request.token_id,
                    request.logprobs,
                    SamplingParams::default(),
                )
            })
            .collect();
        self.execute_decode_items(&requests, 0)
    }

    pub(crate) fn execute_decode_items(
        &self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<DecodeResult> {
        self.poison.ensure_healthy()?;
        anyhow::ensure!(
            !requests.is_empty(),
            "Qwen3.5 TP decode plan requires at least one request"
        );
        validate_decode_requests(requests)?;
        let requests = requests.to_vec();
        let resp_rx = self.dispatch_mutating("decode step", |start, resp| {
            TpWorkerCommand::RunDecodeStep {
                requests: requests.clone(),
                sample_seed,
                start,
                resp,
            }
        })?;
        let responses =
            recv_runtime_responses(&resp_rx, self.world_size, "decode step", &self.poison)?;
        validate_dispatched_responses(
            validate_decode_responses(responses, self.world_size),
            "decode step",
            &self.poison,
        )
    }

    pub(crate) fn execute_unified(&self, plan: &TpUnifiedPlan) -> Result<TpUnifiedResult> {
        self.poison.ensure_healthy()?;
        validate_unified_plan(plan, self.max_batch)?;
        let resp_rx = self.dispatch_mutating("unified step", |start, resp| {
            TpWorkerCommand::RunUnifiedStep {
                plan: plan.clone(),
                start,
                resp,
            }
        })?;
        let responses =
            recv_runtime_responses(&resp_rx, self.world_size, "unified step", &self.poison)?;
        validate_dispatched_responses(
            validate_unified_responses(responses, self.world_size),
            "unified step",
            &self.poison,
        )
    }

    pub(crate) fn poison_artifact_contract(
        &self,
        operation: &'static str,
        err: &anyhow::Error,
    ) -> anyhow::Error {
        let reason = self.poison.poison(format!(
            "invalid Qwen3.5 TP {operation} artifact set: {err:#}"
        ));
        anyhow::anyhow!(reason)
    }

    pub fn drop_request(&self, request_id: RequestId, expectation: DropExpectation) -> Result<()> {
        self.poison.ensure_healthy()?;
        let resp_rx =
            self.dispatch_mutating("drop request", |start, resp| TpWorkerCommand::DropRequest {
                request_id,
                start,
                resp,
            })?;
        let responses =
            recv_runtime_responses(&resp_rx, self.world_size, "drop request", &self.poison)?;
        validate_dispatched_responses(
            validate_drop_responses(responses, self.world_size, expectation),
            "drop request",
            &self.poison,
        )
    }

    #[cfg(test)]
    pub(super) fn snapshot_workers(&self) -> Result<Vec<WorkerStateSnapshot>> {
        self.poison.ensure_healthy()?;
        self.snapshot_workers_unchecked_for_test()
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(super) fn snapshot_workers_unchecked_for_test(&self) -> Result<Vec<WorkerStateSnapshot>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        for worker in &self.workers {
            self.send_or_poison(
                worker,
                TpWorkerCommand::SnapshotState {
                    resp: resp_tx.clone(),
                },
            )?;
        }
        drop(resp_tx);
        wait_for_worker_snapshots(&resp_rx, self.world_size, &self.poison)
    }

    #[cfg(test)]
    pub(super) fn inject_prefill_dispatch_failure_for_test(
        &self,
        chunks: &[TpPrefillChunkItem],
        fail_rank: usize,
    ) -> Result<()> {
        self.poison.ensure_healthy()?;
        anyhow::ensure!(
            fail_rank < self.world_size,
            "injected TP dispatch failure rank {fail_rank} is outside world size {}",
            self.world_size
        );
        validate_prefill_chunks(chunks)?;
        let chunks = chunks.to_vec();
        dispatch_mutating_commands(
            self.world_size,
            "injected prefill chunks",
            &self.poison,
            |start, resp| TpWorkerCommand::RunPrefillChunks {
                chunks: chunks.clone(),
                sample_seed: 0,
                start,
                resp,
            },
            |rank, command| {
                if rank == fail_rank {
                    anyhow::bail!("injected dispatch failure at rank {rank}");
                }
                self.workers[rank].send(command)
            },
        )?;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn remove_worker_request_state_for_test(
        &self,
        rank: usize,
        request_id: RequestId,
    ) -> Result<bool> {
        self.poison.ensure_healthy()?;
        let worker = self
            .workers
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("test worker rank {rank} is out of range"))?;
        let (resp_tx, resp_rx) = mpsc::channel();
        worker.send(TpWorkerCommand::RemoveRequestStateForTest {
            request_id,
            resp: resp_tx,
        })?;
        resp_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|err| anyhow::anyhow!("test worker rank {rank} did not remove state: {err}"))
    }

    #[cfg(test)]
    pub(super) fn disconnect_worker_receiver_for_test(&self, rank: usize) -> Result<()> {
        self.poison.ensure_healthy()?;
        let worker = self
            .workers
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("test worker rank {rank} is out of range"))?;
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        worker.send(TpWorkerCommand::DisconnectForTest { ready: ready_tx })?;
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|err| anyhow::anyhow!("test worker rank {rank} did not disconnect: {err}"))?;

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let (resp_tx, _resp_rx) = mpsc::channel();
            if worker
                .send(TpWorkerCommand::Ping { resp: resp_tx })
                .is_err()
            {
                return Ok(());
            }
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "test worker rank {rank} receiver remained connected"
            );
            std::thread::yield_now();
        }
    }

    pub(super) fn dispatch_mutating(
        &self,
        operation: &'static str,
        build: impl Fn(Arc<TpCommandStartGate>, mpsc::Sender<TpWorkerResponse>) -> TpWorkerCommand,
    ) -> Result<mpsc::Receiver<TpWorkerResponse>> {
        dispatch_mutating_commands(
            self.world_size,
            operation,
            &self.poison,
            build,
            |rank, command| self.workers[rank].send(command),
        )
    }

    #[cfg(test)]
    pub(super) fn send_or_poison(&self, worker: &TpWorker, command: TpWorkerCommand) -> Result<()> {
        worker.send(command).map_err(|err| {
            let reason = self
                .poison
                .poison(format!("failed to dispatch TP worker command: {err:#}"));
            anyhow::anyhow!(reason)
        })
    }
}

pub(super) fn dispatch_mutating_commands(
    world_size: usize,
    operation: &'static str,
    poison: &TpRuntimePoison,
    build: impl Fn(Arc<TpCommandStartGate>, mpsc::Sender<TpWorkerResponse>) -> TpWorkerCommand,
    mut send: impl FnMut(usize, TpWorkerCommand) -> Result<()>,
) -> Result<mpsc::Receiver<TpWorkerResponse>> {
    let start = Arc::new(TpCommandStartGate::default());
    let (resp_tx, resp_rx) = mpsc::channel();
    for rank in 0..world_size {
        let command = build(Arc::clone(&start), resp_tx.clone());
        if let Err(err) = send(rank, command) {
            start.cancel();
            let reason = poison.poison(format!(
                "failed to dispatch {operation} to TP worker rank {rank}: {err:#}"
            ));
            return Err(anyhow::anyhow!(reason));
        }
    }
    drop(resp_tx);
    let resolved = start.execute();
    debug_assert!(resolved, "fresh TP command gate resolved more than once");
    Ok(resp_rx)
}

impl Drop for Qwen35TpExecutor {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.tx.send(TpWorkerCommand::Shutdown);
        }
        for worker in &mut self.workers {
            worker.join_bounded();
        }
    }
}

pub(super) fn validate_prefill_chunks(chunks: &[TpPrefillChunkItem]) -> Result<()> {
    let mut seen = HashSet::with_capacity(chunks.len());
    for chunk in chunks {
        anyhow::ensure!(
            !chunk.prompt_tokens.is_empty(),
            "Qwen3.5 TP prefill chunk for request {} is empty",
            chunk.request_id.get()
        );
        anyhow::ensure!(
            seen.insert(chunk.request_id),
            "duplicate Qwen3.5 TP request id {} in one prefill chunk command",
            chunk.request_id.get()
        );
    }
    Ok(())
}

pub(super) fn validate_decode_requests(requests: &[TpDecodeStepItem]) -> Result<()> {
    let mut seen = HashSet::with_capacity(requests.len());
    for request in requests {
        anyhow::ensure!(
            seen.insert(request.request_id),
            "duplicate Qwen3.5 TP request id {} in one decode command",
            request.request_id.get()
        );
    }
    Ok(())
}

pub(super) fn validate_cuda_ordinals(device_ordinals: &[usize]) -> Result<()> {
    let mut seen = HashSet::with_capacity(device_ordinals.len());
    for &ordinal in device_ordinals {
        anyhow::ensure!(
            ordinal < TRITON_AOT_DEVICE_TABLE_LEN,
            "Qwen3.5 TP CUDA ordinal {ordinal} exceeds the Triton AOT device table bound {TRITON_AOT_DEVICE_TABLE_LEN}"
        );
        anyhow::ensure!(
            seen.insert(ordinal),
            "Qwen3.5 TP CUDA ordinals must be distinct; ordinal {ordinal} appears more than once"
        );
    }
    Ok(())
}

pub(super) fn validate_unified_plan(plan: &TpUnifiedPlan, max_batch: usize) -> Result<()> {
    anyhow::ensure!(
        !plan.prefill.is_empty(),
        "Qwen3.5 TP unified plan requires at least one prefill chunk"
    );
    anyhow::ensure!(
        !plan.decode.is_empty(),
        "Qwen3.5 TP unified plan requires at least one decode request"
    );
    validate_prefill_chunks(&plan.prefill)?;
    validate_decode_requests(&plan.decode)?;
    anyhow::ensure!(
        plan.prefill.len().saturating_add(plan.decode.len()) <= max_batch,
        "Qwen3.5 TP unified plan has {} rows, exceeding scheduler capacity {max_batch}",
        plan.prefill.len().saturating_add(plan.decode.len())
    );

    let prefill_ids: HashSet<_> = plan.prefill.iter().map(|item| item.request_id).collect();
    for decode in &plan.decode {
        anyhow::ensure!(
            !prefill_ids.contains(&decode.request_id),
            "Qwen3.5 TP unified plan request id {} appears in both prefill and decode",
            decode.request_id.get()
        );
    }
    Ok(())
}

pub(super) fn validate_unified_worker_state(
    state: &TpWorkerState,
    plan: &TpUnifiedPlan,
) -> Result<()> {
    validate_unified_worker_layout(plan, state.max_batch, state.requests.len(), |request_id| {
        state
            .request_index(request_id)
            .map(|idx| state.requests[idx].phase)
    })
}

pub(super) fn validate_unified_worker_layout(
    plan: &TpUnifiedPlan,
    max_batch: usize,
    resident_count: usize,
    mut phase_for: impl FnMut(RequestId) -> Option<TpRequestPhase>,
) -> Result<()> {
    validate_unified_plan(plan, max_batch)?;

    let new_prefill_count = plan
        .prefill
        .iter()
        .filter(|item| phase_for(item.request_id).is_none())
        .count();
    anyhow::ensure!(
        resident_count.saturating_add(new_prefill_count) <= max_batch,
        "Qwen3.5 TP unified plan would exceed worker capacity {}",
        max_batch
    );

    for item in &plan.prefill {
        if let Some(phase) = phase_for(item.request_id) {
            anyhow::ensure!(
                phase == TpRequestPhase::Prefilling,
                "Qwen3.5 TP unified prefill request {} is already in decode state",
                item.request_id.get()
            );
        }
    }
    for item in &plan.decode {
        let phase = phase_for(item.request_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Qwen3.5 TP unified decode request {} has no worker state",
                item.request_id.get()
            )
        })?;
        anyhow::ensure!(
            phase == TpRequestPhase::Decoding,
            "Qwen3.5 TP unified request {} is not ready for decode",
            item.request_id.get()
        );
    }
    Ok(())
}

impl From<PrefillStepItem> for TpPrefillChunkItem {
    fn from(request: PrefillStepItem) -> Self {
        Self::new(
            request.request_id,
            request.prompt_tokens,
            request.logprobs,
            true,
        )
    }
}

impl From<DecodeStepItem> for TpDecodeStepItem {
    fn from(request: DecodeStepItem) -> Self {
        Self::new(
            request.request_id,
            request.token_id,
            request.logprobs,
            SamplingParams::default(),
        )
    }
}

#[cfg(test)]
pub(super) fn wait_for_worker_snapshots(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    world_size: usize,
    poison: &TpRuntimePoison,
) -> Result<Vec<WorkerStateSnapshot>> {
    let mut seen_ranks = HashSet::with_capacity(world_size);
    let mut snapshots = Vec::with_capacity(world_size);
    for _ in 0..world_size {
        let response = recv_runtime_response(responses, "snapshot state", poison)?;
        anyhow::ensure!(
            response.rank < world_size,
            "Qwen3.5 TP snapshot returned out-of-range rank {} for world size {world_size}",
            response.rank
        );
        anyhow::ensure!(
            seen_ranks.insert(response.rank),
            "Qwen3.5 TP snapshot returned duplicate rank {}",
            response.rank
        );
        match response.result? {
            TpWorkerReply::Snapshot(snapshot) => {
                anyhow::ensure!(
                    snapshot.rank == response.rank,
                    "Qwen3.5 TP snapshot payload rank {} does not match response rank {}",
                    snapshot.rank,
                    response.rank
                );
                anyhow::ensure!(
                    snapshot.request_count == snapshot.requests.len(),
                    "Qwen3.5 TP rank {} snapshot count {} does not match {} request entries",
                    snapshot.rank,
                    snapshot.request_count,
                    snapshot.requests.len()
                );
                snapshots.push(snapshot);
            }
            TpWorkerReply::Ack => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned acknowledgement")
            }
            TpWorkerReply::DropAck { .. } => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned drop acknowledgement")
            }
            TpWorkerReply::Prefill(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned prefill result")
            }
            TpWorkerReply::Decode(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned decode result")
            }
            TpWorkerReply::Unified(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned unified result")
            }
        }
    }
    anyhow::ensure!(
        (0..world_size).all(|rank| seen_ranks.contains(&rank)),
        "Qwen3.5 TP snapshot response set did not contain every rank"
    );
    snapshots.sort_unstable_by_key(|snapshot| snapshot.rank);
    Ok(snapshots)
}
