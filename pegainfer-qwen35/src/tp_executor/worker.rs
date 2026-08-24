//! Tensor-parallel worker runtime: per-rank worker thread, its state machine,
//! startup/shutdown gating, and thread-bind (CublasThreadGuard). Split out of
//! tp_executor.rs; reached via `use super::*` from the root's other concerns.

use super::*;

pub(super) fn spawn_nccl_startup_watchdog() -> Result<(mpsc::SyncSender<()>, JoinHandle<()>)> {
    let (done_tx, done_rx) = mpsc::sync_channel(1);
    let watchdog = thread::Builder::new()
        .name("qwen35-tp-nccl-startup-watchdog".into())
        .spawn(move || {
            if done_rx.recv_timeout(TP_NCCL_STARTUP_TIMEOUT).is_ok() {
                return;
            }
            eprintln!(
                "Qwen3.5 TP NCCL startup did not complete within {}s; aborting",
                TP_NCCL_STARTUP_TIMEOUT.as_secs()
            );
            log::error!(
                "Qwen3.5 TP NCCL startup did not complete within {}s; aborting",
                TP_NCCL_STARTUP_TIMEOUT.as_secs()
            );
            std::process::abort();
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn Qwen3.5 TP NCCL watchdog: {err}"))?;
    Ok((done_tx, watchdog))
}

#[allow(clippy::needless_pass_by_value)]
pub(super) fn disarm_nccl_startup_watchdog(
    done_tx: mpsc::SyncSender<()>,
    watchdog: JoinHandle<()>,
) -> Result<()> {
    done_tx
        .send(())
        .map_err(|_| anyhow::anyhow!("Qwen3.5 TP NCCL watchdog exited unexpectedly"))?;
    watchdog
        .join()
        .map_err(|_| anyhow::anyhow!("Qwen3.5 TP NCCL watchdog panicked"))
}

pub(super) struct TpWorker {
    pub(super) tx: mpsc::Sender<TpWorkerCommand>,
    pub(super) handle: Option<JoinHandle<()>>,
    pub(super) done: mpsc::Receiver<()>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum TpStartupDecision {
    #[default]
    Pending,
    Connect,
    Cancel,
}

#[derive(Default)]
pub(super) struct TpStartupGate {
    pub(super) decision: Mutex<TpStartupDecision>,
    pub(super) changed: Condvar,
}

impl TpStartupGate {
    pub(super) fn connect(&self) {
        self.set(TpStartupDecision::Connect);
    }

    pub(super) fn cancel(&self) {
        self.set(TpStartupDecision::Cancel);
    }

    pub(super) fn wait(&self) -> bool {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        while *decision == TpStartupDecision::Pending {
            decision = self
                .changed
                .wait(decision)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *decision == TpStartupDecision::Connect
    }

    pub(super) fn set(&self, next: TpStartupDecision) {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        if *decision == TpStartupDecision::Pending {
            *decision = next;
            self.changed.notify_all();
        }
    }
}

impl TpWorker {
    #[allow(clippy::type_complexity)]
    pub(super) fn spawn(
        rank: usize,
        world_size: usize,
        model: Qwen35Model,
        max_batch: usize,
        max_prefill_tokens: usize,
        nccl_id: cudarc::nccl::safe::Id,
        startup_gate: Arc<TpStartupGate>,
        effective_max_batch: Arc<AtomicUsize>,
        poison: Arc<TpRuntimePoison>,
    ) -> Result<(
        Self,
        mpsc::Receiver<Result<usize>>,
        mpsc::Receiver<Result<()>>,
    )> {
        let (tx, rx) = mpsc::channel();
        let (preflight_tx, preflight_rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let panic_poison = Arc::clone(&poison);
        let handle = thread::Builder::new()
            .name(format!("qwen35-tp-rank-{rank}"))
            .spawn(move || {
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    let prepared = TpWorkerPrepared::new(
                        rank,
                        world_size,
                        model,
                        max_batch,
                        max_prefill_tokens,
                    );
                    let prepared = match prepared {
                        Ok((prepared, rank_max_batch)) => {
                            let _ = preflight_tx.send(Ok(rank_max_batch));
                            prepared
                        }
                        Err(err) => {
                            let _ = preflight_tx.send(Err(err));
                            return;
                        }
                    };
                    if !startup_gate.wait() {
                        return;
                    }
                    let max_batch = effective_max_batch.load(Ordering::Acquire);
                    match prepared.connect(nccl_id, max_batch, poison) {
                        Ok(mut state) => {
                            let _ = startup_tx.send(Ok(()));
                            state.run(rx);
                        }
                        Err(err) => {
                            let _ = startup_tx.send(Err(err));
                        }
                    }
                }));
                if outcome.is_err() {
                    panic_poison.poison(format!("worker rank {rank} panicked"));
                }
                let _ = done_tx.send(());
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn Qwen3.5 TP worker {rank}: {e}"))?;

        Ok((
            Self {
                tx,
                handle: Some(handle),
                done: done_rx,
            },
            preflight_rx,
            startup_rx,
        ))
    }

    pub(super) fn send(&self, command: TpWorkerCommand) -> Result<()> {
        self.tx
            .send(command)
            .map_err(|_| anyhow::anyhow!("Qwen3.5 TP worker channel closed"))
    }

    pub(super) fn join_bounded(&mut self) {
        if self.handle.is_none() {
            return;
        }
        if self.done.recv_timeout(TP_WORKER_SHUTDOWN_TIMEOUT).is_err() {
            fatal_tp_abort("Qwen3.5 TP worker did not exit during bounded shutdown");
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TpWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(TpWorkerCommand::Shutdown);
        self.join_bounded();
    }
}

pub(super) struct TpWorkerState {
    pub(super) rank: usize,
    pub(super) _world_size: usize,
    pub(super) max_batch: usize,
    pub(super) model: Qwen35Model,
    pub(super) requests: Vec<TpRequestState>,
    pub(super) decode_buffers: BatchDecodeBuffers35,
    pub(super) sample_scratch: pegainfer_sample::SampleScratch,
    pub(super) _cublas_guard: CublasThreadGuard,
    pub(super) poison: Arc<TpRuntimePoison>,
}

pub(super) struct TpWorkerPrepared {
    pub(super) rank: usize,
    pub(super) world_size: usize,
    pub(super) max_batch: usize,
    pub(super) model: Qwen35Model,
    pub(super) decode_buffers: BatchDecodeBuffers35,
    pub(super) sample_scratch: pegainfer_sample::SampleScratch,
    pub(super) cublas_guard: CublasThreadGuard,
}

pub(super) struct TpRequestState {
    pub(super) request_id: RequestId,
    pub(super) phase: TpRequestPhase,
    pub(super) kv: KvState,
    pub(super) recurrent: RecurrentState,
    pub(super) linear_pointer_tables: LinearStatePointerTables,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TpRequestPhase {
    Prefilling,
    Decoding,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct WorkerStateSnapshot {
    pub(super) rank: usize,
    pub(super) request_count: usize,
    pub(super) requests: Vec<(RequestId, TpRequestPhase)>,
}

impl TpWorkerPrepared {
    pub(super) fn new(
        rank: usize,
        world_size: usize,
        model: Qwen35Model,
        requested_max_batch: usize,
        max_prefill_tokens: usize,
    ) -> Result<(Self, usize)> {
        let cublas_guard = bind_worker_thread(&model)?;
        let (free_bytes, total_bytes) = model
            .device_ctx()
            .ctx
            .mem_get_info()
            .map_err(|err| anyhow::anyhow!("failed to query TP rank {rank} memory: {err}"))?;
        let recurrent_bytes = RecurrentState::allocation_bytes(model.config());
        let prefill_scratch_tokens = prefill_scratch_tokens(max_prefill_tokens);
        let prefill_scratch_bytes =
            GdrChunkwiseScratch35::estimate_bytes(model.config(), prefill_scratch_tokens);
        let max_batch = effective_recurrent_capacity(
            requested_max_batch,
            free_bytes,
            recurrent_bytes,
            TP_RUNTIME_MEMORY_RESERVE_BYTES,
            prefill_scratch_bytes,
        );
        anyhow::ensure!(
            max_batch > 0,
            "Qwen3.5 TP rank {rank} has {} MiB free after fixed buffers, but one recurrent request needs {} MiB plus {} MiB runtime reserve and {} MiB prefill scratch for {} tokens",
            free_bytes / (1024 * 1024),
            recurrent_bytes / (1024 * 1024),
            TP_RUNTIME_MEMORY_RESERVE_BYTES / (1024 * 1024),
            prefill_scratch_bytes / (1024 * 1024),
            prefill_scratch_tokens,
        );
        log::info!(
            "Qwen3.5 TP rank {rank} recurrent capacity: requested={requested_max_batch}, effective={max_batch}, per_request={:.3} MiB, free={:.0} MiB/{:.0} MiB, runtime_reserve={} MiB, prefill_tokens={}, prefill_scratch={:.0} MiB",
            recurrent_bytes as f64 / 1024.0 / 1024.0,
            free_bytes as f64 / 1024.0 / 1024.0,
            total_bytes as f64 / 1024.0 / 1024.0,
            TP_RUNTIME_MEMORY_RESERVE_BYTES / (1024 * 1024),
            prefill_scratch_tokens,
            prefill_scratch_bytes as f64 / 1024.0 / 1024.0,
        );
        let decode_buffers = model.create_batch_decode_buffers_with_capacity(max_batch)?;
        let sample_scratch = pegainfer_sample::SampleScratch::new(
            model.device_ctx(),
            model.config().selection_vocab,
            max_batch,
        )?;
        Ok((
            Self {
                rank,
                world_size,
                max_batch,
                model,
                decode_buffers,
                sample_scratch,
                cublas_guard,
            },
            max_batch,
        ))
    }

    pub(super) fn connect(
        self,
        nccl_id: cudarc::nccl::safe::Id,
        effective_max_batch: usize,
        poison: Arc<TpRuntimePoison>,
    ) -> Result<TpWorkerState> {
        let Self {
            rank,
            world_size,
            max_batch,
            mut model,
            decode_buffers,
            sample_scratch,
            cublas_guard,
        } = self;
        anyhow::ensure!(
            effective_max_batch > 0 && effective_max_batch <= max_batch,
            "Qwen3.5 TP rank {rank} effective max_batch {effective_max_batch} exceeds local capacity {max_batch}"
        );
        let comm = cudarc::nccl::safe::Comm::from_rank(
            model.device_ctx().stream.clone(),
            rank,
            world_size,
            nccl_id,
        )
        .map_err(|e| anyhow::anyhow!("failed to initialize Qwen3.5 TP NCCL rank {rank}: {e:?}"))?;
        model.attach_tp_comm(comm);
        Ok(TpWorkerState {
            rank,
            _world_size: world_size,
            max_batch: effective_max_batch,
            model,
            requests: Vec::new(),
            decode_buffers,
            sample_scratch,
            _cublas_guard: cublas_guard,
            poison,
        })
    }
}

pub(super) fn prefill_scratch_tokens(max_prefill_tokens: usize) -> usize {
    max_prefill_tokens.min(PREFILL_CHUNK_LEN)
}

pub(super) fn effective_recurrent_capacity(
    requested_max_batch: usize,
    free_bytes: usize,
    recurrent_bytes_per_request: usize,
    runtime_reserve_bytes: usize,
    prefill_scratch_bytes: usize,
) -> usize {
    if recurrent_bytes_per_request == 0 {
        return requested_max_batch;
    }
    requested_max_batch.min(
        free_bytes
            .saturating_sub(runtime_reserve_bytes)
            .saturating_sub(prefill_scratch_bytes)
            / recurrent_bytes_per_request,
    )
}

impl TpWorkerState {
    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn run(&mut self, rx: mpsc::Receiver<TpWorkerCommand>) {
        while let Ok(command) = rx.recv() {
            let fatal = match command {
                TpWorkerCommand::Ping { resp } => {
                    self.respond(resp, "ping", Ok(TpWorkerReply::Ack))
                }
                TpWorkerCommand::RunPrefillChunks {
                    chunks,
                    sample_seed,
                    start,
                    resp,
                } => {
                    if start.wait() == TpCommandDecision::Cancel {
                        false
                    } else {
                        let result = self.execute_prefill_chunks(&chunks, sample_seed);
                        self.respond(resp, "prefill", result)
                    }
                }
                TpWorkerCommand::RunDecodeStep {
                    requests,
                    sample_seed,
                    start,
                    resp,
                } => {
                    if start.wait() == TpCommandDecision::Cancel {
                        false
                    } else {
                        let result = self.execute_decode(&requests, sample_seed);
                        self.respond(resp, "decode", result)
                    }
                }
                TpWorkerCommand::RunUnifiedStep { plan, start, resp } => {
                    if start.wait() == TpCommandDecision::Cancel {
                        false
                    } else {
                        let result = self.execute_unified(&plan);
                        self.respond(resp, "unified step", result)
                    }
                }
                TpWorkerCommand::DropRequest {
                    request_id,
                    start,
                    resp,
                } => {
                    if start.wait() == TpCommandDecision::Cancel {
                        false
                    } else {
                        let existed = self.drop_request(request_id);
                        self.respond(resp, "drop request", Ok(TpWorkerReply::DropAck { existed }))
                    }
                }
                #[cfg(test)]
                TpWorkerCommand::SnapshotState { resp } => {
                    let snapshot = WorkerStateSnapshot {
                        rank: self.rank,
                        request_count: self.requests.len(),
                        requests: self
                            .requests
                            .iter()
                            .map(|state| (state.request_id, state.phase))
                            .collect(),
                    };
                    self.respond(
                        resp,
                        "snapshot state",
                        Ok(TpWorkerReply::Snapshot(snapshot)),
                    )
                }
                #[cfg(test)]
                TpWorkerCommand::RemoveRequestStateForTest { request_id, resp } => {
                    let _ = resp.send(self.drop_request(request_id));
                    false
                }
                #[cfg(test)]
                TpWorkerCommand::DisconnectForTest { ready } => {
                    let _ = ready.send(());
                    break;
                }
                TpWorkerCommand::Shutdown => break,
            };
            if fatal {
                break;
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn respond(
        &self,
        resp: mpsc::Sender<TpWorkerResponse>,
        operation: &'static str,
        result: Result<TpWorkerReply>,
    ) -> bool {
        match result {
            Ok(reply) => {
                let _ = resp.send(TpWorkerResponse {
                    rank: self.rank,
                    result: Ok(reply),
                });
                false
            }
            Err(err) => {
                let reason = self.poison.poison(format!(
                    "rank {} failed during {operation}: {err:#}",
                    self.rank
                ));
                let _ = resp.send(TpWorkerResponse {
                    rank: self.rank,
                    result: Err(anyhow::anyhow!(reason)),
                });
                true
            }
        }
    }

    pub(super) fn execute_prefill_chunks(
        &mut self,
        chunks: &[TpPrefillChunkItem],
        sample_seed: u64,
    ) -> Result<TpWorkerReply> {
        let requests = self.execute_prefill_rows(chunks, sample_seed)?;
        if self.rank == 0 {
            Ok(TpWorkerReply::Prefill(PrefillResult { requests }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    pub(super) fn execute_prefill_rows(
        &mut self,
        chunks: &[TpPrefillChunkItem],
        sample_seed: u64,
    ) -> Result<Vec<PrefillRequestResult>> {
        anyhow::ensure!(
            !chunks.is_empty(),
            "Qwen3.5 TP prefill chunk command requires at least one chunk"
        );
        validate_prefill_chunks(chunks)?;
        let new_requests = chunks
            .iter()
            .filter(|chunk| self.request_index(chunk.request_id).is_none())
            .count();
        anyhow::ensure!(
            self.requests.len() + new_requests <= self.max_batch,
            "Qwen3.5 TP prefill chunks would exceed worker capacity {}",
            self.max_batch
        );

        let mut primary_results = Vec::new();
        let mut final_row_idx = 0usize;
        for chunk in chunks {
            let state_idx = self.ensure_prefill_state(chunk.request_id)?;
            let state = &mut self.requests[state_idx];
            anyhow::ensure!(
                state.phase == TpRequestPhase::Prefilling,
                "Qwen3.5 TP request {} is already in decode state",
                chunk.request_id.get()
            );

            let prompt = [chunk.prompt_tokens.as_slice()];
            let mut recurrent_refs = vec![&mut state.recurrent];
            let logits = self.model.batch_prefill_logits(
                &prompt,
                std::slice::from_mut(&mut state.kv),
                &mut recurrent_refs,
            )?;

            if chunk.finish_prefill {
                if self.rank == 0 {
                    // TP prefill samples final chunks one row at a time. Offset
                    // by the final-row index so rows from the same command do
                    // not reuse the same sampling stream.
                    let row_seed = sample_seed.wrapping_add(final_row_idx as u64);
                    let result = self.sample_final_prefill_chunk(chunk, &logits, row_seed)?;
                    primary_results.push(result);
                }
                final_row_idx += 1;
                self.requests[state_idx].phase = TpRequestPhase::Decoding;
            }
        }

        Ok(primary_results)
    }

    pub(super) fn sample_final_prefill_chunk(
        &mut self,
        chunk: &TpPrefillChunkItem,
        logits: &pegainfer_core::tensor::HiddenStates,
        sample_seed: u64,
    ) -> Result<PrefillRequestResult> {
        let cpu_logits =
            snapshot_requested_logprobs(self.model.device_ctx(), logits, &[chunk.logprobs])?;
        let params_refs = [&chunk.sampling_params];
        let tokens = pegainfer_sample::select_batch(
            self.model.device_ctx(),
            logits,
            &params_refs,
            &[0],
            sample_seed,
            &mut self.sample_scratch,
        )?;
        let first_token = tokens[0];
        let first_token_logprob = cpu_logits[0].as_ref().and_then(|row| {
            pegainfer_sample::token_logprob_from_row(row, first_token, chunk.logprobs)
        });
        Ok(PrefillRequestResult {
            request_id: chunk.request_id,
            first_token,
            first_token_logprob,
        })
    }

    pub(super) fn execute_decode(
        &mut self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<TpWorkerReply> {
        let requests = self.execute_decode_rows(requests, sample_seed)?;
        if self.rank == 0 {
            Ok(TpWorkerReply::Decode(DecodeResult { requests }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    pub(super) fn execute_decode_rows(
        &mut self,
        requests: &[TpDecodeStepItem],
        sample_seed: u64,
    ) -> Result<Vec<DecodeRequestResult>> {
        anyhow::ensure!(
            !requests.is_empty(),
            "Qwen3.5 TP decode command requires at least one request"
        );
        validate_decode_requests(requests)?;
        anyhow::ensure!(
            requests.len() <= self.max_batch,
            "Qwen3.5 TP decode batch {} exceeds worker capacity {}",
            requests.len(),
            self.max_batch
        );

        let mut primary_results =
            Vec::with_capacity(if self.rank == 0 { requests.len() } else { 0 });
        for (row_idx, request) in requests.iter().enumerate() {
            let state_idx = self.request_index(request.request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 TP decode request {} has no worker state",
                    request.request_id.get()
                )
            })?;
            anyhow::ensure!(
                self.requests[state_idx].phase == TpRequestPhase::Decoding,
                "Qwen3.5 TP request {} is not ready for decode",
                request.request_id.get()
            );

            {
                let state = &mut self.requests[state_idx];
                let mut kv_refs = [&mut state.kv];
                let mut recurrent_refs = [&mut state.recurrent];
                self.model.batch_decode_eager_logits(
                    &[request.token_id],
                    &mut kv_refs,
                    &mut recurrent_refs,
                    &state.linear_pointer_tables,
                    &mut self.decode_buffers,
                )?;
            }

            if self.rank == 0 {
                let cpu_logits = snapshot_requested_logprobs(
                    self.model.device_ctx(),
                    &self.decode_buffers.logits,
                    &[request.logprobs],
                )?;
                let params_refs = [&request.sampling_params];
                let tokens = pegainfer_sample::select_batch(
                    self.model.device_ctx(),
                    &self.decode_buffers.logits,
                    &params_refs,
                    &[0],
                    sample_seed.wrapping_add(row_idx as u64),
                    &mut self.sample_scratch,
                )?;
                let token = tokens[0];
                let logprob = cpu_logits[0].as_ref().and_then(|row| {
                    pegainfer_sample::token_logprob_from_row(row, token, request.logprobs)
                });
                primary_results.push(DecodeRequestResult {
                    request_id: request.request_id,
                    token,
                    logprob,
                });
            }
        }

        Ok(primary_results)
    }

    pub(super) fn execute_unified(&mut self, plan: &TpUnifiedPlan) -> Result<TpWorkerReply> {
        validate_unified_worker_state(self, plan)?;

        // The command order is canonical across ranks. Sampling seeds are
        // selected by the scheduler in decode-then-prefill order, independent
        // of this forward order.
        let prefill_requests =
            self.execute_prefill_rows(&plan.prefill, plan.prefill_sample_seed)?;
        let decode_requests = self.execute_decode_rows(&plan.decode, plan.decode_sample_seed)?;

        if self.rank == 0 {
            Ok(TpWorkerReply::Unified(TpUnifiedResult {
                prefill: PrefillResult {
                    requests: prefill_requests,
                },
                decode: DecodeResult {
                    requests: decode_requests,
                },
            }))
        } else {
            Ok(TpWorkerReply::Ack)
        }
    }

    pub(super) fn ensure_prefill_state(&mut self, request_id: RequestId) -> Result<usize> {
        if let Some(idx) = self.request_index(request_id) {
            return Ok(idx);
        }
        let mut recurrent = RecurrentState::new(self.model.device_ctx(), self.model.config())?;
        let linear_pointer_tables = {
            let mut recurrent_refs = [&mut recurrent];
            LinearStatePointerTables::from_recurrent_refs(
                self.model.device_ctx(),
                self.model.config(),
                &mut recurrent_refs,
                1,
                "Qwen3.5 TP eager",
            )?
        };
        let state = TpRequestState {
            request_id,
            phase: TpRequestPhase::Prefilling,
            kv: self.model.alloc_kv(),
            recurrent,
            linear_pointer_tables,
        };
        self.requests.push(state);
        Ok(self.requests.len() - 1)
    }

    pub(super) fn request_index(&self, request_id: RequestId) -> Option<usize> {
        self.requests
            .iter()
            .position(|state| state.request_id == request_id)
    }

    pub(super) fn drop_request(&mut self, request_id: RequestId) -> bool {
        if let Some(idx) = self.request_index(request_id) {
            self.requests.swap_remove(idx);
            true
        } else {
            false
        }
    }
}
pub(super) struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

pub(super) fn bind_worker_thread(model: &Qwen35Model) -> Result<CublasThreadGuard> {
    let ctx = model.device_ctx();
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on Qwen3.5 TP worker thread: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!("Failed to bind CUDA context to Qwen3.5 TP worker thread: {e}")
    })?;
    unsafe {
        crate::ffi::cublas_init();
    }
    Ok(CublasThreadGuard)
}
