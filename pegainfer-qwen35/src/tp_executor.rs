//! Tensor-parallel worker runtime for Qwen3.5.
//!
//! Adds one canonical eager unified command while retaining the replicated
//! linear-attention state layout across TP ranks.

use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::thread::{self};

use anyhow::Result;
use pegainfer_core::kv_pool::KvState;
use pegainfer_frontend::sampler::SamplingParams;

use crate::Error;
use crate::config::TensorParallelConfig;
use crate::decode_buffers::BatchDecodeBuffers35;
use crate::executor::DecodePlan;
use crate::executor::DecodeRequestResult;
use crate::executor::DecodeResult;
use crate::executor::DecodeStepItem;
use crate::executor::PrefillPlan;
use crate::executor::PrefillRequestResult;
use crate::executor::PrefillResult;
use crate::executor::PrefillStepItem;
use crate::executor::RequestId;
use crate::forward::prefill::PREFILL_CHUNK_LEN;
use crate::logprobs::snapshot_requested_logprobs;
use crate::prefill_buffers::GdrChunkwiseScratch35;
use crate::recurrent_state::LinearStatePointerTables;
use crate::recurrent_state::RecurrentState;
use crate::weights::ModelRuntimeConfig;
use crate::weights::Qwen35Model;
pub(super) mod worker;
use worker::*;
pub(super) mod responses;
use responses::*;
pub(crate) mod executor;
pub use executor::Qwen35TpExecutor;
pub(crate) use executor::TpDecodeStepItem;
pub(crate) use executor::TpPrefillChunkItem;
pub(crate) use executor::TpUnifiedPlan;
pub(crate) use executor::TpUnifiedResult;
use executor::*;

const TP_NCCL_STARTUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
const TP_RUNTIME_STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
const TP_WORKER_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const TP_RUNTIME_MEMORY_RESERVE_BYTES: usize = 512 * 1024 * 1024;
const TRITON_AOT_DEVICE_TABLE_LEN: usize = 16;

#[allow(dead_code)]
enum TpWorkerCommand {
    Ping {
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunPrefillChunks {
        chunks: Vec<TpPrefillChunkItem>,
        sample_seed: u64,
        start: Arc<TpCommandStartGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunDecodeStep {
        requests: Vec<TpDecodeStepItem>,
        sample_seed: u64,
        start: Arc<TpCommandStartGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    RunUnifiedStep {
        plan: TpUnifiedPlan,
        start: Arc<TpCommandStartGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    DropRequest {
        request_id: RequestId,
        start: Arc<TpCommandStartGate>,
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    #[cfg(test)]
    SnapshotState {
        resp: mpsc::Sender<TpWorkerResponse>,
    },
    #[cfg(test)]
    RemoveRequestStateForTest {
        request_id: RequestId,
        resp: mpsc::Sender<bool>,
    },
    #[cfg(test)]
    DisconnectForTest {
        ready: mpsc::SyncSender<()>,
    },
    Shutdown,
}

/// Scheduler-owned lifecycle proof required from every TP rank during cleanup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropExpectation {
    MustBeAbsent,
    MustExist,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TpCommandDecision {
    #[default]
    Pending,
    Execute,
    Cancel,
}

#[derive(Default)]
struct TpCommandStartGate {
    decision: Mutex<TpCommandDecision>,
    changed: Condvar,
}

impl TpCommandStartGate {
    fn execute(&self) -> bool {
        self.resolve(TpCommandDecision::Execute)
    }

    fn cancel(&self) -> bool {
        self.resolve(TpCommandDecision::Cancel)
    }

    fn wait(&self) -> TpCommandDecision {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        while *decision == TpCommandDecision::Pending {
            decision = self
                .changed
                .wait(decision)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *decision
    }

    fn resolve(&self, next: TpCommandDecision) -> bool {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        if *decision != TpCommandDecision::Pending {
            return false;
        }
        *decision = next;
        self.changed.notify_all();
        true
    }
}

#[derive(Default)]
struct TpRuntimePoison {
    reason: Mutex<Option<String>>,
}

impl TpRuntimePoison {
    fn poison(&self, reason: String) -> String {
        let mut current = self.reason.lock().unwrap_or_else(PoisonError::into_inner);
        current.get_or_insert(reason).clone()
    }

    fn ensure_healthy(&self) -> Result<()> {
        if let Some(reason) = self
            .reason
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
        {
            anyhow::bail!("Qwen3.5 TP executor is poisoned: {reason}");
        }
        Ok(())
    }
}

fn fatal_tp_abort(message: &str) -> ! {
    eprintln!("{message}; aborting");
    log::error!("{message}; aborting");
    std::process::abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_gate_cancel_releases_waiting_workers() {
        let gate = Arc::new(TpStartupGate::default());
        let worker_gate = Arc::clone(&gate);
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let _ = done_tx.send(worker_gate.wait());
        });

        gate.cancel();

        assert!(
            !done_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("cancelled startup gate should release workers within one second")
        );
        waiter.join().unwrap();
    }

    #[test]
    fn nccl_startup_watchdog_disarms_after_success() {
        let (done_tx, watchdog) = spawn_nccl_startup_watchdog().unwrap();
        disarm_nccl_startup_watchdog(done_tx, watchdog).unwrap();
    }

    #[test]
    fn runtime_poison_preserves_first_failure() {
        let poison = TpRuntimePoison::default();
        assert_eq!(poison.poison("rank 1 OOM".into()), "rank 1 OOM");
        assert_eq!(poison.poison("rank 0 NCCL error".into()), "rank 1 OOM");
        let err = poison.ensure_healthy().unwrap_err().to_string();
        assert!(err.contains("rank 1 OOM"));
        assert!(!err.contains("rank 0 NCCL error"));
    }

    #[test]
    fn runtime_response_failure_poisons_executor() {
        let poison = TpRuntimePoison::default();
        let responses = vec![
            reply(0, TpWorkerReply::Ack),
            TpWorkerResponse {
                rank: 1,
                result: Err(anyhow::anyhow!("rank 1 failed")),
            },
        ];

        let err = validate_dispatched_responses(
            validate_ack_responses(responses, 2, "test"),
            "test",
            &poison,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("rank 1 failed"));
        assert!(poison.ensure_healthy().is_err());
    }

    #[test]
    fn runtime_response_collection_fails_fast_when_peer_never_responds() {
        let poison = TpRuntimePoison::default();
        let (tx, rx) = mpsc::channel();
        tx.send(TpWorkerResponse {
            rank: 0,
            result: Err(anyhow::anyhow!("rank 0 failed")),
        })
        .unwrap();
        let _keep_peer_channel_connected = tx;
        let mut receive_attempts = 0;

        let err = collect_runtime_responses(2, "test", &poison, || {
            receive_attempts += 1;
            rx.recv_timeout(std::time::Duration::from_millis(50))
                .map_err(|err| anyhow::anyhow!("waited for nonresponding rank: {err}"))
        })
        .unwrap_err()
        .to_string();

        assert_eq!(receive_attempts, 1, "collector waited for the missing rank");
        assert!(err.contains("rank 0 failed"));
        assert!(!err.contains("waited for nonresponding rank"));
        assert!(poison.ensure_healthy().is_err());
    }

    #[test]
    fn disconnected_runtime_response_poisons_executor() {
        let poison = TpRuntimePoison::default();
        let (tx, rx) = mpsc::channel();
        drop(tx);

        let err = recv_runtime_response(&rx, "test", &poison)
            .unwrap_err()
            .to_string();
        assert!(err.contains("response channel disconnected during test"));
        assert!(poison.ensure_healthy().is_err());
    }

    fn reply(rank: usize, reply: TpWorkerReply) -> TpWorkerResponse {
        TpWorkerResponse {
            rank,
            result: Ok(reply),
        }
    }

    #[test]
    fn mutating_partial_dispatch_cancels_delivered_prefix_and_poisons() {
        let poison = TpRuntimePoison::default();
        let (rank0_tx, rank0_rx) = mpsc::channel();
        let (rank1_tx, rank1_rx) = mpsc::channel::<TpWorkerCommand>();
        let senders = [rank0_tx, rank1_tx];
        let err = dispatch_mutating_commands(
            2,
            "test prefill",
            &poison,
            |start, resp| TpWorkerCommand::RunPrefillChunks {
                chunks: vec![TpPrefillChunkItem::new(
                    RequestId::new(2),
                    vec![9707],
                    0,
                    true,
                )],
                sample_seed: 0,
                start,
                resp,
            },
            |rank, command| {
                if rank == 1 {
                    anyhow::bail!("injected prefix-only dispatch failure");
                }
                senders[rank]
                    .send(command)
                    .map_err(|_| anyhow::anyhow!("test receiver disconnected"))
            },
        )
        .unwrap_err()
        .to_string();

        assert!(err.contains("injected prefix-only dispatch failure"));
        let TpWorkerCommand::RunPrefillChunks { start, .. } = rank0_rx.recv().unwrap() else {
            panic!("expected prefill command")
        };
        assert_eq!(start.wait(), TpCommandDecision::Cancel);
        assert!(matches!(
            rank1_rx.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        assert!(poison.ensure_healthy().is_err());
    }

    #[test]
    fn limits_constructor_rejects_zero_prefill_budget_before_loading() {
        let err = match Qwen35TpExecutor::from_runtime_with_limits("unused", false, &[0, 1], 1, 0) {
            Ok(_) => panic!("zero TP prefill budget should fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("max_prefill_tokens must be positive"));
    }

    #[test]
    fn rejects_single_device_topology() {
        let err = match Qwen35TpExecutor::from_runtime_with_capacity("unused", false, &[0], 1) {
            Ok(_) => panic!("single-device TP topology should fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("requires at least two CUDA devices"));
    }

    #[test]
    fn rejects_tensor_parallel_cuda_graph() {
        let err = match Qwen35TpExecutor::from_runtime_with_capacity("unused", true, &[0, 1], 1) {
            Ok(_) => panic!("TP CUDA Graph should fail"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("eager execution only"));
    }

    #[test]
    fn validates_prefill_chunk_shape() {
        let empty = [TpPrefillChunkItem::new(RequestId::new(1), vec![], 0, false)];
        let err = validate_prefill_chunks(&empty).unwrap_err().to_string();
        assert!(err.contains("is empty"));

        let duplicate = [
            TpPrefillChunkItem::new(RequestId::new(1), vec![151_646], 0, false),
            TpPrefillChunkItem::new(RequestId::new(1), vec![9707], 0, true),
        ];
        let err = validate_prefill_chunks(&duplicate).unwrap_err().to_string();
        assert!(err.contains("duplicate"));
    }

    #[test]
    fn validates_decode_request_shape() {
        validate_decode_requests(&[TpDecodeStepItem::new(
            RequestId::new(1),
            9707,
            0,
            SamplingParams::default(),
        )])
        .expect("single decode request is valid");

        let duplicate = [
            TpDecodeStepItem::new(RequestId::new(1), 9707, 0, SamplingParams::default()),
            TpDecodeStepItem::new(RequestId::new(1), 560, 0, SamplingParams::default()),
        ];
        let err = validate_decode_requests(&duplicate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate"));
    }

    fn assert_workers_empty(executor: &Qwen35TpExecutor) {
        let snapshots = executor
            .snapshot_workers()
            .expect("snapshot healthy TP workers");
        assert_snapshots_empty(&snapshots, executor.world_size());
    }

    fn assert_snapshots_empty(snapshots: &[WorkerStateSnapshot], world_size: usize) {
        assert_eq!(snapshots.len(), world_size);
        for (rank, snapshot) in snapshots.iter().enumerate() {
            assert_eq!(snapshot.rank, rank);
            assert_eq!(snapshot.request_count, 0, "rank {rank} retained requests");
            assert!(
                snapshot.requests.is_empty(),
                "rank {rank} retained request IDs"
            );
        }
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_drop_expectations_detect_rank_lifecycle_divergence() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_drop_expectations_detect_rank_lifecycle_divergence",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");

        executor
            .drop_request(RequestId::new(400), DropExpectation::MustBeAbsent)
            .expect("pre-materialization drop should observe all ranks absent");
        executor.ping_all().expect("absent drop preserves health");

        let clean_id = RequestId::new(401);
        executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(clean_id, vec![151_646, 9707], 0)],
            })
            .expect("materialize clean request");
        executor
            .drop_request(clean_id, DropExpectation::MustExist)
            .expect("materialized drop should observe all ranks present");
        assert_workers_empty(&executor);

        let divergent_id = RequestId::new(402);
        executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(divergent_id, vec![151_646, 9707], 0)],
            })
            .expect("materialize divergent request");
        assert!(
            executor
                .remove_worker_request_state_for_test(1, divergent_id)
                .expect("remove rank-1 request state")
        );
        let err = executor
            .drop_request(divergent_id, DropExpectation::MustExist)
            .unwrap_err()
            .to_string();
        assert!(err.contains("MustExist"));
        assert!(executor.ping_all().is_err());
        let snapshots = executor
            .snapshot_workers_unchecked_for_test()
            .expect("snapshot workers after mixed drop poison");
        assert_snapshots_empty(&snapshots, executor.world_size());
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_partial_dispatch_gate_prevents_rank_local_mutation() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_partial_dispatch_gate_prevents_rank_local_mutation",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let chunk = TpPrefillChunkItem::new(RequestId::new(410), vec![151_646, 9707], 0, true);

        let err = executor
            .inject_prefill_dispatch_failure_for_test(&[chunk], 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("injected dispatch failure at rank 1"));
        assert!(executor.ping_all().is_err());
        let snapshots = executor
            .snapshot_workers_unchecked_for_test()
            .expect("snapshot workers after cancelled prefix dispatch");
        assert_snapshots_empty(&snapshots, executor.world_size());
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_worker_receiver_disconnect_poisons_without_snapshot_claim() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_worker_receiver_disconnect_poisons_without_snapshot_claim",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        executor
            .disconnect_worker_receiver_for_test(1)
            .expect("disconnect rank-1 worker receiver");

        let err = executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(
                    RequestId::new(420),
                    vec![151_646, 9707],
                    0,
                )],
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("failed to dispatch prefill chunks to TP worker rank 1"));
        assert!(executor.ping_all().is_err());
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_unified_step_advances_prefill_and_decode_together() {
        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_unified_step_advances_prefill_and_decode_together",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 2)
            .expect("start TP2 executor");
        let decode_id = RequestId::new(30);
        let decode_prefill = executor
            .execute_prefill(PrefillPlan {
                requests: &[PrefillStepItem::new(decode_id, vec![151_646, 9707], 1)],
            })
            .expect("materialize TP2 decode request");
        let prefill_id = RequestId::new(31);
        let unified = executor
            .execute_unified(&TpUnifiedPlan {
                prefill: vec![TpPrefillChunkItem::new(
                    prefill_id,
                    vec![151_646, 9707],
                    1,
                    true,
                )],
                decode: vec![TpDecodeStepItem::new(
                    decode_id,
                    decode_prefill.requests[0].first_token,
                    1,
                    SamplingParams::default(),
                )],
                prefill_sample_seed: 102,
                decode_sample_seed: 101,
            })
            .expect("run TP2 unified step");

        assert_eq!(unified.prefill.requests.len(), 1);
        assert_eq!(unified.prefill.requests[0].request_id, prefill_id);
        assert!(unified.prefill.requests[0].first_token_logprob.is_some());
        assert_eq!(unified.decode.requests.len(), 1);
        assert_eq!(unified.decode.requests[0].request_id, decode_id);
        assert!(unified.decode.requests[0].logprob.is_some());
        for snapshot in executor.snapshot_workers().expect("snapshot unified state") {
            assert_eq!(snapshot.request_count, 2);
            assert!(
                snapshot
                    .requests
                    .iter()
                    .all(|(_, phase)| *phase == TpRequestPhase::Decoding)
            );
        }

        for request_id in [decode_id, prefill_id] {
            executor
                .drop_request(request_id, DropExpectation::MustExist)
                .expect("drop unified request");
        }
        assert_workers_empty(&executor);
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_drop_all_restores_complete_request_capacity() {
        const CONFIGURED_MAX_BATCH: usize = 2;

        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_drop_all_restores_complete_request_capacity",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(
            &model_path,
            false,
            &[0, 1],
            CONFIGURED_MAX_BATCH,
        )
        .expect("start TP2 executor");
        assert_eq!(executor.max_batch(), CONFIGURED_MAX_BATCH);
        assert_workers_empty(&executor);

        let first_ids: Vec<_> = (100..100 + CONFIGURED_MAX_BATCH as u64)
            .map(RequestId::new)
            .collect();
        let first_requests: Vec<_> = first_ids
            .iter()
            .map(|&request_id| PrefillStepItem::new(request_id, vec![151_646, 9707], 0))
            .collect();
        let first_results = executor
            .execute_prefill(PrefillPlan {
                requests: &first_requests,
            })
            .expect("fill complete TP2 request capacity");
        assert_eq!(first_results.requests.len(), CONFIGURED_MAX_BATCH);
        let expected_ids: HashSet<_> = first_ids.iter().copied().collect();
        for snapshot in executor
            .snapshot_workers()
            .expect("snapshot full TP2 request capacity")
        {
            assert_eq!(snapshot.request_count, CONFIGURED_MAX_BATCH);
            assert_eq!(
                snapshot
                    .requests
                    .iter()
                    .map(|(request_id, _)| *request_id)
                    .collect::<HashSet<_>>(),
                expected_ids
            );
            assert!(
                snapshot
                    .requests
                    .iter()
                    .all(|(_, phase)| *phase == TpRequestPhase::Decoding),
                "rank {} retained a non-decoding request after final prefill",
                snapshot.rank
            );
        }
        for request_id in &first_ids {
            executor
                .drop_request(*request_id, DropExpectation::MustExist)
                .expect("drop first-pass TP2 request");
        }
        assert_workers_empty(&executor);

        let second_ids: Vec<_> = (200..200 + CONFIGURED_MAX_BATCH as u64)
            .map(RequestId::new)
            .collect();
        let second_requests: Vec<_> = second_ids
            .iter()
            .map(|&request_id| PrefillStepItem::new(request_id, vec![151_646, 9707], 0))
            .collect();
        let second_prefill = executor
            .execute_prefill(PrefillPlan {
                requests: &second_requests,
            })
            .expect("refill complete TP2 request capacity");
        assert_eq!(second_prefill.requests.len(), CONFIGURED_MAX_BATCH);
        let decode_requests: Vec<_> = second_prefill
            .requests
            .iter()
            .map(|result| DecodeStepItem::new(result.request_id, result.first_token, 0))
            .collect();
        let decode = executor
            .execute_decode(DecodePlan {
                requests: &decode_requests,
            })
            .expect("complete one decode step after TP2 capacity refill");
        assert_eq!(decode.requests.len(), CONFIGURED_MAX_BATCH);
        for request_id in &second_ids {
            executor
                .drop_request(*request_id, DropExpectation::MustExist)
                .expect("drop second-pass TP2 request");
        }
        assert_workers_empty(&executor);
    }

    #[test]
    #[ignore = "requires two CUDA devices and Qwen3.5 weights"]
    fn tp2_readmission_matches_clean_first_token_artifact() {
        const REQUESTED_LOGPROBS: usize = 5;

        let Some(model_path) = crate::test_fixture::model_path_or_skip(
            "tp2_readmission_matches_clean_first_token_artifact",
        ) else {
            return;
        };
        let executor = Qwen35TpExecutor::from_runtime_with_capacity(&model_path, false, &[0, 1], 1)
            .expect("start TP2 executor");
        let prompt = vec![151_646, 9707];

        let clean_id = RequestId::new(300);
        let clean_request = PrefillStepItem::new(clean_id, prompt.clone(), REQUESTED_LOGPROBS);
        let clean = executor
            .execute_prefill(PrefillPlan {
                requests: &[clean_request],
            })
            .expect("run clean TP2 prefill");
        assert_eq!(clean.requests.len(), 1);
        assert!(clean.requests[0].first_token_logprob.is_some());
        let clean_artifact = (
            clean.requests[0].first_token,
            clean.requests[0].first_token_logprob.clone(),
        );
        executor
            .drop_request(clean_id, DropExpectation::MustExist)
            .expect("drop clean TP2 request");
        assert_workers_empty(&executor);

        let readmitted_id = RequestId::new(301);
        let readmitted_request = PrefillStepItem::new(readmitted_id, prompt, REQUESTED_LOGPROBS);
        let readmitted = executor
            .execute_prefill(PrefillPlan {
                requests: &[readmitted_request],
            })
            .expect("run readmitted TP2 prefill");
        assert_eq!(readmitted.requests.len(), 1);
        let readmitted_artifact = (
            readmitted.requests[0].first_token,
            readmitted.requests[0].first_token_logprob.clone(),
        );
        assert_eq!(readmitted_artifact, clean_artifact);
        executor
            .drop_request(readmitted_id, DropExpectation::MustExist)
            .expect("drop readmitted TP2 request");
        assert_workers_empty(&executor);
    }
}
