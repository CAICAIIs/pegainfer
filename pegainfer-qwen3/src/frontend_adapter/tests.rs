//! Contract tests: drive a full `Qwen3Scheduler` partition through the
//! engine contract (submit → step stream → terminal) with a fake executor.
//! These pin the protocol a frontend can rely on, not scheduler internals.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use pegainfer_frontend::engine::AbortReason;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineControlError;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::LivePartition;
use pegainfer_frontend::engine::LoadLoraAdapterRequest;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::StepReceiver;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::engine::UnloadLoraAdapterRequest;

use super::start_with_executor;
use crate::scheduler::DEFAULT_MAX_PREFILL_TOKENS;
use crate::scheduler::test_support::FakeExecutor;
use crate::scheduler::test_support::request;
use crate::scheduler::test_support::request_with_lora;

fn launch(executor: FakeExecutor, lora_control: bool) -> (LivePartition, StepCollector) {
    let mut engine: Engine =
        start_with_executor(executor, 42, DEFAULT_MAX_PREFILL_TOKENS, lora_control);
    let mut partition = engine.partitions.remove(0);
    let steps = partition
        .handle
        .take_steps()
        .expect("fresh partition yields its step stream once");
    (partition, StepCollector::new(steps))
}

/// Demultiplex the step stream per request, preserving each request's update
/// order. Tests await one request's next update without dropping interleaved
/// updates of others.
struct StepCollector {
    steps: StepReceiver,
    buffered: HashMap<RequestId, VecDeque<RequestUpdate>>,
}

impl StepCollector {
    fn new(steps: StepReceiver) -> Self {
        Self {
            steps,
            buffered: HashMap::new(),
        }
    }

    fn next_for(&mut self, id: RequestId) -> RequestUpdate {
        loop {
            if let Some(update) = self.buffered.get_mut(&id).and_then(VecDeque::pop_front) {
                return update;
            }
            let step = self
                .steps
                .blocking_recv()
                .expect("step stream closed while awaiting an update");
            for update in step.updates {
                self.buffered
                    .entry(update.id)
                    .or_default()
                    .push_back(update);
            }
        }
    }

    /// Fold this request's stream to its end: all tokens in order plus the
    /// terminal. Panics if the stream closes without a terminal.
    fn collect_terminal(&mut self, id: RequestId) -> (Vec<u32>, Terminal) {
        let mut tokens = Vec::new();
        loop {
            let update = self.next_for(id);
            tokens.extend_from_slice(&update.tokens);
            if let Some(terminal) = update.terminal {
                return (tokens, terminal);
            }
        }
    }

    /// Drain the remaining stream (until the scheduler is gone) and return
    /// every terminal seen for `id`. For asserting silence after an abort.
    fn drain_terminals_for(&mut self, id: RequestId) -> Vec<Terminal> {
        let mut terminals: Vec<Terminal> = self
            .buffered
            .remove(&id)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|update| update.terminal)
            .collect();
        while let Some(step) = self.steps.blocking_recv() {
            for update in step.updates {
                if update.id == id
                    && let Some(terminal) = update.terminal
                {
                    terminals.push(terminal);
                }
            }
        }
        terminals
    }
}

fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

#[test]
fn unknown_lora_request_is_rejected_without_blocking_base_request() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, dropped);
    let (partition, mut steps) = launch(executor, false);

    let unknown = partition
        .handle
        .submit(request_with_lora(16, 1, Some("missing-adapter")));
    let base = partition.handle.submit(request_with_lora(16, 1, None));

    let (tokens, terminal) = steps.collect_terminal(unknown.id());
    assert!(tokens.is_empty());
    match terminal {
        Terminal::Rejected {
            reason: RejectReason::UnknownLoraAdapter { name },
            prompt_tokens,
        } => {
            assert_eq!(name, "missing-adapter");
            assert_eq!(prompt_tokens, 16);
        }
        other => panic!("unknown adapter request should be rejected, got {other:?}"),
    }

    let (tokens, terminal) = steps.collect_terminal(base.id());
    assert_eq!(
        tokens,
        vec![101],
        "base request should still run after unknown adapter rejection"
    );
    assert!(matches!(terminal, Terminal::Finished { .. }));
}

#[test]
fn decode_error_fails_the_request_and_scheduler_recovers() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, Arc::clone(&dropped)).with_decode_failure();
    let (partition, mut steps) = launch(executor, false);

    let will_fail = partition.handle.submit(request(16, 2));
    let (tokens, terminal) = steps.collect_terminal(will_fail.id());
    assert_eq!(
        tokens,
        vec![100],
        "the prefill token ships before the decode failure"
    );
    match terminal {
        Terminal::Failed {
            message,
            prompt_tokens,
            completion_tokens,
        } => {
            assert!(message.contains("fake decode KV capacity exhausted"));
            assert_eq!(prompt_tokens, 16);
            assert_eq!(completion_tokens, 1);
        }
        other => panic!("decode failure should surface as Terminal::Failed, got {other:?}"),
    }
    assert!(
        wait_until(Duration::from_secs(1), || dropped
            .lock()
            .unwrap()
            .contains(&0)),
        "failed request state should be dropped"
    );

    let after_failure = partition.handle.submit(request(16, 1));
    let (tokens, terminal) = steps.collect_terminal(after_failure.id());
    assert_eq!(
        tokens,
        vec![101],
        "scheduler should accept new work after a decode error"
    );
    assert!(matches!(terminal, Terminal::Finished { .. }));
}

#[test]
fn same_step_finishes_all_reach_their_terminals() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(8, Arc::clone(&dropped));
    let (partition, mut steps) = launch(executor, false);

    // Identical shapes: all three prefill in one step and finish together on
    // the same decode step, exercising the multi-retire path.
    let controls: Vec<_> = (0..3)
        .map(|_| partition.handle.submit(request(16, 2)))
        .collect();
    for (index, control) in controls.iter().enumerate() {
        let (tokens, terminal) = steps.collect_terminal(control.id());
        let id = index as u32;
        assert_eq!(tokens, vec![100 + id, 200 + id]);
        assert!(matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                completion_tokens: 2,
                ..
            }
        ));
    }
    let mut dropped = dropped.lock().unwrap().clone();
    dropped.sort_unstable();
    assert_eq!(dropped, vec![0, 1, 2]);
}

#[test]
fn aborted_request_retires_silently_and_frees_engine_state() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let executor =
        FakeExecutor::new(64, Arc::clone(&dropped)).with_decode_delay(Duration::from_millis(5));
    let (partition, mut steps) = launch(executor, false);

    // Long enough to still be decoding when the abort lands, small enough to
    // fit the fake's per-request block cap.
    let control = partition.handle.submit(request(16, 100));
    let first = steps.next_for(control.id());
    assert!(
        first.scheduled.is_some(),
        "request must be admitted, not rejected: {first:?}"
    );

    control.abort(AbortReason::Cancelled);
    assert!(
        wait_until(Duration::from_secs(1), || dropped
            .lock()
            .unwrap()
            .contains(&0)),
        "aborted request state should be dropped"
    );

    // The abort is the frontend's own act; the scheduler answers with
    // silence, not a terminal. Drain to engine shutdown to prove it.
    drop(partition.handle);
    let terminals = steps.drain_terminals_for(control.id());
    assert!(
        terminals.is_empty(),
        "aborted request must not receive a terminal: {terminals:?}"
    );
    partition.join.join().expect("scheduler thread exits");
}

#[test]
fn lora_control_unloads_adapter_when_idle() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, dropped).with_lora_adapters(&["adapter-a"]);
    let (partition, _steps) = launch(executor, true);
    let control = partition.handle.control_client();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("build runtime");
    runtime
        .block_on(control.unload_lora_adapter(UnloadLoraAdapterRequest {
            lora_name: "adapter-a".to_string(),
            lora_int_id: None,
        }))
        .expect("unload adapter");
    assert_eq!(
        runtime
            .block_on(control.list_lora_adapters())
            .expect("list adapters"),
        Vec::<String>::new()
    );
}

#[test]
fn lora_control_waits_until_scheduler_idle() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let executor = FakeExecutor::new(4, dropped).with_decode_delay(Duration::from_millis(80));
    let (partition, mut steps) = launch(executor, true);

    let long_running = partition.handle.submit(request(16, 3));
    let first = steps.next_for(long_running.id());
    assert_eq!(
        first.tokens,
        vec![100],
        "first token should be emitted before decode"
    );

    let load_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let load_done_thread = Arc::clone(&load_done);
    let control = partition.handle.control_client();
    let load_thread = std::thread::spawn(move || {
        let result = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build runtime")
            .block_on(control.load_lora_adapter(LoadLoraAdapterRequest {
                lora_name: "adapter-a".to_string(),
                lora_path: "/tmp/adapter-a".into(),
                load_inplace: false,
            }));
        load_done_thread.store(true, std::sync::atomic::Ordering::SeqCst);
        result
    });

    std::thread::sleep(Duration::from_millis(20));
    assert!(
        !load_done.load(std::sync::atomic::Ordering::SeqCst),
        "load_lora_adapter should wait while generation is active"
    );

    let (_, terminal) = steps.collect_terminal(long_running.id());
    assert!(matches!(terminal, Terminal::Finished { .. }));

    let error = load_thread
        .join()
        .expect("join load thread")
        .expect_err("adapter load should be a stub error");
    assert!(matches!(error, EngineControlError::OperationFailed(_)));
}
