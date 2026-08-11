//! Wiring between one frontend and one scheduler partition, plus the engine
//! bundle a model line hands back from `launch`.
//!
//! Both ends of a partition are minted together by [`partition_pair`], so a
//! model crate cannot cross-wire or forget a line: the scheduler side arrives
//! as one [`PartitionBackend`] value, the frontend side as one
//! [`PartitionHandle`]. Channel choices per direction: intake and control are
//! crossbeam (sync consumer on the scheduler thread; senders never block on
//! unbounded channels), the step stream is tokio (async consumer in the
//! protocol stack; the sync producer's send never blocks either), load is a
//! shared cell (read-only pull, deliberately unsubscribable — see
//! [`LoadPublisher`]).

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

use tokio::sync::oneshot;

use super::control::EngineControlError;
use super::control::EngineControlRequest;
use super::control::EngineControlResult;
use super::control::LoadLoraAdapterRequest;
use super::control::UnloadLoraAdapterRequest;
use super::handle::LoadSnapshot;
use super::kv::KvCapacity;
use super::step::Request;
use super::step::RequestId;
use super::ticket::HandleCore;
use super::ticket::IntakeTicket;
use super::ticket::RequestControl;
use super::ticket::StepReceiver;

/// Everything the scheduler thread consumes and produces. The driver
/// ([`super::drive`]) destructures this; model code only ever sees the
/// emitter and, through trait arguments, the tickets.
pub struct PartitionBackend {
    pub(crate) intake: crossbeam_channel::Receiver<IntakeTicket>,
    pub(crate) control: crossbeam_channel::Receiver<EngineControlRequest>,
    pub(crate) emitter: super::emitter::StepEmitter,
    pub(crate) load: LoadPublisher,
}

/// Sole writer of a partition's load cell; the driver publishes once per
/// iteration from [`super::Scheduler::load`].
///
/// Deliberately a plain cell and not a `watch` channel: the driver busy-polls,
/// so a subscription edge (`changed()`) would fire per spin and turn any
/// subscriber into a message flood at idle. With only [`PartitionHandle::load`]
/// to read it, "notify me on load change" is unrepresentable — consumers pull
/// the snapshot at the moment they need one. A `Mutex` (not per-field atomics)
/// so a reader never sees fields torn across two steps; both sides touch it
/// uncontended for nanoseconds.
pub struct LoadPublisher(Arc<Mutex<LoadSnapshot>>);

impl LoadPublisher {
    pub(crate) fn publish(&self, snapshot: LoadSnapshot) {
        *self.0.lock().expect("load cell poisoned") = snapshot;
    }
}

/// The frontend's end of one scheduler partition.
pub struct PartitionHandle {
    intake_tx: crossbeam_channel::Sender<IntakeTicket>,
    control_tx: crossbeam_channel::Sender<EngineControlRequest>,
    steps: Option<StepReceiver>,
    load: Arc<Mutex<LoadSnapshot>>,
    next_id: AtomicU64,
    /// Kept so tickets minted after the scheduler thread exits still get
    /// their drop-bomb terminal delivered (the ticket needs a live sender).
    step_tx: super::ticket::StepSender,
}

impl PartitionHandle {
    /// Mint identity, queue timestamp, and abort flag, then hand the request
    /// to the scheduler. Never fails: if the scheduler is gone, the ticket's
    /// drop bomb answers the request with a `Failed` terminal on the step
    /// stream, which the caller observes like any other terminal.
    pub fn submit(&self, request: Request) -> RequestControl {
        let id = RequestId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let abort = Arc::new(AtomicU8::new(0));
        let control = RequestControl::new(id, Arc::clone(&abort));
        let ticket = IntakeTicket::new(
            HandleCore {
                id,
                abort,
                tx: self.step_tx.clone(),
            },
            request,
            Instant::now(),
        );
        if let Err(returned) = self.intake_tx.send(ticket) {
            drop(returned.into_inner());
        }
        control
    }

    /// The step stream, handed out once (there is one stream and one
    /// consumer — the protocol stack's translation loop).
    pub fn take_steps(&mut self) -> Option<StepReceiver> {
        self.steps.take()
    }

    /// The scheduler's most recent load snapshot. Pull-only by design (see
    /// [`LoadPublisher`]): read it at the moment you need one — routing a
    /// request, stamping stats onto an outgoing batch, serving a scrape.
    pub fn load(&self) -> LoadSnapshot {
        *self.load.lock().expect("load cell poisoned")
    }

    /// A cloneable control-plane client for this partition, for consumers
    /// (LoRA routes) that outlive or share the handle.
    pub fn control_client(&self) -> ControlClient {
        ControlClient {
            tx: self.control_tx.clone(),
        }
    }
}

/// Async client for a partition's control plane. Requests reach
/// [`super::Scheduler::control`] through the driver; the base trait
/// implementation answers every one with the unsupported sentinel, which maps
/// back to [`EngineControlError::Unsupported`] here.
#[derive(Clone)]
pub struct ControlClient {
    tx: crossbeam_channel::Sender<EngineControlRequest>,
}

impl ControlClient {
    pub async fn load_lora_adapter(
        &self,
        request: LoadLoraAdapterRequest,
    ) -> EngineControlResult<()> {
        let (response_tx, response_rx) = oneshot::channel();
        self.send_control(
            EngineControlRequest::LoadLoraAdapter {
                request,
                response_tx,
            },
            response_rx,
        )
        .await
    }

    pub async fn unload_lora_adapter(
        &self,
        request: UnloadLoraAdapterRequest,
    ) -> EngineControlResult<()> {
        let (response_tx, response_rx) = oneshot::channel();
        self.send_control(
            EngineControlRequest::UnloadLoraAdapter {
                request,
                response_tx,
            },
            response_rx,
        )
        .await
    }

    pub async fn list_lora_adapters(&self) -> EngineControlResult<Vec<String>> {
        let (response_tx, response_rx) = oneshot::channel();
        self.send_control(
            EngineControlRequest::ListLoraAdapters { response_tx },
            response_rx,
        )
        .await
    }

    async fn send_control<T>(
        &self,
        request: EngineControlRequest,
        response_rx: oneshot::Receiver<Result<T, String>>,
    ) -> EngineControlResult<T> {
        self.tx
            .send(request)
            .map_err(|_| EngineControlError::ChannelClosed)?;
        let response = response_rx
            .await
            .map_err(|_| EngineControlError::ChannelClosed)?;
        match response {
            Ok(value) => Ok(value),
            Err(message) if message == super::control::UNSUPPORTED_CONTROL_MESSAGE => Err(
                EngineControlError::Unsupported(super::control::UNSUPPORTED_CONTROL_MESSAGE),
            ),
            Err(message) => Err(EngineControlError::OperationFailed(message)),
        }
    }
}

/// Mint both ends of one partition.
#[must_use]
pub fn partition_pair() -> (PartitionHandle, PartitionBackend) {
    let (intake_tx, intake_rx) = crossbeam_channel::unbounded();
    let (control_tx, control_rx) = crossbeam_channel::unbounded();
    let (step_tx, step_rx) = tokio::sync::mpsc::unbounded_channel();
    let load = Arc::new(Mutex::new(LoadSnapshot::default()));
    (
        PartitionHandle {
            intake_tx,
            control_tx,
            steps: Some(step_rx),
            load: Arc::clone(&load),
            next_id: AtomicU64::new(0),
            step_tx: step_tx.clone(),
        },
        PartitionBackend {
            intake: intake_rx,
            control: control_rx,
            emitter: super::emitter::StepEmitter::new(step_tx),
            load: LoadPublisher(load),
        },
    )
}

/// What `ModelLine::launch` returns for a step-driven engine. Required fields
/// are the onboarding checklist: an engine that reports no capacity or no
/// servable length says so explicitly with `None`, it cannot just forget.
pub struct Engine {
    /// One entry per scheduler partition (logical DP rank).
    pub partitions: Vec<LivePartition>,
    pub info: EngineInfo,
}

pub struct LivePartition {
    pub handle: PartitionHandle,
    /// The driver thread. Joined by the server at shutdown, after the handles
    /// (and with them the intake senders) are dropped.
    pub join: std::thread::JoinHandle<()>,
}

/// What `ModelLine::launch` returns during the contract migration: either a
/// legacy per-token engine or a step-driven one. Deleted (in favor of
/// [`Engine`] alone) once every model line is migrated.
pub enum LaunchedEngine {
    Handle(super::handle::EngineHandle),
    Stepped(Engine),
}

impl From<super::handle::EngineHandle> for LaunchedEngine {
    fn from(handle: super::handle::EngineHandle) -> Self {
        Self::Handle(handle)
    }
}

impl From<Engine> for LaunchedEngine {
    fn from(engine: Engine) -> Self {
        Self::Stepped(engine)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EngineInfo {
    /// KV pool capacity, or an explicit `None` for engines that do not report
    /// one (the frontend then skips batch-fit checks).
    pub kv_capacity: Option<KvCapacity>,
    /// Longest servable request in tokens, or an explicit `None` to leave the
    /// protocol stack's max-length validation at the model context window.
    pub servable_len: Option<u32>,
    /// Whether the engine answers LoRA control requests, gating the
    /// `/v1/{load,unload}_lora_adapter` routes.
    pub lora_control: bool,
}
