# GLM5.2 P/D native-MTP handoff

> **TL;DR:** TP4 prefill transfers 99 target plus two committed native-MTP
> arenas and a five-token proposal to EP decode; gates from 89 tokens to 16K
> restore byte-identically over RDMA with `first_step=verify`. End-to-end
> through a dual-endpoint router (P TP4 + D EP8 across two trays): parity
> gate byte-exact, GSM8K full 1,276/1,315 strict (0.970) at c32 — parity
> with the single-instance reference — and random-IO sweeps show the
> throughput knee at c32 (~1.6k tok/s out, ~14.1k tok/s in ceiling). Under
> sustained long-generation load the hard-coded 15 s handoff deadline is the
> first limit: c64 rejects ~24% on decode1, c128 ~83%; a GSM8K boundary case
> also exposed and fixed a fatal admission-capacity drift at
> `(input+output) ≡ 1 (mod 64)` (`d791dffc`).
>
> **Last touched:** 2026-07

## Preparation

- **Read**:
  - `docs/index.md` — GLM5.2 P/D state belongs with the model line, and the
    existing P/D execution record is authoritative for page naming, strict
    restore, and handoff failure semantics.
  - `docs/models/glm52/pd-m2-execution.md` — the merged target-only contract
    transfers 78 MLA plus 21 index-K arenas, forwards the first target token,
    and admits D at `suffix == 1`; speculative state is explicitly absent.
  - `docs/models/glm52/tp4-prefill-only.md` — native TP4 prefill already emits
    the canonical 656-byte MLA and 132-byte index-K rows consumed by EP
    decode, but currently rejects external P/D and returns its predicted token
    without writing that token's target KV.
  - `docs/models/glm52/native-mtp-accuracy.md` — native MTP consumes the
    target's final-normalized hidden boundary and owns a separate layer-78 MLA
    plus index-K cache whose continuity affects acceptance.
- **Relevant history**:
  - `docs/models/glm52/pd-m2-execution.md` established that a strict D worker
    must never silently recompute missing prompt state and that transfer
    completion can lag the P response.
  - Native MTP is currently restricted to single-process EP8: its layer-78
    build path hard-codes EP8, its round uses the EP8 collective state, and
    its cache is not returned by `Glm52RankModel::kv_arenas`.
  - The desired boundary is stronger than the existing `suffix == 1`
    handoff: P returns an anchor plus initial draft token IDs, and D's first
    target step verifies that span directly.
  - Post-review found that the original admission log claimed
    `first_step=verify` without changing the slot out of its one-token prompt
    suffix. The historical hardware runs restored the intended bytes but did
    not verify the transferred proposal on their first target step.
- **Plan**:
  1. Specify and unit-test the handoff state machine: P transfers committed
     target pages, committed MTP layer-78 pages, `anchor`, initial draft token
     IDs, committed lengths, and page metadata; speculative MTP tail pages
     are not authoritative. D installs the proposal as its first
     `SpanKind::Speculative`, verifies it, then rebuilds MTP continuity from
     verifier hidden rows before making the next proposal.
  2. Generalize native layer 78 away from the EP8-only build boundary. Add a
     TP4 producer context path that consumes each target prefill chunk's
     final-normalized hidden rows and shifted prompt token IDs in batch,
     writes MTP MLA/index-K pages, and makes one initial proposal after the
     target anchor is sampled.
  3. Extend the PegaFlow registration contract from 99 target arenas to 101
     target-plus-MTP arenas. Give the MTP MLA/index-K pages stable names and
     page-first geometry, select one identical TP4 producer copy rather than
     concatenating rank shards, and keep incomplete restores fail-closed.
  4. Extend EP decode admission to restore the MTP arenas and seed the
     forwarded proposal so its first forward is verification. Preserve the
     existing target-only P/D mode when native MTP handoff metadata is absent.
  5. Gate CPU state transitions and arena geometry, release builds/tests, and
     hardware behavior. First validate TP4 P → EP4 D on GB300 with exact
     target output, `first_step=verify`, MTP committed-length continuity, and
     acceptance telemetry; repeat TP4 P → EP16 D when a 16-rank decode
     environment is available.
- **Risks / open questions**:
  - Chunk-boundary shifting must not create or omit the MTP row spanning the
    last token of one target chunk and the first token of the next.
  - The P-side TP4 layer-78 MoE produces proposals with different reduction
    numerics from EP decode. Target verification preserves output correctness,
    but first-round acceptance must be measured rather than assumed.
  - MTP MLA is logically replicated under TP4 even though query heads and MoE
    compute are sharded; this needs a byte-equality gate across producer ranks
    before registering only one copy.
  - The existing external producer is vLLM TP8. Native OpenInfer TP4 producer
    metadata must be a versioned extension, not an implicit reinterpretation
    of the merged target-only protocol.
  - EP16 requires multi-node collective and deployment validation beyond the
    four-GPU local development host; EP4 is the first executable consumer
    gate.

## Execution Log

### Full P/D-stack validation: router parity, GSM8K, load sweeps (2026-07-30)

Same deployment as the multi-process section above (P TP4 tray03, D EP8
tray13+tray14, vLLM router on tray03:10001, round_robin over both decode
endpoints), exercised end to end through the router.

- **Router parity gate**: for a prompt of N ids with anchor a,
  `router(ids, mt=6).text == D(ids, mt=1).text + D(ids+[a], mt=5).text` —
  byte-exact at N=148 and N=4,096 on BOTH decode endpoints (round_robin
  confirmed to hit decode0 rank=0 and decode1 rank=4, each admitting with
  `first_step=verify`). 4K router handoff ~352 ms versus ~29 s decode-local
  recompute. The gate script treats the anchor as the client's first
  generated token (`completion_tokens=6` includes it).
- **Fatal admission-capacity drift found by GSM8K and fixed** (`d791dffc`):
  `adopt_external_prefill_anchor` reclassifies the anchor from input to
  generated (`num_input_tokens -= 1`), so `RequestKv::lifetime_blocks()` lost
  one block of capacity exactly when `(input + output) ≡ 1 (mod 64)` —
  e.g. 1,601+4,096=5,697 → 90 blocks dropped to 89, tripping the
  admission `ensure` and fail-stopping the whole EP fleet (the second
  decode's `CUDA_ERROR_LAUNCH_FAILED` was collateral of the first decode's
  death). `RequestKv` now freezes its lifetime capacity at construction;
  regression test `external_prefill_anchor_promotion_keeps_lifetime_capacity`.
- **GSM8K (8-shot, temp 0, max_tokens 4,096, through the router)**:
  n200 c32 = 196/200 strict (0.980), 0 errors; full 1,319 c32 = 1,276/1,315
  strict (0.970), 4 errors. At parity with the single-instance
  admission-fix reference 1,280/1,315 (0.973). The 4 full-run errors were a
  startup transient: they were rejected while decode1 was still draining
  timed-out handoffs from the preceding c64 experiment; steady state at c32
  is clean.
- **Load ceiling: the 15 s handoff deadline is the first thing to bind.**
  `REMOTE_FETCH_DEADLINE` (scheduler/offload.rs) caps one request's
  remote-KV wait; a parked request that outlives it is rejected
  ("GLM5.2 native-MTP P/D handoff incomplete after 15.0s (full-page
  transfer)") and the client sees a 500 (the router runs
  `--disable-retries`). GSM8K-class load (sustained queue, long
  generations): c64 drove decode1 to a steady ~16 rejects/min (~24% of
  traffic, decode0 clean); c128 rejected ~83% of requests. One TP4 prefill
  cannot feed the handoff pipeline at those rates; decode1 (ranks 4..8)
  saturates before decode0. Sustainable envelope for that workload is
  ~c32. Raising the deadline only converts rejects into TTFT; the real
  lever is more prefill capacity (second P) or a faster save/publish
  pipeline.
- **vllm-bench random-IO sweeps through the router** (temperature 0,
  `ignore_eos`, 0 failed requests at every point; artifacts in
  `/mnt/shared/home/susun/bench-results/glm52-pd-*-20260730-165646.json`):
  - mid in=1,024/out=512: output throughput saturates past the knee —
    1,590 tok/s at c32 (TPOT p50 18.5 ms, TTFT p50 267 ms) versus 1,629
    tok/s at c64 (TPOT 37.5 ms, TTFT p99 7.4 s). c1: 164 tok/s, TPOT 5.1 ms.
  - prefill-heavy in=4,096/out=128: total (in+out) throughput plateaus at
    ~14.1k tok/s by c16–c64; TTFT p50 degrades 466 ms (c4) → 3.6 s (c16) →
    17.8 s (c64, queue-bound).
  - decode-heavy in=256/out=2,048: single stream 187 tok/s out (TPOT
    4.7 ms, native MTP on); fleet output ~1,640 tok/s at c64 with TPOT p50
    38 ms and TTFT p99 72 s. All 512 prompts completed.
  - The same c64 concurrencies that flooded GSM8K produced only ~2 rejects
    per bench run: the deadline binds under *sustained* long-generation
    load, not short random-IO bursts.
- **EP16/EP32 decode-scale attempt: blocked by cluster contention.** A
  4-node k3 job took tray09–12 and other tenants took tray04/08 within
  minutes of the free-machine scan (decode2/3 died on
  `CUDA_ERROR_OUT_OF_MEMORY` at `W13Weight` alloc). EP16 fleet was torn
  down; tray01/02/06 remain provisioned (`openinfer-ep32-decode`, NCCL
  2.30.7 checked) for the next idle window. `scripts/glm52_pd_stack.sh`
  supports `D_TOPO=ep16/ep32` + `decode-only` for that rerun.

### Multi-process decode fleet: TP4 P → EP8 D across two trays (2026-07-30)

- First hardware run of native P/D against a **multi-process** decode fleet:
  P TP4 on tray03; D EP8 split tray13 (ranks 0..4) + tray14 (ranks 4..8)
  under `--glm52-rendezvous`; metaserver on tray03:50056; transfers over
  `mlx5_bond_0`; vLLM router v0.1.15 on tray03:10001. The bring-up used
  `scripts/glm52_pd_stack.sh`, extended for multi-process decode
  (`D_TOPO`/`D_HOSTS`, rendezvous gating, ssh-proxied health probes).
- **148-token gate byte-identical on both decode endpoints** (tray13
  handoff 194 ms vs 660 ms local baseline; tray14 220 ms vs 683 ms). The
  4,096-token/64-page gate is byte-identical on tray13 (104.5 ms vs
  14,607 ms local baseline, ~140×). Anchor, drafts, and tail key match the
  single-process EP4 run bit-for-bit.
- Both decode nodes restore independently: admissions logged
  `first_step=verify` with `committed_len=148/4096` on rank=1 (tray13) and
  rank=5 (tray14) — each D node registers only its local arenas and pulls
  rank-locally, hardware-verifying the free-running claim that cross-node
  KV offload needs no central arena registry.
- The 4K first verify round on EP8 D accepted **5 of 5 drafts plus the
  bonus token** (the single-process EP4 run took 4 of 5) — first observed
  full-proposal acceptance.
- Router-driven smoke through the P/D loop passes (`ok: true`, TTFT
  117 ms).

### Post-shell-split replay: free-running DP architecture (2026-07-30)

- Context: the DP coordinator was split into per-rank autonomous engines
  (`free-running-dp.md` migration step 2, branch
  `feat/glm52-free-running-dp-gates`, head `22d7d047`). The full P/D chain was
  replayed on that code to confirm the handoff contract survived the
  restructure: TP4 P on tray03 + EP4 D on tray04 + metaserver over
  `mlx5_bond_0` (RoCE), native MTP on both sides.
- **148-token gate: byte-identical.** P → D first 161 ms versus EP4
  local-prefill TTFT 680 ms.
- **4,096-token gate (64 pages): byte-identical.** The first verify round
  accepted 4 of the 5 forwarded drafts plus the bonus token (5 tokens total,
  `mean_accepted_drafts=4.000`); D log confirmed `first_step=verify`. P → D
  first 104 ms versus EP4 local-prefill TTFT 15,339 ms (~147×).
- One 75-token ambiguous prompt produced a deterministic fork (` point` vs
  ` paragraph`) between the P/D and local-prefill paths. This is the top-1
  near-tie class already declared in `native-mtp-accuracy.md` (logit-margin
  level, movable by bucket/topology numerics), not a handoff defect; all
  non-near-tie prompts matched byte-for-byte.
- This replay settles the earlier outstanding item ("hardware first-verify
  telemetry must be rerun after the state fix"): post-split, first-verify
  admission and initial-proposal acceptance are confirmed on hardware.

### Post-review first-verify and context-cap fixes

- Native D admission now marks the forwarded anchor as the current decode
  token, with the prompt fully fed and D-side completion still zero. The
  installed five-token proposal therefore makes the first target span
  speculative instead of a `PrefillBoundary` that clears the drafts.
- TP4 native-MTP prefill now reserves the four extra positions used after
  draft-1 by the fixed five-token proposal loop. Requests above
  `max_model_len - 4` are rejected at intake instead of failing the prefill
  engine inside proposal generation. Plain TP4 prefill keeps its original
  context limit.
- Release validation: the two focused regression tests passed, followed by
  the full GLM5.2 library suite (`97 passed`, `21 ignored`). The historical
  hardware TTFT rows remain useful transfer/latency measurements, but their
  initial-proposal acceptance claim is withdrawn until the EP4 handoff is
  rerun.

### TP4 P → EP4 D hardware handoff

- On the prefill node, TP4 prefill registered four rank-local PegaFlow
  instances with 101 arenas each under
  `openinfer-glm52-l78-p64-mla656-idxk132-mtp1`. A five-token prompt returned
  target text ` Paris`, five draft IDs `[13, 576, 3283, 374, 1112]`,
  `committed_len=5`, `tail_len=5`, and a non-null 128-bit partial-page key.
- On the decode node, EP4 decode restored that page over RDMA from the prefill
  node's endpoint: one block, 101 arena slots, 3.3 MiB. Admission reported
  `cached_tokens=5` and
  `first_step=verify`; the six-token continuation
  `. Distance from Paris to Lyon` was byte-identical to an EP4 local-prefill
  baseline.
- The first cross-page gate found a producer-side bug before transfer:
  prompts longer than one 64-token page made the redundant small-M MTP
  boundary recomputation produce all-`-inf` argmax values and fail-stop the
  TP4 P engine. Both repetitive and ordinary prose reproduced it, so this is
  a page-boundary defect rather than an adversarial-text artifact.
- Staged finite-value logging localized the first invalid value to small-M MTP
  attention: `prepare` was finite, the historical MLA and index-K pages were
  populated, and the full indexer selected the correct physical slots, but
  attention returned NaN from element zero.
- Root cause: TP4's local FlashInfer decode backend consumes a statically
  quantized 576-byte cache row, while the P/D wire contract deliberately
  persists D-compatible `fp8_ds_mla` rows at 656 bytes. The original proposal
  loop interpreted the 656-byte transferable cache with 576-byte strides. It
  could appear to work inside the first page, then accumulated enough address
  error to fail at the first cross-page prompt.
- Fix: TP4 P owns two MTP MLA views. The 656-byte cache remains authoritative
  for PegaFlow/RDMA; a local 576-byte FlashInfer cache exists only for draft
  proposal. The large-M MTP pass fills both from the same raw layer-78 state,
  and synchronizes the layout-compatible index-K cache before proposal.
  Speculative writes and partial-page backup/restore touch only the local
  proposal view, so unverified state never contaminates transferable pages.
  Startup logs state both contracts explicitly, for example
  `execution_backend=FlashInferFp8 execution_bytes/token=576
  wire_layout=fp8_ds_mla wire_bytes/token=656`.
- The repaired 89-token gate returned target text `We`, drafts
  `[1184, 311, 387, 63141, 382]`, `committed_len=89`, `tail_len=25`, and 101
  arenas. EP4 D fetched one full block and the partial tail separately over
  RDMA (101 slots and 3.3 MiB each), admitted with `first_step=verify`, and
  accepted one draft in its first verify round. Its six-token result
  ` need to answer: "In` exactly matched a no-handoff EP4 baseline.
- After removing the staged finite-value diagnostics, the same TP4 producer
  gate was repeated with no debug environment variables and the normal
  CUDA-Graph-enabled server path. It returned the same target token, five
  drafts, committed length, arena count, and tail length.
- Completion text is not always a lossless way to construct the D request:
  appending the decoded anchor `We` to a prompt ending in `.` retokenized the
  pair as one `.We` token. The successful gate sent the original 89 prompt
  token IDs plus the anchor token ID, preserving the required
  `prompt = committed KV + anchor` length of 90. A router must forward token
  identity, not reconstruct the handoff boundary by concatenating text.

### Native P/D versus EP4 local-prefill TTFT

The A/B used a TP4 prefill node and EP4 decode node over the bonded RDMA
interface, with
native MTP enabled on both sides, a 4,096-token P chunk, and a 16,384-token
context cap. Each row is five deterministic, non-prefix-sharing token-ID
prompts after shape warmup at concurrency one and temperature zero. D was
restarted between the local-prefill and P/D phases so the P/D phase could not
hit baseline HBM or host cache. The 16K row uses 16,379 input tokens; its D
request asks for five rather than six output tokens to remain inside the
context cap.

`P first` is the first target token that a router can stream immediately.
`D handoff` starts when the router sends D the committed prompt, anchor,
proposal, and transfer metadata; it includes remote restore, admission, and
the first verify step. `P → D first` is the conservative latency if a router
waits for D's first new token before exposing any output.

| Input | EP4 local TTFT p50 | P first p50 | D handoff p50 | P → D first p50 | Combined delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 89 | 317.35 ms | 47.76 ms | 34.79 ms | 82.43 ms | −74.0% |
| 4,096 | 13,853.16 ms | 309.72 ms | 75.90 ms | 385.61 ms | −97.2% |
| 16,379 | 55,932.50 ms | 1,290.94 ms | 175.30 ms | 1,465.38 ms | −97.4% |

This is not evidence that disaggregation makes an optimized EP prefill
kernel faster: EP4 local prefill currently feeds the prompt through the
decode-oriented path. It does show the deployment-relevant result that TP4
large-M prefill plus transfer is dramatically cheaper than asking this EP4 D
worker to prefill locally. The additional latency after P has produced the
first token is 34.79/75.90/175.30 ms p50 for 89/4K/16K.

The transfer telemetry explains only part of that extra latency. At 4K, D
restored 63 full blocks (210.4 MiB across 101 arenas) in 11.2–11.7 ms, then
one 3.3 MiB explicit tail in 0.4–0.5 ms. At 16K, it restored 255 full blocks
(851.8 MiB) in 44.1–45.1 ms, then the tail in 0.4–1.1 ms. Every measured
request admitted with five drafts and `first_step=verify`.

The sweep found and fixed two long-context boundary bugs before producing the
table:

- **Exactly aligned committed length:** 4,096 initially advertised
  `tail_len=0`. PegaFlow fetched all 63 lineage-hashed full blocks in about
  11 ms, but D could only rematch 4,032 tokens and rejected after the strict
  15-second deadline. kvbm requires a dangling token after a sealed hash, so
  the final committed page remains an explicit tail even when it contains 64
  tokens. Native P/D now computes tail length in `1..=64`; the 4K metadata
  carries `tail_len=64` and a tail key. The
  `native_pd_tail_keeps_the_last_aligned_page_explicit` regression test covers
  both sides of the boundary.
- **Context-cap block table:** the first 16K P run panicked while copying 257
  page IDs into a 256-entry MTP attention table. kvbm may eagerly own one
  dangling generation page at the cap, but no valid position can address it.
  MTP now validates the requested logical page and copies at most the
  max-model-length table width. A 16,379-token TP4 P → EP4 D hardware gate and
  all five measured runs pass after the fix.

### Step 1: audit native-MTP physical page identity

- Created `feat/glm52-pd-mtp-arenas`.
- A first 101-arena registration passed the release library suite, but deeper
  inspection found it was semantically invalid and it was reverted before
  commit: target cache pages use BlockPool physical IDs, while native MTP
  currently uses fixed `1 + slot * pages_per_slot` IDs. Registering both under
  one content hash would therefore attach unrelated MTP bytes to a target
  page.
- Correct page-first storage needs two MTP regions: committed rows addressed
  by the target request's BlockPool page table, plus per-slot speculative
  scratch pages for proposal rows that target verification has not allocated
  or committed yet. A proposal crossing a 64-token boundary makes the scratch
  separation mandatory.
- Restore must also install MTP committed lengths; cache bytes alone do not
  establish continuity.
- Validation:
  - `cargo fmt --all -- --check` passed.
  - `cargo test --release -p openinfer-glm52 --lib` passed in the development
    container
    with NCCL 2.30 from the installed Python wheel: 88 passed, 21
    GPU-dependent tests ignored.

### Implementation state

- The committed half of the page-addressing refactor is implemented:
  `RequestKv::current_page_indices` exposes only pages covering committed KV;
  the scheduler attaches that table to every MTP context append; layer 78's
  first pass now writes through those target BlockPool IDs. Focused release
  MTP scheduler tests pass.
- Finish the other half by replacing proposal-time fixed-slot addressing with
  explicit scratch pages. This is now implemented: two pages per slot live
  beyond the transferable BlockPool range; partial committed pages are copied
  before drafting, and aligned/unaligned boundary tests pass.
- The two layer-78 arenas now register only the committed allocation prefix,
  producing 101 transferable arenas while excluding proposal scratch.
- Make MTP committed length an explicit restore/install state, then enable
  the 101-arena PegaFlow path behind a native P/D contract.
- D-side reset/resume now derives the installed committed length from the
  first restored append position, after the layer-78 bytes have been restored
  under the same BlockPool pages.
- TP4 producer weight loading now admits native MTP: the resident pass loads
  layer-78 bookends/attention/router, and the existing topology-aware TP slice
  loader includes layer 78 for all routed/shared experts. Focused weight-plan
  tests pass.
- The producer execution boundary is now explicit: TP4 cannot reuse the EP8
  decode-MTP buckets. It needs a large-M context pass over every prefill row,
  using shifted prompt tokens (and the sampled anchor at the boundary), target
  final-normalized hidden rows, layer-78 attention/indexer cache writes, and
  the existing TP4 expert-slice MoE path.
- The large-M TP4 layer-78 context pass is implemented and hardware-validated
  on 4xGB300. A real prefill-only request returned `Paris`; the same pass ran
  during kernel preflight and request execution while writing the committed
  layer-78 MLA/index-K rows through target BlockPool page IDs.
- The prefill result now separates the target anchor from native-MTP proposal
  metadata. Layer 78's boundary residual goes through
  `shared_head.norm + vocabulary head` to produce draft-1, and all four TP
  ranks fail closed unless both the target token and draft-1 match. The
  4xGB300 request gate passes; the release library suite is 90 passed /
  21 GPU-dependent ignored.
- Draft-1 now continues through four scratch-page iterations to form the
  complete five-token initial proposal. The HTTP response carries versioned
  native-P/D metadata, and strict D admission restores all 101 arenas, seeds
  the proposal, and begins with speculative verification.
- The cleaned implementation passes the release library suite: 92 passed,
  zero failed, and 21 GPU-dependent tests ignored.
- Remaining deployment gate: repeat the same contract on a real EP16 D fleet.
  The arena geometry and MTP launch restrictions are topology-independent,
  but multi-node collective startup and end-to-end EP16 restore have not yet
  been exercised. A post-cleanup EP4 decode-node replay also stopped before
  serving
  at DeepEP `ncclDevCommCreate` with a system error, including with unlimited
  memlock; no request reached the cleaned D code in that attempt. The earlier
  successful cross-page EP4 handoff remains the functional evidence, while
  the machine-level NCCL initialization needs a separate rerun.

## Debrief

The transferable cache format and the producer's fastest local execution
format are different contracts. Keeping one buffer and relying on identical
logical dimensions hid a physical-stride mismatch until a page boundary.
Future P/D additions should log both wire and execution layouts at startup and
must either prove them byte-identical or make the conversion boundary explicit.

The post-review state bug also showed that a log describing the intended next
step is not evidence of the slot's actual span kind. First-verify gates must
assert the state transition or observed speculative span.

Next action: the post-shell-split EP4 replay and the full router-path
validation are done (see the 2026-07-30 sections above). Two follow-ups
remain: (1) rerun the token-ID handoff contract against an EP16/EP32 D
fleet when four or more decode trays are simultaneously idle (tray01/02/06
stay provisioned); (2) if c64-class sustained load matters, attack the 15 s
handoff ceiling with a second prefill instance or faster save/publish —
raising `REMOTE_FETCH_DEADLINE` alone only converts rejects into TTFT.
